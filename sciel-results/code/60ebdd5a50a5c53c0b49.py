import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s

    def hex_vertices(cx, cy, side, angle_rad):
        """Return 6x2 array of vertices of a regular hexagon centered at (cx,cy)
        with the given side length and rotation angle in radians."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_outward_normals(verts):
        """Return 6x2 array of outward unit normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        # Outward normal for CCW polygon: (edge_y, -edge_x)
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def sat_penetration(v1, v2):
        """Compute minimum penetration depth between two convex polygons using SAT.
        Returns 0 if separated, positive value if overlapping."""
        normals1 = get_outward_normals(v1)
        normals2 = get_outward_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return max(0.0, min_overlap)

    def min_enclosing_hex_side(verts_all, theta):
        """Given all inner hexagon vertices and outer rotation theta,
        compute the minimum side length of a regular hexagon (rotated by theta,
        centered at origin) that encloses all points."""
        # Outer hexagon with side s, rotation theta, centered at origin
        # Vertices at angles: theta + 0, 60, ..., 300 degrees
        # The constraint is: for each of the 6 supporting lines,
        # the max projection of all points must be <= s * cos(0) = s * 1
        # Actually for a regular hexagon with side s, the distance from center
        # to each edge (apothem) is s * sqrt(3)/2
        # So the constraint is: max projection of all points along each of the 6
        # outward normals <= apothem = s * sqrt(3)/2
        # Therefore s >= 2/sqrt(3) * max_projection for each direction
        # And s >= max over all 6 directions
        
        # The 6 outward normals for outer hex with rotation theta
        # Edge normals are perpendicular to edges. For a flat-top hex (theta=0),
        # edges are at angles 0, 60, ... so normals are at 90, 150, 210, 270, 330, 30
        # Actually let me think more carefully.
        
        # A regular hexagon with side s, rotation theta:
        # Vertices at (s*cos(theta+k*pi/3), s*sin(theta+k*pi/3)) for k=0..5
        # The 6 edges connect consecutive vertices.
        # Edge k goes from vertex k to vertex k+1.
        # The outward normal of edge k points in direction theta + k*pi/3 + pi/6
        # Wait, let me verify: for theta=0 (pointy-top), vertices at angles 0,60,...
        # Edge from angle 0 to 60: midpoint at 30 deg, normal at 30 deg? 
        # Actually the outward normal of the edge between vertex at angle a and a+60
        # is in direction a+30. So normals are at theta+30, theta+90, theta+150, ...
        
        # Let me just compute the normals from the outer hex vertices
        outer_verts_template = hex_vertices(0, 0, 1.0, theta)
        outer_normals = get_outward_normals(outer_verts_template)
        
        # Project all inner vertices onto each of the 6 outward normals
        projections = verts_all @ outer_normals.T  # (num_points, 6)
        max_projections = projections.max(axis=0)  # (6,)
        
        # The apothem (distance from center to edge) of a unit hexagon is sqrt(3)/2
        # For side length s, apothem = s * sqrt(3)/2
        # So we need s * sqrt(3)/2 >= max_projection for each direction
        # s >= max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
        
        required_s = max_projections / (SQRT3 / 2.0)
        return np.max(required_s)

    def compute_all_penalties(positions, alphas, s, theta):
        """Compute total penalty for overlap and containment violations."""
        total_penalty = 0.0
        verts_list = []
        
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i])
            verts_list.append(verts)
        
        # Check pairwise disjointness
        for i in range(N):
            for j in range(i + 1, N):
                pd = sat_penetration(verts_list[i], verts_list[j])
                total_penalty += pd * pd * 100.0  # Quadratic penalty
        
        # Check containment
        if s > 0:
            # Compute the actual minimum required side
            all_verts = np.vstack(verts_list)
            required_s = min_enclosing_hex_side(all_verts, theta)
            if s < required_s:
                total_penalty += (required_s - s) * (required_s - s) * 100.0
        
        return total_penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        positions = params[:2 * N].reshape(N, 2)
        alphas = params[2 * N : 2 * N + N]
        s = params[2 * N + N]
        theta = params[2 * N + N + 1]
        
        if s < 1.5:
            return 1e6
            
        penalty = compute_all_penalties(positions, alphas, s, theta)
        return s + penalty

    def exact_min_side(positions, alphas, theta):
        """Compute the exact minimum outer side length for given positions/angles/rotation."""
        verts_list = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        return min_enclosing_hex_side(all_verts, theta)

    def validate(positions, alphas, s, theta):
        """Check if the configuration is valid (no overlaps, all inside)."""
        verts_list = [hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i]) for i in range(N)]
        
        # Check pairwise disjointness with strict tolerance
        for i in range(N):
            for j in range(i + 1, N):
                pd = sat_penetration(verts_list[i], verts_list[j])
                if pd > 1e-7:
                    return False
        
        # Check containment
        all_verts = np.vstack(verts_list)
        required_s = min_enclosing_hex_side(all_verts, theta)
        if s < required_s - 1e-9:
            return False
        
        return True

    def make_honeycomb(rows, cols, flat_top):
        """Generate honeycomb pattern positions, centered at origin."""
        pos = []
        if flat_top:
            dx = SQRT3
            dy = 1.5
            shift = SQRT3 / 2
            for r in range(rows):
                for c in range(cols):
                    x = c * dx
                    if r % 2 == 1:
                        x += shift
                    y = r * dy
                    pos.append([x, y])
        else:
            dx = 1.5
            dy = SQRT3
            shift = 0.75
            for r in range(rows):
                for c in range(cols):
                    y = c * dy
                    if r % 2 == 1:
                        y += shift
                    x = r * dx
                    pos.append([x, y])
        
        pos = np.array(pos)
        center = pos.mean(axis=0)
        pos = pos - center
        return pos

    def init_hex_cluster():
        """Generate a hexagonal cluster of 12 hexagons."""
        d1 = SQRT3
        d2 = 2 * SQRT3
        pos = [[0.0, 0.0]]
        alphas = [0.0]
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
            alphas.append(0.0)
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
            alphas.append(0.0)
        return np.array(pos), np.array(alphas)

    # Generate initial configurations
    pos_34 = make_honeycomb(3, 4, True)
    alphas_34 = np.full(N, np.radians(30))
    
    pos_43 = make_honeycomb(4, 3, False)
    alphas_43 = np.full(N, 0.0)
    
    pos_hc, alphas_hc = init_hex_cluster()
    
    # Also try rotated versions
    def rotate_positions(pos, angle):
        c, s_ = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        return pos @ rot.T

    inits = []
    for theta_outer in [0, np.radians(30)]:
        # 3x4 flat-top
        s0 = exact_min_side(pos_34, alphas_34, theta_outer)
        inits.append((pos_34, alphas_34, s0, theta_outer))
        
        # 4x3 pointy-top
        s0 = exact_min_side(pos_43, alphas_43, theta_outer)
        inits.append((pos_43, alphas_43, s0, theta_outer))
        
        # Hex cluster
        s0 = exact_min_side(pos_hc, alphas_hc, theta_outer)
        inits.append((pos_hc, alphas_hc, s0, theta_outer))
        
        # Rotated 3x4
        pos_rot = rotate_positions(pos_34, np.radians(30))
        s0 = exact_min_side(pos_rot, alphas_34, theta_outer)
        inits.append((pos_rot, alphas_34, s0, theta_outer))
        
        # Rotated 4x3
        pos_rot2 = rotate_positions(pos_43, np.radians(30))
        s0 = exact_min_side(pos_rot2, alphas_43, theta_outer)
        inits.append((pos_rot2, alphas_43, s0, theta_outer))

    best_s = np.inf
    best_params = None
    best_pos = None
    best_alphas = None
    best_theta = None

    # Multi-start optimization
    max_runs = 12
    for run in range(max_runs):
        if time.time() > deadline - 20:
            break
            
        idx = run % len(inits)
        init_pos, init_alphas, init_s, init_theta = inits[idx]
        
        # Add slight randomness for diversity in later runs
        if run >= len(inits):
            init_pos = init_pos + np.random.normal(0, 0.05, init_pos.shape)
            init_alphas = init_alphas + np.random.uniform(-0.05, 0.05, N)
            init_theta = init_theta + np.random.uniform(-0.05, 0.05)
        
        x0 = np.concatenate([
            init_pos.flatten(), 
            init_alphas, 
            [init_s], 
            [init_theta]
        ])
        
        try:
            # Phase 1: L-BFGS-B with moderate tolerance
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-4}
            )
            
            pos = res.x[:2 * N].reshape(N, 2)
            alphas = res.x[2 * N : 2 * N + N]
            s = res.x[2 * N + N]
            theta = res.x[2 * N + N + 1]
            
            # Compute exact minimum side for this configuration
            exact_s = exact_min_side(pos, alphas, theta)
            
            if validate(pos, alphas, exact_s, theta):
                if exact_s < best_s:
                    best_s = exact_s
                    best_params = np.concatenate([pos.flatten(), alphas, [exact_s], [theta]])
                    best_pos = pos.copy()
                    best_alphas = alphas.copy()
                    best_theta = theta
            else:
                # Try with the s from optimization
                if validate(pos, alphas, s, theta) and s < best_s:
                    best_s = s
                    best_params = res.x.copy()
                    best_pos = pos.copy()
                    best_alphas = alphas.copy()
                    best_theta = theta
                    
        except Exception:
            pass

    # Final polish phase
    if best_params is not None and time.time() < deadline - 5:
        try:
            # Use Nelder-Mead for fine-tuning
            res_polish = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            pos = res_polish.x[:2 * N].reshape(N, 2)
            alphas = res_polish.x[2 * N : 2 * N + N]
            theta = res_polish.x[2 * N + N + 1]
            
            exact_s = exact_min_side(pos, alphas, theta)
            
            if validate(pos, alphas, exact_s, theta) and exact_s < best_s:
                best_s = exact_s
                best_params = np.concatenate([pos.flatten(), alphas, [exact_s], [theta]])
                best_pos = pos.copy()
                best_alphas = alphas.copy()
                best_theta = theta
        except Exception:
            pass

    # If no valid solution found, fallback
    if best_params is None:
        pos, alphas, s, theta = inits[0]
        best_s = s
        best_params = np.concatenate([pos.flatten(), alphas, [s], [theta]])
        best_pos = pos.copy()
        best_alphas = alphas.copy()
        best_theta = theta

    # Extract final positions
    final_pos = best_pos
    final_alphas = best_alphas
    final_s = best_s
    final_theta = best_theta
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_alphas[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
