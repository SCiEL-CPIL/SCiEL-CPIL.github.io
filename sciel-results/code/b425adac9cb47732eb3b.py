import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 148.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _sdf_hex_point(p, center, side, angle_deg):
    """
    Signed distance from point p to a regular hexagon.
    Positive if outside, negative if inside.
    Uses the standard SDF for a regular polygon via sector mapping.
    """
    x, y = p
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    
    # Transform to hexagon local frame
    dx = x - cx
    dy = y - cy
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # Hexagon with side 's' has apothem a = s * sqrt(3)/2
    # In local frame, the hexagon is defined by 6 half-planes.
    # Due to 6-fold symmetry, we can map to one sector (0 to 60 degrees).
    
    s = side
    apothem = s * np.sqrt(3) / 2
    
    # Map to first sector: angle in [0, 60]
    # The hexagon vertices in local frame (angle=0) are at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # Use the standard SDF formula for regular polygons
    # For a regular n-gon with inradius r (apothem):
    # 1. Compute angle and radius in local frame
    # 2. Map to [0, 2*pi/n)
    # 3. Compute distance to the edge of that sector
    
    n = 6
    r = apothem  # inradius
    
    # Angle and radius
    phi = np.arctan2(ly, lx)
    rho = np.sqrt(lx*lx + ly*ly)
    
    # Map to [0, 2*pi/n)
    sector = 2 * np.pi / n
    phi_mod = phi % sector
    # Center of sector is at sector/2
    phi_center = phi_mod - sector / 2
    
    # Distance to the edge of the sector (the perpendicular to the edge)
    # The edge is at distance r from center, and the angle from sector center to edge normal is 0
    # So the distance from point to the edge line is: r - rho * cos(phi_center)
    d_edge = r - rho * np.cos(phi_center)
    
    # But we also need to check if the point is within the angular range of the sector
    # If not, the closest point is a vertex
    # The vertex is at angle sector/2 from the edge normal, at distance r / cos(sector/2) = s (circumradius)
    
    # Actually, let's use a simpler approach: compute distance to each of the 6 edges and take min
    # This is more robust and still differentiable
    
    # The 6 edge normals in local frame (for angle=0 hexagon)
    # Edge k is perpendicular to direction (k * 60 degrees)
    # The edge is at distance r from center along the normal
    
    min_d = np.inf
    for k in range(n):
        angle_k = k * sector
        nx = np.cos(angle_k)
        ny = np.sin(angle_k)
        # Signed distance to the half-plane: n . p - r
        d = nx * lx + ny * ly - r
        if d < min_d:
            min_d = d
    
    # This gives the signed distance to the intersection of half-planes (the hexagon)
    # Positive means outside, negative means inside
    return min_d

def _sdf_hex_hex(v1, v2):
    """
    Compute a smooth measure of separation between two hexagons.
    Returns a positive value if separated, negative if overlapping.
    Uses the minimum of SDFs of vertices of one hexagon to the other.
    """
    # For each vertex of v1, compute SDF to v2, take min
    # For each vertex of v2, compute SDF to v1, take min
    # The separation is the max of these two mins (if both positive, they're separated)
    
    # Extract center, side, angle from vertices
    # This is tricky. Let's pass parameters instead.
    
    # Actually, let's compute this differently.
    # We'll compute the minimum distance between the two hexagons using a smooth approximation.
    
    # For now, use the vertex-to-SDF approach
    # But we need the parameters, not just vertices.
    
    # Let's restructure: pass centers, angles, and use SDF directly
    pass

def _smooth_penalty(centers, angles, outer_center, outer_side, outer_angle):
    """
    Compute a smooth penalty for:
    1. Inner hexagons overlapping each other
    2. Inner hexagons outside the outer hexagon
    
    Returns (penalty_value, penalty_grad) where grad is w.r.t. [centers, angles]
    """
    n = len(centers)
    penalty = 0.0
    grad = np.zeros_like(centers)  # 2N
    angle_grad = np.zeros(n)
    
    # Helper: SDF of point to hexagon with given center, angle
    def sdf_pt_to_hex(px, py, hx, hy, ha):
        theta = np.deg2rad(ha)
        dx = px - hx
        dy = py - hy
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t
        
        s = 1.0
        apothem = s * np.sqrt(3) / 2
        n_hex = 6
        sector = 2 * np.pi / n_hex
        
        min_d = np.inf
        for k in range(n_hex):
            angle_k = k * sector
            nx = np.cos(angle_k)
            ny = np.sin(angle_k)
            d = nx * lx + ny * ly - apothem
            if d < min_d:
                min_d = d
        return min_d
    
    # 1. Inner-inner overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            # Compute vertices of hex i
            ci = centers[i]
            ai = angles[i]
            theta_i = np.deg2rad(ai)
            verts_i = np.array([
                [ci[0] + np.cos(theta_i + k * np.pi/3), ci[1] + np.sin(theta_i + k * np.pi/3)]
                for k in range(6)
            ])
            
            # SDF of each vertex of i to hex j
            min_sdf = np.inf
            for v in verts_i:
                d = sdf_pt_to_hex(v[0], v[1], centers[j][0], centers[j][1], angles[j])
                if d < min_sdf:
                    min_sdf = d
            
            # Also check vertices of j to hex i
            cj = centers[j]
            aj = angles[j]
            theta_j = np.deg2rad(aj)
            verts_j = np.array([
                [cj[0] + np.cos(theta_j + k * np.pi/3), cj[1] + np.sin(theta_j + k * np.pi/3)]
                for k in range(6)
            ])
            
            min_sdf_j = np.inf
            for v in verts_j:
                d = sdf_pt_to_hex(v[0], v[1], centers[i][0], centers[i][1], angles[i])
                if d < min_sdf_j:
                    min_sdf_j = d
            
            # The separation is the max of the two min SDFs
            # If both are positive, they're separated
            # If either is negative, they overlap
            sep = max(min_sdf, min_sdf_j)
            
            if sep < 0:
                # Overlapping: penalty is quadratic in the overlap depth
                penalty += sep * sep
                # Gradient: d(penalty)/d(sep) = 2*sep
                # d(sep)/d(centers[i]) and d(sep)/d(centers[j])
                # This is complex. For now, use a finite-difference approximation or
                # just push them apart based on the vector between centers.
                
                # Simple gradient: push i away from j and vice versa
                dvec = centers[i] - centers[j]
                dnorm = np.linalg.norm(dvec)
                if dnorm > 1e-9:
                    dhat = dvec / dnorm
                    # Gradient of -sep (we want to increase sep)
                    # Approximate: the gradient points in the direction that increases separation
                    grad[i] += 2 * sep * dhat  # Push i away from j
                    grad[j] -= 2 * sep * dhat  # Push j away from i
                else:
                    # If centers coincide, use a random direction
                    rand_dir = np.array([np.cos(np.random.rand() * 2 * np.pi), 
                                         np.sin(np.random.rand() * 2 * np.pi)])
                    grad[i] += 2 * sep * rand_dir
                    grad[j] -= 2 * sep * rand_dir
    
    # 2. Containment penalty: inner hexagons must be inside outer hexagon
    # The outer hexagon has center outer_center, side outer_side, angle outer_angle
    # For each vertex of each inner hexagon, check if it's inside the outer hexagon
    # Use SDF: the SDF of a point to the outer hexagon should be <= 0 (inside)
    
    oc = outer_center
    os = outer_side
    oa = outer_angle
    
    for i in range(n):
        ci = centers[i]
        ai = angles[i]
        theta_i = np.deg2rad(ai)
        verts_i = np.array([
            [ci[0] + np.cos(theta_i + k * np.pi/3), ci[1] + np.sin(theta_i + k * np.pi/3)]
            for k in range(6)
        ])
        
        for v in verts_i:
            # SDF of v to the outer hexagon
            # The outer hexagon has side os, so its apothem is os * sqrt(3)/2
            theta_o = np.deg2rad(oa)
            dx = v[0] - oc[0]
            dy = v[1] - oc[1]
            cos_t = np.cos(-theta_o)
            sin_t = np.sin(-theta_o)
            lx = dx * cos_t - dy * sin_t
            ly = dx * sin_t + dy * cos_t
            
            apothem_o = os * np.sqrt(3) / 2
            n_hex = 6
            sector = 2 * np.pi / n_hex
            
            min_d = np.inf
            for k in range(n_hex):
                angle_k = k * sector
                nx = np.cos(angle_k)
                ny = np.sin(angle_k)
                d = nx * lx + ny * ly - apothem_o
                if d < min_d:
                    min_d = d
            
            # min_d is the signed distance: negative if inside, positive if outside
            # We want min_d <= 0, so penalty is max(0, min_d)^2
            if min_d > 0:
                penalty += min_d * min_d
                # Gradient w.r.t. centers[i]
                # d(min_d)/d(v) = the normal of the closest edge
                # d(v)/d(centers[i]) = I (identity)
                # So grad[i] += 2 * min_d * normal_of_closest_edge
                
                # Find the closest edge normal
                theta_o2 = np.deg2rad(oa)
                dx2 = v[0] - oc[0]
                dy2 = v[1] - oc[1]
                cos_t2 = np.cos(-theta_o2)
                sin_t2 = np.sin(-theta_o2)
                lx2 = dx2 * cos_t2 - dy2 * sin_t2
                ly2 = dx2 * sin_t2 + dy2 * cos_t2
                
                best_d = np.inf
                best_normal = np.array([0.0, 0.0])
                for k in range(n_hex):
                    angle_k = k * sector
                    nx = np.cos(angle_k)
                    ny = np.sin(angle_k)
                    d = nx * lx2 + ny * ly2 - apothem_o
                    if d < best_d:
                        best_d = d
                        # Transform normal back to world frame
                        cos_t3 = np.cos(theta_o2)
                        sin_t3 = np.sin(theta_o2)
                        best_normal = np.array([
                            nx * cos_t3 - ny * sin_t3,
                            nx * sin_t3 + ny * cos_t3
                        ])
                
                grad[i] += 2 * min_d * best_normal
    
    return penalty, grad, angle_grad

def _objective(x, outer_angle):
    """
    Objective function for L-BFGS-B.
    x: 3N vector of inner hexagon parameters (x1, y1, a1, x2, y2, a2, ...)
    Returns: (value, grad)
    """
    params = x.reshape(N, 3)
    centers = params[:, :2]
    angles = params[:, 2]
    
    # Compute the required outer side length to contain all inner hexagons
    # given the outer angle
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            verts_i = np.array([
                [centers[i][0] + np.cos(theta_i + m * np.pi/3), 
                 centers[i][1] + np.sin(theta_i + m * np.pi/3)]
                for m in range(6)
            ])
            projs = np.dot(verts_i, normal)
            max_proj = max(max_proj, np.max(projs))
    
    side_length = max_proj * 2.0 / np.sqrt(3.0)
    
    # Compute penalty
    penalty, grad, angle_grad = _smooth_penalty(centers, angles, (0.0, 0.0), side_length, outer_angle)
    
    # Total objective: side_length + penalty
    value = side_length + penalty
    
    # Gradient
    grad_full = np.zeros_like(x)
    grad_full[:, :2] = grad
    # The gradient of side_length w.r.t. centers is complex.
    # For now, approximate: the gradient of side_length is the normal of the 
    # outermost projection direction, scaled appropriately.
    # This is a rough approximation that should work well enough for L-BFGS-B.
    
    # Find the direction of maximum projection
    max_proj_dir = None
    max_proj_val = -np.inf
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            verts_i = np.array([
                [centers[i][0] + np.cos(theta_i + m * np.pi/3), 
                 centers[i][1] + np.sin(theta_i + m * np.pi/3)]
                for m in range(6)
            ])
            projs = np.dot(verts_i, normal)
            if np.max(projs) > max_proj_val:
                max_proj_val = np.max(projs)
                max_proj_dir = normal
                max_proj_idx = i
    
    if max_proj_dir is not None and max_proj_idx is not None:
        # The gradient of side_length w.r.t. centers[max_proj_idx] is 
        # (2/sqrt(3)) * max_proj_dir
        grad_full[max_proj_idx, :2] += (2.0 / np.sqrt(3.0)) * max_proj_dir
    
    return value, grad_full

def _compute_side_length(centers, angles, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0) 
    to contain all inner hexagons, given the outer angle.
    """
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            theta_i = np.deg2rad(angles[i])
            verts_i = np.array([
                [centers[i][0] + np.cos(theta_i + m * np.pi/3), 
                 centers[i][1] + np.sin(theta_i + m * np.pi/3)]
                for m in range(6)
            ])
            projs = np.dot(verts_i, normal)
            max_proj = max(max_proj, np.max(projs))
    
    return max_proj * 2.0 / np.sqrt(3.0)

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness using SAT
    def _sat_overlap(v1, v2):
        axes = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal = normal / norm
                axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
                
    return True, "valid"

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # We will try a few outer angles
    outer_angles = [0.0, 30.0]
    
    # Generate initial guesses
    initial_guesses = []
    
    # Guess 1: Hexagonal ring (Center + 6 + 5)
    pts = [[0,0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle), 0.0])
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle), 0.0])
        if len(pts) == 12:
            break
    initial_guesses.append(np.array(pts).flatten())
    
    # Guess 2: Random
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.uniform(-3, 3, (N, 3))
        pts[:, 2] = np.random.choice([0, 30, 60, 90, 120, 150], N)
        initial_guesses.append(pts.flatten())
        
    # Guess 3: Triangular lattice
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            lattice_pts.append([x, y, 0.0])
    lattice_pts = np.array(lattice_pts)
    
    picked = []
    sorted_idx = np.argsort(np.sum(lattice_pts**2, axis=1))
    for idx in sorted_idx:
        p = lattice_pts[idx]
        valid = True
        for q in picked:
            if np.linalg.norm(p[:2] - q[:2]) < np.sqrt(3) * 0.99:
                valid = False
                break
        if valid:
            picked.append(p)
            if len(picked) == 12:
                break
    if len(picked) == 12:
        initial_guesses.append(np.array(picked).flatten())

    for outer_angle in outer_angles:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
            
        for guess in initial_guesses:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            try:
                # Use L-BFGS-B
                res = minimize(
                    _objective,
                    guess,
                    args=(outer_angle,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'gtol': 1e-6, 'ftol': 1e-8, 'maxfun': 5000}
                )
                
                x_opt = res.x
                params_opt = x_opt.reshape(N, 3)
                
                # Verify validity and compute exact side length
                inner_hexes = [[params_opt[i][0], params_opt[i][1], params_opt[i][2]] for i in range(N)]
                
                centers = np.array([h[:2] for h in inner_hexes])
                angles = np.array([h[2] for h in inner_hexes])
                
                side = _compute_side_length(centers, angles, outer_angle)
                
                is_valid, reason = _check_validity(inner_hexes, side, (0.0, 0.0), outer_angle)
                
                if is_valid and side < best_side:
                    best_side = side
                    best_params = inner_hexes
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
                    
            except Exception as e:
                pass
                
    if best_params is not None:
        return (best_params, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
