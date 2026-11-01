import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, s=1.0, angle_deg=0.0):
        """Return 6 vertices of a regular hexagon centered at (cx, cy) with side s, rotated by angle_deg."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ])

    def sat_min_separation(v1, v2):
        """Return minimum separation between two convex hexagons using SAT. Negative if overlapping."""
        min_sep = float('inf')
        for poly in (v1, v2):
            for i in range(6):
                j = (i + 1) % 6
                ex, ey = poly[j] - poly[i]
                nx, ny = -ey, ex
                nl = np.hypot(nx, ny)
                if nl < 1e-12:
                    continue
                nx, ny = nx / nl, ny / nl
                proj1 = v1 @ np.array([nx, ny])
                proj2 = v2 @ np.array([nx, ny])
                # Separation along this normal
                sep = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                min_sep = min(min_sep, sep)
        return min_sep

    def all_disjoint(positions, angles, tol=1e-6):
        """Check all pairs are disjoint using SAT."""
        verts = [hex_vertices(*p, s=1.0, angle_deg=a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i + 1, N):
                if sat_min_separation(verts[i], verts[j]) < -tol:
                    return False
        return True

    def required_outer_side(positions, angles):
        """Compute minimum outer hexagon side length (axis-aligned, centered at origin) to contain all inner hexagons."""
        all_v = np.vstack([hex_vertices(*p, s=1.0, angle_deg=a) for p, a in zip(positions, angles)])
        x, y = all_v[:, 0], all_v[:, 1]
        s3 = np.sqrt(3.0)
        # The six half-width constraints for a regular hexagon with side s, flat-top, centered at origin
        # A point (x,y) is inside if:
        #   |x| <= s * (2/3) * something... Actually let me think about this more carefully.
        # For a regular hexagon with side length S, flat-top, centered at origin:
        # The six vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance S from center.
        # Wait, for flat-top, vertices are at 0, 60, ... degrees? Let me reconsider.
        # A regular hexagon with side S has circumradius R = S.
        # For flat-top orientation, vertices are at angles 0°, 60°, 120°, 180°, 240°, 300°.
        # The support function in direction θ gives the maximum dot product.
        # To contain all points, we need the maximum support in each of the 6 edge normals to be <= S.
        # The edge normals for a flat-top hexagon point in directions 30°, 90°, 150°, 210°, 270°, 330°.
        # Actually, let me use the half-widths:
        # For flat-top hex with side S:
        #   max |x| = S
        #   max |y| = S * sqrt(3)/2
        #   But it's more nuanced. The constraint is:
        #   For each of the 6 edge normals n_k (unit vectors at angles 30+60k degrees),
        #   max over all points p of (p · n_k) <= S * (3/2) / |n_k|... 
        # Let me just compute it directly: the half-width in direction of each edge normal.
        # Edge normals for flat-top hex: directions at 30°, 90°, 150°, 210°, 270°, 330°
        # The distance from center to each edge is the apothem = S * sqrt(3)/2
        # So for containment, we need: max_p (p · n_k) <= S * sqrt(3)/2 for all 6 normals
        # Therefore S >= max over all k of (max_p (p · n_k)) / (sqrt(3)/2)
        
        normals = []
        for k in range(6):
            ang = np.radians(30 + 60 * k)
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        max_proj = np.max(all_v @ normals.T, axis=0)  # max projection in each normal direction
        s3 = np.sqrt(3.0)
        return np.max(max_proj) / (s3 / 2.0)

    def objective(params):
        """Objective: outer side length + penalty for overlaps."""
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        side = required_outer_side(positions, angles)
        
        # Penalty for overlaps
        verts = [hex_vertices(*p, s=1.0, angle_deg=a) for p, a in zip(positions, angles)]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_min_separation(verts[i], verts[j])
                if sep < 0:
                    pen += (-sep) ** 2
        
        return side + 10.0 * pen

    def objective_smooth(params):
        """Smoothed objective using signed distance approximation."""
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        side = required_outer_side(positions, angles)
        
        # Use vertex-to-vertex minimum distance as a proxy for separation
        verts = [hex_vertices(*p, s=1.0, angle_deg=a) for p, a in zip(positions, angles)]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Minimum distance between vertices
                d = np.min([np.linalg.norm(a - b) for a in verts[i] for b in verts[j]])
                # For two unit hexagons, they are tangent when center distance = 2*sqrt(3) (for parallel orientation)
                # Use a smooth penalty
                if d < 1.5:
                    pen += (1.5 - d) ** 2
        
        return side + 5.0 * pen

    # Initial configurations
    def init_hex_lattice():
        """Hexagonal close-packing layout."""
        s3 = np.sqrt(3)
        positions = np.array([
            [0, 0],
            [s3, 0], [-s3, 0],
            [s3/2, 1.5], [-s3/2, 1.5],
            [s3/2, -1.5], [-s3/2, -1.5],
            [s3*1.5, 0.75], [s3*1.5, -0.75],
            [-s3*1.5, 0.75], [-s3*1.5, -0.75],
            [0, s3]
        ])
        angles = np.zeros(N)
        return positions, angles

    def init_circular():
        """Circular arrangement."""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            a = 2 * np.pi * i / N
            r = 2.8
            positions[i] = [r * np.cos(a), r * np.sin(a)]
            angles[i] = a * 180 / np.pi  # Orient each hexagon radially
        return positions, angles

    def init_random(seed=42):
        np.random.seed(seed)
        positions = np.random.uniform(-4, 4, (N, 2))
        angles = np.random.uniform(0, 60, N)
        return positions, angles

    inits = [
        init_hex_lattice(),
        init_circular(),
        init_random(1),
        init_random(7),
        init_random(99),
        init_random(123),
    ]

    best_params = None
    best_side = float('inf')
    best_positions = None
    best_angles = None

    # Local optimization from multiple starts
    for idx, (pos0, ang0) in enumerate(inits):
        if time.time() > deadline - 25:
            break

        # Ensure initial is valid
        params0 = np.concatenate([pos0.flatten(), ang0])
        
        # Try to find a valid nearby starting point
        valid = all_disjoint(pos0, ang0, 1e-4)
        if not valid:
            # Scale up to make valid
            for scale in [1.2, 1.5, 2.0, 2.5]:
                pos0_s = pos0 * scale
                if all_disjoint(pos0_s, ang0, 1e-4):
                    pos0 = pos0_s
                    params0 = np.concatenate([pos0.flatten(), ang0])
                    valid = True
                    break
            if not valid:
                continue

        # Run local optimization
        try:
            res = minimize(
                objective, params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            params = res.x
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            
            if all_disjoint(positions, angles, 1e-5):
                side = required_outer_side(positions, angles)
                if side < best_side:
                    best_side = side
                    best_params = params.copy()
                    best_positions = positions.copy()
                    best_angles = angles.copy()
        except Exception:
            continue

    # Simulated annealing refinement
    if best_positions is not None and time.time() < deadline - 15:
        cur_pos = best_positions.copy()
        cur_ang = best_angles.copy()
        cur_side = best_side
        T = 0.3
        
        np.random.seed(42)
        for it in range(30000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(N)
            
            # Random perturbation
            if np.random.random() < 0.7:
                # Move position
                dx, dy = np.random.normal(0, 0.25, 2)
                new_pos = cur_pos.copy()
                new_pos[idx] += [dx, dy]
                new_ang = cur_ang.copy()
            else:
                # Change angle
                new_pos = cur_pos.copy()
                new_ang = cur_ang.copy()
                new_ang[idx] = (cur_ang[idx] + np.random.uniform(-15, 15)) % 60
            
            # Check validity
            if not all_disjoint(new_pos, new_ang, 1e-5):
                continue
            
            ns = required_outer_side(new_pos, new_ang)
            d = ns - cur_side
            if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-8)):
                cur_pos = new_pos
                cur_ang = new_ang
                cur_side = ns
                if cur_side < best_side:
                    best_side = cur_side
                    best_positions = cur_pos.copy()
                    best_angles = cur_ang.copy()
            T *= 0.9999

    # Fallback
    if best_positions is None:
        best_positions = np.array([[i * 3.0, 0.0] for i in range(N)], dtype=float)
        best_angles = np.zeros(N)
        best_side = required_outer_side(best_positions, best_angles)

    # Final validity check
    if not all_disjoint(best_positions, best_angles, 1e-4):
        # Last resort: spread out widely
        best_positions = np.array([[i * 4.0, 0.0] for i in range(N)], dtype=float)
        best_angles = np.zeros(N)
        best_side = required_outer_side(best_positions, best_angles)

    final_side = required_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    return inner, (0.0, 0.0), float(final_side), 0.0
