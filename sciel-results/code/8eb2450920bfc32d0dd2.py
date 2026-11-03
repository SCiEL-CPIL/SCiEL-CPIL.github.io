import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 110.0  # Safety margin

# Precompute unit hexagon vertices (flat-top, angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for the unit hexagon (pointing outward)
# For a flat-top hexagon, edges are at 0, 60, 120, 180, 240, 300 degrees from the x-axis?
# No, vertices are at 0, 60, ...
# Edge 0 connects V0 to V1. Normal is perpendicular to edge, pointing out.
# V0=(1,0), V1=(0.5, sqrt(3)/2). Edge vector = (-0.5, sqrt(3)/2). Normal = (sqrt(3)/2, 0.5) -> angle 30 deg.
# So normals are at 30, 90, 150, 210, 270, 330 degrees.
_NORMAL_ANGLES = np.radians(30 + np.arange(6) * 60)
_EDGE_NORMALS = np.column_stack((np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)))

def construct():
    t0 = time.time()
    
    sqrt3 = math.sqrt(3)
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _nx = _EDGE_NORMALS[:, 0]
    _ny = _EDGE_NORMALS[:, 1]

    def compute_side_and_center(positions, angles_deg):
        """
        Computes the minimum enclosing regular hexagon side length and center.
        Returns (side, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        # Rotate vertices
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        
        # Translate
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        verts_flat_x = verts_x.reshape(-1)
        verts_flat_y = verts_y.reshape(-1)
        
        center_x = np.mean(verts_flat_x)
        center_y = np.mean(verts_flat_y)
        
        ox = verts_flat_x - center_x
        oy = verts_flat_y - center_y
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # The enclosing hexagon has normals at phi + 30, phi + 90, ...
            # Wait, the outer hexagon rotation 'angle' usually defines its vertices or edges?
            # Task says "outer side length and rotation".
            # If outer hexagon has rotation angle A, its vertices are at A, A+60...
            # Its supporting lines (normals) are at A+30, A+90...
            # The width of the packing in direction normal_n must be <= side * sqrt(3).
            # So side >= width / sqrt(3).
            # We want to minimize max(widths).
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj_x = ox[:, None] * nx[None, :]
            proj_y = oy[:, None] * ny[None, :]
            proj = proj_x + proj_y
            
            # Width is max - min
            widths = np.max(proj, axis=0) - np.min(proj, axis=0)
            
            # Side length s such that width <= s * sqrt(3)
            # So s = width / sqrt(3)
            # We need s to satisfy all 6 directions.
            # So s = max(widths) / sqrt(3)
            return np.max(widths) / sqrt3
        
        # Coarse scan for optimal rotation
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        try:
            res = minimize_scalar_local(eval_phi, lo, hi)
            if res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, center_x, center_y, best_phi

    def minimize_scalar_local(func, lo, hi):
        # Simple golden section search to avoid scipy import overhead if needed, 
        # but scipy is available. Let's use scipy for robustness.
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(func, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
        return res

    def collision_penalty(h1, h2):
        """
        Computes a smooth penalty based on the Minkowski sum distance.
        If disjoint, penalty is 0.
        If overlapping, penalty is proportional to the penetration depth.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        
        # Quick reject
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5: # (2*0.866 + margin)^2 approx, actually max dist for touching is 2*sqrt(3) ~ 3.46, squared ~ 12?
            # Wait, max distance between centers of touching unit hexagons is 2 * apothem? No.
            # Max distance is when they touch at vertices. Distance = 2 * circumradius = 2.
            # Min distance is when they touch at edges. Distance = 2 * apothem = sqrt(3) ~ 1.732.
            # So if dist > 2 + epsilon, they are definitely disjoint?
            # No, if they are rotated, they can be further apart?
            # Actually, the Minkowski sum of two unit hexagons is a hexagon with "radius" up to 2.
            # If distance > 2, they are disjoint?
            # Let's use a tighter bound. If dist > 3.5, definitely disjoint.
            return 0.0

        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Vertices
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Normals
        en1x = _nx * c1 - _ny * s1
        en1y = _nx * s1 + _ny * c1
        en2x = _nx * c2 - _ny * s2
        en2y = _nx * s2 + _ny * c2
        
        max_pen = 0.0
        
        # Check separation axes from Hex 1
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            # Overlap amount = max(min1, min2) - min(max1, max2)
            # If negative, they are separated.
            # We want the maximum penetration (positive value)
            pen = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if pen > max_pen:
                max_pen = pen
        
        # Check separation axes from Hex 2
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            pen = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if pen > max_pen:
                max_pen = pen
        
        if max_pen <= 0:
            return 0.0
        else:
            # Smooth penalty: squared penetration
            return max_pen * max_pen

    def objective(params):
        """
        params: [x0, y0, a0, x1, y1, a1, ...]
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        
        # Compute side
        side, _, _, _ = compute_side_and_center(positions, angles)
        
        # Compute collision penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], math.radians(angles[i]))
                h2 = (positions[j, 0], positions[j, 1], math.radians(angles[j]))
                p = collision_penalty(h1, h2)
                if p > 0:
                    penalty += p
        
        # Weighted objective
        # If penalty is high, side doesn't matter as much, but we want to minimize side eventually.
        # We can use a large multiplier for penalty.
        return side + 10.0 * penalty

    # Initial configurations
    configs = []
    
    # 1. Honeycomb
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0
    configs.append((base_positions, base_angles))
    
    # 2. Random scatter
    for _ in range(5):
        pos = np.random.normal(0, 2.0, (N, 2))
        ang = np.random.uniform(0, 360, N)
        configs.append((pos, ang))
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    time_per_config = (DEADLINE - 10.0) / len(configs)

    for pos_init, ang_init in configs:
        if time.time() - t0 > DEADLINE - 5.0:
            break
            
        params = np.zeros(3*N)
        for i in range(N):
            params[3*i] = pos_init[i, 0]
            params[3*i+1] = pos_init[i, 1]
            params[3*i+2] = ang_init[i]
            
        try:
            # L-BFGS-B
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            # Extract results
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = res.x[3*i]
                new_pos[i, 1] = res.x[3*i+1]
                new_ang[i] = res.x[3*i+2]
            
            # Check validity (no collisions)
            valid = True
            angles_rad = np.radians(new_ang)
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                    h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                    if collision_penalty(h1, h2) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side, cx, cy, phi = compute_side_and_center(new_pos, new_ang)
                if side < best_side:
                    best_side = side
                    best_positions = new_pos.copy()
                    best_angles = new_ang.copy()
                    best_phi = phi
                    best_cx = cx
                    best_cy = cy
                    
        except Exception as e:
            pass

    # If no valid solution found, fallback to honeycomb
    if best_positions is None:
        best_positions = configs[0][0].copy()
        best_angles = configs[0][1].copy()
        best_side, best_cx, best_cy, best_phi = compute_side_and_center(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

