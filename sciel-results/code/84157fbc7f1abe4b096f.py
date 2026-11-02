import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def get_all_vertices(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        # Returns (12, 6, 2) array
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            verts[i] = hexagon_vertices(x, y, 1.0, a)
        return verts

    def compute_min_side(verts, outer_center, outer_angle):
        # verts: (N, 6, 2)
        # outer_center: (2,)
        # outer_angle: float (deg)
        
        # Normals of outer hexagon
        # The sides of a regular hexagon are perpendicular to angles:
        # phi, phi+60, ..., phi+300
        # The distance from center to side is S * sqrt(3)/2
        # So S = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        
        phi = np.radians(outer_angle)
        normals = []
        for k in range(6):
            theta = phi + np.radians(30) + k * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals) # (6, 2)
        
        # Translate vertices by outer center
        offsets = verts - np.array(outer_center) # (N, 6, 2)
        
        # Project onto normals
        # offsets: (N, 6, 2)
        # normals: (6, 2) -> (1, 6, 2)
        # We need dot product for each vertex and each normal.
        # Use einsum: 'ijk, kl -> ijl' ? No.
        # Projected values: offsets[..., 0] * normals[0] + offsets[..., 1] * normals[1]
        
        # Flatten vertices for easy dot product?
        # Or just loop over 6 normals (fast enough)
        
        max_extent = 0.0
        for n in normals:
            # Dot product of all vertices with normal n
            # offsets: (N, 6, 2), n: (2,)
            dots = np.sum(offsets * n, axis=2) # (N, 6)
            abs_dots = np.abs(dots)
            current_max = np.max(abs_dots)
            if current_max > max_extent:
                max_extent = current_max
                
        return (2.0 / np.sqrt(3)) * max_extent

    def penalty_overlap(params):
        # Calculate pairwise center distances
        centers = params[:N*3].reshape(N, 3)[:, :2]
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        # Min distance for non-overlap is sqrt(3) ~ 1.732
        # Penalty for d < 1.732
        min_dist = np.sqrt(3)
        # Only consider i < j
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        overlaps = np.maximum(0, min_dist - d)
        return np.sum(overlaps**2)

    def objective(params):
        # params: [x1, y1, a1, ..., x12, y12, a12, cx_out, cy_out, angle_out]
        inner_params = params[:N*3]
        cx_out = params[N*3]
        cy_out = params[N*3+1]
        angle_out = params[N*3+2]
        
        verts = get_all_vertices(inner_params)
        S = compute_min_side(verts, [cx_out, cy_out], angle_out)
        pen = penalty_overlap(inner_params)
        
        return S + 100.0 * pen

    def generate_honeycomb():
        # Generate a 3x4 honeycomb
        # Spacing: dx = 1.5, dy = sqrt(3)/2
        # Actually, for flat top:
        # Horizontal neighbor: dx = sqrt(3), dy = 0
        # Vertical neighbor (offset): dx = sqrt(3)/2, dy = 1.5
        # Let's use a grid
        pos = []
        # 4 columns, 3 rows
        for col in range(4):
            for row in range(3):
                x = col * np.sqrt(3)
                y = row * 1.5
                if col % 2 == 1:
                    y += 0.75 # Offset
                pos.append((x, y, 0.0))
        # Add 12th? No, 4x3=12.
        # Center it
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        params = []
        for x, y, a in pos:
            params.extend([x-cx, y-cy, a])
        
        # Outer center and angle
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)

    def generate_grid():
        pos = []
        for i in range(3):
            for j in range(4):
                x = j * 2.0
                y = i * np.sqrt(3)
                pos.append((x, y, 0.0))
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        params = []
        for x, y, a in pos:
            params.extend([x-cx, y-cy, a])
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)

    best_S = float('inf')
    best_params = None

    inits = [generate_honeycomb(), generate_grid()]
    
    # Add random inits
    for _ in range(5):
        p = generate_honeycomb()
        p += np.random.normal(0, 0.1, p.size)
        inits.append(p)

    for p0 in inits:
        if time.time() > deadline - 20:
            break
            
        # Optimize
        # Nelder-Mead
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success or res.fun < best_S:
            # Check validity
            inner_params = res.x[:N*3]
            verts = get_all_vertices(inner_params)
            cx_out = res.x[N*3]
            cy_out = res.x[N*3+1]
            angle_out = res.x[N*3+2]
            
            # Simple validity check: no overlaps (using distance)
            centers = inner_params.reshape(N, 3)[:, :2]
            dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            if np.min(d) > np.sqrt(3) - 1e-5: # Valid
                S_val = res.x[N*3] # No, S is in objective
                # Recalculate S
                S_val = compute_min_side(verts, [cx_out, cy_out], angle_out)
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x

    # Final result
    if best_params is None:
        # Fallback
        p = generate_honeycomb()
        best_params = p
        verts = get_all_vertices(p[:N*3])
        best_S = compute_min_side(verts, [p[N*3], p[N*3+1]], p[N*3+2])

    # Extract
    inner = []
    for i in range(N):
        inner.append([float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])])
    
    center = [float(best_params[N*3]), float(best_params[N*3+1])]
    side = float(best_S)
    angle = float(best_params[N*3+2])
    
    return inner, center, side, angle
