from scipy.optimize import differential_evolution
import math
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def sat_separated(v1, v2):
        """Check if two convex polygons are separated using SAT. Returns min separation (positive = separated)."""
        min_sep = float('inf')
        # Precompute edge normals for both polygons
        normals = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                # Normalize
                norm = math.sqrt(nx*nx + ny*ny)
                if norm > 1e-12:
                    nx /= norm
                    ny /= norm
                normals.append((nx, ny))
        
        for nx, ny in normals:
            proj1 = np.dot(v1, [nx, ny])
            proj2 = np.dot(v2, [nx, ny])
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
            if min_sep <= 0:
                return min_sep
        return min_sep

    def compute_outer_side(centers, angles, otheta):
        """Compute the minimum side length of a regular hexagon (centered at origin, rotated by otheta) containing all inner hexagons."""
        otheta_rad = math.radians(otheta)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(12):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        # For a regular hexagon with side length s, the apothem is s*sqrt(3)/2
        # The support function in direction n (perpendicular to edge) equals the apothem.
        # So max_proj = apothem = s*sqrt(3)/2
        # Therefore s = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def get_penalty(centers, angles):
        """Compute overlap penalty between all pairs of inner hexagons."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = sat_separated(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        return penalty

    def is_valid(centers, angles, eps=1e-6):
        """Check if all inner hexagons are disjoint."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if sat_separated(verts[i], verts[j]) < -eps:
                    return False
        return True

    def objective(params):
        if time.time() > deadline:
            return 1e9
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        side = compute_outer_side(centers, angles, otheta)
        pen = get_penalty(centers, angles)
        # Strong penalty for overlap
        return side + 10000.0 * pen

    def generate_hexagonal_cluster():
        """
        Generate a valid hexagonal cluster of 12 hexagons.
        Structure: 1 center, 6 in first ring, 5 in second ring.
        Spacing: d = sqrt(3) + 0.1 to ensure no overlap initially.
        """
        d = math.sqrt(3) + 0.1
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        
        # Ring 1
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        # Ring 2 candidates (neighbors of ring 1 that are not center or ring 1)
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        seen = set()
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                key = (round(float(p2[0]), 6), round(float(p2[1]), 6))
                if key in seen:
                    continue
                if np.linalg.norm(p2) < 0.1:
                    continue
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-6):
                        is_r1 = True
                        break
                if not is_r1:
                    seen.add(key)
                    ring2_candidates.append(p2)
                    
        # Sort by distance to center to pick the closest 5
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        angles = [0.0] * 12
        return centers, angles

    def params_from_config(centers, angles, otheta=0.0):
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(otheta)
        return np.array(params)

    # 1. Generate initial valid configuration
    centers_init, angles_init = generate_hexagonal_cluster()
    init_params = params_from_config(centers_init, angles_init)
    
    # Verify initial validity
    if not is_valid(centers_init, angles_init):
        # Fallback: spread them out more if somehow invalid (should not happen)
        print("Warning: Initial cluster invalid, increasing spacing.")
        # Retry with larger spacing
        d = math.sqrt(3) + 0.5
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers_init = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers_init.append(vec.copy())
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        seen = set()
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                key = (round(float(p2[0]), 6), round(float(p2[1]), 6))
                if key in seen: continue
                if np.linalg.norm(p2) < 0.1: continue
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-6):
                        is_r1 = True
                        break
                if not is_r1:
                    seen.add(key)
                    ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers_init.append(ring2_candidates[i].copy())
        angles_init = [0.0] * 12
        init_params = params_from_config(centers_init, angles_init)
        
    best_params = init_params.copy()
    best_val = compute_outer_side(centers_init, angles_init, 0.0)
    
    # 2. Local Refinement using Nelder-Mead
    # We start from a valid point, so the penalty is 0.
    # We must ensure the optimizer doesn't step into a large overlap.
    # Using a small initial simplex size helps.
    
    if time.time() < deadline - 20:
        try:
            # Use Nelder-Mead with tight tolerances
            # Note: The objective function has a hard penalty for overlap.
            # If the optimizer moves into overlap, the cost spikes, pushing it back.
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex': None})
            
            new_centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            new_angles = [res.x[24+i] for i in range(12)]
            
            if is_valid(new_centers, new_angles):
                new_val = compute_outer_side(new_centers, new_angles, res.x[36])
                if new_val < best_val:
                    best_val = new_val
                    best_params = res.x.copy()
        except Exception as e:
            print(f"Minimization error: {e}")

    # 3. Fine-tune outer hexagon angle
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    
    # Grid search for optimal outer angle
    best_side = best_val
    best_otheta = best_params[36]
    
    # Coarse to fine search
    for step in [10, 2, 0.5, 0.1, 0.02]:
        for da in np.linspace(-step, step, int(2*step/0.1)+1 if step > 0.1 else 21):
             # Actually simpler: just iterate around current best
             pass
             
    # Simpler robust angle search
    for da in np.linspace(-30, 30, 61):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
            
    for da in np.linspace(-5, 5, 21):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
            
    for da in np.linspace(-1, 1, 21):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
            
    for da in np.linspace(-0.2, 0.2, 21):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Final check
    final_centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    final_angles = [best_params[24+i] for i in range(12)]
    
    if not is_valid(final_centers, final_angles):
        # If somehow invalid, revert to initial valid config
        final_centers = centers_init
        final_angles = angles_init
        best_side = compute_outer_side(final_centers, final_angles, 0.0)
        best_otheta = 0.0

    inner = [[c[0], c[1], a] for c, a in zip(final_centers, final_angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
