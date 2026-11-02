import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s

    # --- Geometry Helpers ---

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        # Outward normals for CCW polygon: (dy, -dx)
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def penetration_depth(v1, v2):
        """Returns max overlap depth (0 if disjoint) using SAT."""
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
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

    def inside_violation(inner_verts, outer_verts):
        """Returns max distance inner sticks out of outer (0 if inside)."""
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def compute_penalty(positions, alphas, s, theta):
        """
        Computes the penalty term only (0 if valid).
        """
        outer_verts = hex_vertices(0, 0, s, theta)
        total_penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alphas[i])
            inner_verts_list.append(verts)
            
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 10.0
            
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 10.0
                
        return total_penalty

    def compute_min_side(positions, alphas, theta):
        """
        Given positions, alphas, and outer rotation theta,
        compute the minimum side length s required to contain all inner hexagons.
        """
        outer_verts_ref = hex_vertices(0, 0, 1.0, theta)
        outer_normals = get_normals(outer_verts_ref)
        
        required_s = 0.0
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i])
            for k in range(6):
                proj = np.dot(verts, outer_normals[k])
                max_proj = np.max(proj)
                s_req = max_proj / (np.sqrt(3) / 2.0)
                if s_req > required_s:
                    required_s = s_req
                    
        return required_s

    # --- Initializations ---

    def init_honeycomb_3x4():
        """
        Creates a 3x4 grid of hexagons in honeycomb arrangement.
        This is a known dense packing for 12 hexagons.
        """
        pos = []
        alphas = [0.0] * N
        
        # Honeycomb lattice vectors
        # Distance between centers of adjacent hexagons is sqrt(3) * side = sqrt(3)
        d = np.sqrt(3)
        
        # 3 rows, 4 columns
        # Row 0: 4 hexagons
        # Row 1: 3 hexagons (shifted)
        # Row 2: 4 hexagons? No, 3+4+5? Or 4+3+4?
        # Let's try 4-3-4 or 3-4-3?
        # Actually, a compact cluster for 12 is often 1-6-5 or 3-4-5?
        # Let's use the standard honeycomb lattice points.
        
        # Lattice basis:
        # u = (d, 0)
        # v = (d/2, d*sqrt(3)/2)
        
        # Let's place them in a 3x4 grid with offset
        # Row 0: x = 0, d, 2d, 3d; y = 0
        # Row 1: x = d/2, 3d/2, 5d/2; y = d*sqrt(3)/2
        # Row 2: x = 0, d, 2d, 3d; y = d*sqrt(3)
        
        # Total: 4 + 3 + 4 = 11. Need 12.
        # Maybe 4-4-4? No, that's not honeycomb.
        # Let's try 5-4-3?
        # Or just use a 3x4 grid of square packing? No, honeycomb is denser.
        
        # Let's use a 1-6-5 flower pattern which is very compact.
        # Center: (0,0)
        # Ring 1: 6 hexagons at distance d
        # Ring 2: 5 hexagons at distance 2d (skipping one to keep compactness or just fill 5 spots)
        
        pos.append([0.0, 0.0])
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d * np.cos(a), d * np.sin(a)])
        
        # Ring 2: place 5 hexagons at the "corners" of the hexagon formed by ring 1
        # The 6 ring 1 hexagons are at angles 0, 60, 120, 180, 240, 300.
        # The gaps are at 30, 90, 150, 210, 270, 330.
        # Distance to gap centers is d * sqrt(3) ? No.
        # Center of gap between two adjacent ring 1 hexagons:
        # Midpoint of two ring 1 centers is at distance d/2 * cos(30) ... no.
        # Let's just place them at radius 2d at angles 0, 60, 120, 180, 240.
        # This might cause overlaps with ring 1?
        # Distance between ring 1 (angle 0) and ring 2 (angle 0) is d. They touch.
        # Distance between ring 1 (angle 0) and ring 2 (angle 60)?
        # Ring 1 (0): (d, 0)
        # Ring 2 (60): (2d cos 60, 2d sin 60) = (d, d*sqrt(3))
        # Dist^2 = 0 + 3d^2 = 3d^2. Dist = d*sqrt(3) > d. OK.
        
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([2 * d * np.cos(a), 2 * d * np.sin(a)])
            
        return np.array(pos), np.array(alphas)

    def init_honeycomb_4x3():
        """
        Alternative honeycomb arrangement.
        """
        pos = []
        alphas = [0.0] * N
        d = np.sqrt(3)
        
        # 4 rows, 3 cols?
        # Row 0: 3
        # Row 1: 3
        # Row 2: 3
        # Row 3: 3
        # With offsets?
        
        # Let's stick to the flower 1-6-5 which is known to be compact.
        return init_honeycomb_3x4()

    def init_random():
        pos = []
        alphas = []
        pos.append([0.0, 0.0])
        alphas.append(0.0)
        for k in range(11):
            a = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(1, 4)
            pos.append([r * np.cos(a), r * np.sin(a)])
            alphas.append(np.random.uniform(0, np.pi / 3))
        return np.array(pos), np.array(alphas)

    initializations = [
        init_honeycomb_3x4,
        init_random
    ]

    best_s = np.inf
    best_params = None

    # --- Optimization Strategy ---
    # 1. Try Nelder-Mead from multiple starts.
    # 2. Use the result to potentially refine s using compute_min_side.
    
    max_runs = 15
    
    for run in range(max_runs):
        if time.time() > deadline - 20:
            break
            
        init_func = initializations[run % len(initializations)]
        init_pos, init_alphas = init_func()
        
        if run >= len(initializations):
            init_pos = init_pos + np.random.normal(0, 0.2, init_pos.shape)
            init_alphas = init_alphas + np.random.uniform(-0.1, 0.1, N)
            
        init_s = 4.5
        init_theta = 0.0
        
        x0 = np.concatenate([
            init_pos.flatten(), 
            init_alphas, 
            [init_s], 
            [init_theta]
        ])
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
                
            positions = params[:2 * N].reshape(N, 2)
            alphas = params[2 * N : 2 * N + N]
            s = params[2 * N + N]
            theta = params[2 * N + N + 1]
            
            if s < 1.5:
                return 1e6
                
            penalty = compute_penalty(positions, alphas, s, theta)
            return s + penalty

        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.fun < best_s:
                pos = res.x[:2 * N].reshape(N, 2)
                alphas = res.x[2 * N : 2 * N + N]
                s = res.x[2 * N + N]
                theta = res.x[2 * N + N + 1]
                
                penalty = compute_penalty(pos, alphas, s, theta)
                
                if penalty < 1e-4:
                    s_refined = compute_min_side(pos, alphas, theta)
                    s_refined += 1e-6 
                    
                    penalty_refined = compute_penalty(pos, alphas, s_refined, theta)
                    if penalty_refined < 1e-4:
                        best_s = s_refined
                        best_params = np.concatenate([pos.flatten(), alphas, [s_refined], [theta]])
                    else:
                        best_s = s
                        best_params = res.x.copy()
                        
        except Exception:
            pass

    # Fallback
    if best_params is None:
        pos, alphas = init_honeycomb_3x4()
        s = compute_min_side(pos, alphas, 0.0)
        best_params = np.concatenate([pos.flatten(), alphas, [s], [0.0]])
        best_s = s

    # Extract final
    final_pos = best_params[:2 * N].reshape(N, 2)
    final_alphas = best_params[2 * N : 2 * N + N]
    final_s = best_params[2 * N + N]
    final_theta = best_params[2 * N + N + 1]
    
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_alphas[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg

