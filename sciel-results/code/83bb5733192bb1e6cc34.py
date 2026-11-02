import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random
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

    def compute_penalty_and_side(positions, alphas, s, theta):
        """
        Computes the objective value: s + penalty.
        Also returns the actual required side length for validation.
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

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # params: [x1, y1, ..., x12, y12, a1, ..., a12, s, theta]
        positions = params[:2 * N].reshape(N, 2)
        alphas = params[2 * N : 2 * N + N]
        s = params[2 * N + N]
        theta = params[2 * N + N + 1]
        
        if s < 1.5:  # Sanity check
            return 1e6
            
        penalty = compute_penalty_and_side(positions, alphas, s, theta)
        return s + penalty

    def compute_min_side(positions, alphas, theta):
        """
        Given positions, alphas, and outer rotation theta,
        compute the minimum side length s required to contain all inner hexagons.
        This is used for final validation and potentially as a refinement step.
        """
        # The outer hexagon is centered at (0,0) with rotation theta.
        # Its support function in direction n is s * |n| * (some constant)?
        # Actually, for a regular hexagon of side s centered at origin:
        # The distance from center to any side is apothem = s * sqrt(3)/2.
        # The distance from center to a vertex is s.
        
        # Let's compute the required s by checking all vertices of inner hexagons
        # against the 6 half-planes of the outer hexagon.
        
        # Outer hexagon vertices (for normals):
        outer_verts_ref = hex_vertices(0, 0, 1.0, theta)
        outer_normals = get_normals(outer_verts_ref)
        
        # For a unit hexagon (side=1), the max projection of its vertices onto its own normals is the apothem?
        # No, for a regular polygon, the support function h(n) is the max dot product of vertices with n.
        # For a regular hexagon of side s, h(n) = s * h_unit(n).
        # We need: max_{v in inner_i} (v . n_k) <= s * h_unit(n_k) for all k.
        # So s >= max_{i, v, k} ( (v - 0) . n_k ) / h_unit(n_k)
        
        # Let's compute h_unit for the 6 normals of the unit hexagon.
        # The normals are perpendicular to the edges.
        # For a unit hexagon, the distance from center to edge is sqrt(3)/2.
        # The projection of the vertices onto the normal of an edge is exactly sqrt(3)/2.
        # Wait, let's verify.
        # Vertex at angle 0: (1, 0). Normal of edge between -30 and 30 deg?
        # Edge 0 is between v0 and v1. v0=(1,0), v1=(1/2, sqrt(3)/2).
        # Midpoint (3/4, sqrt(3)/4). Normal points out.
        # Actually, simpler: The hexagon is the intersection of 6 strips.
        # For a unit hexagon, the inequality is v . n_k <= sqrt(3)/2 ?
        # Let's check v0=(1,0). n0 is normal to edge opposite to v3?
        # Let's just compute h_unit numerically for the specific normals.
        
        h_unit = np.zeros(6)
        for k in range(6):
            # Max projection of unit hexagon vertices onto normal k
            # We can just use the fact that for a regular hexagon, the support function
            # in the direction of a face normal is the apothem.
            h_unit[k] = np.sqrt(3) / 2.0
            
        # However, the normals might not be exactly aligned if theta is arbitrary?
        # No, outer_normals are computed from the rotated hexagon, so they are consistent.
        # The support function of a regular hexagon of side s in the direction of its face normals is s * sqrt(3)/2.
        # In other directions, it's different. But we only need to check the 6 face normals of the OUTER hexagon.
        # The inner hexagons can be rotated arbitrarily.
        
        required_s = 0.0
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i])
            for k in range(6):
                proj = np.dot(verts, outer_normals[k])
                max_proj = np.max(proj)
                # s * (sqrt(3)/2) >= max_proj
                s_req = max_proj / (np.sqrt(3) / 2.0)
                if s_req > required_s:
                    required_s = s_req
                    
        return required_s

    # --- Initializations ---
    
    def init_hex_cluster():
        # 1 center, 6 ring 1, 5 ring 2
        d1 = np.sqrt(3)
        d2 = 2 * np.sqrt(3)
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

    def init_circular():
        pos = []
        alphas = []
        pos.append([0.0, 0.0])
        alphas.append(0.0)
        r = 3.0
        for k in range(11):
            a = np.radians(360.0 / 11 * k)
            pos.append([r * np.cos(a), r * np.sin(a)])
            alphas.append(0.0)
        return np.array(pos), np.array(alphas)

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
        init_hex_cluster,
        init_circular,
        init_random
    ]

    best_s = np.inf
    best_params = None

    # --- Optimization Strategy ---
    # 1. Try L-BFGS-B from multiple starts.
    # 2. Use the result to potentially refine s using compute_min_side.
    
    max_runs = 10
    
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
        
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.fun < best_s:
                pos = res.x[:2 * N].reshape(N, 2)
                alphas = res.x[2 * N : 2 * N + N]
                s = res.x[2 * N + N]
                theta = res.x[2 * N + N + 1]
                
                penalty = compute_penalty_and_side(pos, alphas, s, theta)
                
                if penalty < 1e-4:
                    # Refine s to be as small as possible
                    s_refined = compute_min_side(pos, alphas, theta)
                    # Add a tiny epsilon to ensure strict containment if needed, 
                    # though the evaluator likely allows touching.
                    s_refined += 1e-6 
                    
                    # Verify with the refined s
                    penalty_refined = compute_penalty_and_side(pos, alphas, s_refined, theta)
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
        pos, alphas = init_hex_cluster()
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
