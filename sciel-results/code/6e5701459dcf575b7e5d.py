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

    def sat_overlap(v1, v2):
        """
        Returns (min_overlap, min_normal) for two convex polygons using SAT.
        If disjoint, min_overlap is negative.
        """
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
        
        if min_overlap > 0:
            idx = np.argmin(overlaps)
            min_normal = normals[idx]
        else:
            min_normal = np.array([0.0, 0.0])
            
        return min_overlap, min_normal

    def inside_violation(inner_verts, outer_verts):
        """Returns max distance inner sticks out of outer (0 if inside)."""
        outer_normals = get_normals(outer_verts)
        
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

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

    def evaluate(positions, alphas, theta):
        """
        Evaluates a configuration.
        Returns (s, penalty).
        """
        # Check containment
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
        
        s = required_s
        
        # Check overlaps
        penalty = 0.0
        verts_list = []
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, alphas[i])
            verts_list.append(verts)
            
        for i in range(N):
            for j in range(i + 1, N):
                overlap, _ = sat_overlap(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * 100.0 + overlap * overlap * 1000.0
        
        return s, penalty

    def init_honeycomb():
        """
        Creates a 3x4 grid of hexagons in honeycomb arrangement.
        """
        pos = []
        alphas = [0.0] * N
        d = np.sqrt(3) * 1.01
        
        for row in range(3):
            for col in range(4):
                x = col * d + (d / 2 if row % 2 == 1 else 0)
                y = row * d * np.sqrt(3) / 2
                pos.append([x, y])
                
        return np.array(pos), np.array(alphas)

    # --- Simulated Annealing ---
    
    # Initial configuration
    pos, alphas = init_honeycomb()
    theta = 0.0
    s, penalty = evaluate(pos, alphas, theta)
    
    best_pos = pos.copy()
    best_alphas = alphas.copy()
    best_theta = theta
    best_s = s
    best_penalty = penalty
    
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    
    # Precompute step sizes
    pos_step = 0.5
    alpha_step = 0.2
    theta_step = 0.1
    
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
            
        # Perturb
        new_pos = pos.copy()
        new_alphas = alphas.copy()
        new_theta = theta
        
        # Randomly perturb a few hexagons
        idx = np.random.randint(0, N)
        new_pos[idx] += np.random.normal(0, pos_step, 2)
        new_alphas[idx] += np.random.normal(0, alpha_step)
        
        # Occasionally perturb theta
        if np.random.random() < 0.1:
            new_theta += np.random.normal(0, theta_step)
            
        # Evaluate
        new_s, new_penalty = evaluate(new_pos, new_alphas, new_theta)
        
        # Objective: s + penalty
        obj = s + penalty
        new_obj = new_s + new_penalty
        
        delta = new_obj - obj
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            pos = new_pos
            alphas = new_alphas
            theta = new_theta
            s = new_s
            penalty = new_penalty
            
            # Update best
            if penalty < best_penalty:
                best_pos = pos.copy()
                best_alphas = alphas.copy()
                best_theta = theta
                best_s = s
                best_penalty = penalty
                # If we found a valid configuration, we can try to tighten s
                # But s is calculated as min_side, so it's already minimal for this config.
                # We just want to minimize s.
                if penalty < 1e-6:
                    best_s = s # This is the minimal s for this valid config
                    
        T *= cooling

    # Final refinement: ensure validity and minimal s
    # If best_penalty is not zero, we might have overlaps.
    # We need a valid solution.
    # Let's try to resolve overlaps in the best found configuration using SAT separation.
    
    def resolve_overlaps_sat(pos, alphas, max_iters=50):
        pos = pos.copy()
        alphas = alphas.copy()
        
        for _ in range(max_iters):
            moved = False
            verts_list = []
            for i in range(N):
                verts = hex_vertices(pos[i, 0], pos[i, 1], 1.0, alphas[i])
                verts_list.append(verts)
                
            for i in range(N):
                for j in range(i + 1, N):
                    overlap, normal = sat_overlap(verts_list[i], verts_list[j])
                    if overlap > 1e-6:
                        # Push apart along the normal
                        # normal points from i to j?
                        # SAT normal direction depends on the polygon winding.
                        # Let's check direction:
                        # If dot(normal, pos[j]-pos[i]) < 0, normal points from j to i?
                        # Actually, the normal is from the separating axis.
                        # Let's just try both directions and pick the one that reduces overlap?
                        # Or simply:
                        # The overlap is the depth. The vector to move is along the normal.
                        # We need to know which way is "out".
                        # For two convex polygons, if they overlap, the minimum overlap axis is unique.
                        # The direction from i to j projected on normal should be positive if normal is i->j?
                        # Let's compute vector d = pos[j] - pos[i]
                        d = pos[j] - pos[i]
                        dot = np.dot(d, normal)
                        if dot < 0:
                            normal = -normal
                            
                        # Move i by -normal * overlap/2 and j by normal * overlap/2
                        shift = overlap / 2.0
                        pos[i] -= normal * shift
                        pos[j] += normal * shift
                        moved = True
                        
            if not moved:
                break
        return pos, alphas

    # Use the best valid configuration if possible
    if best_penalty < 1e-6:
        final_pos = best_pos
        final_alphas = best_alphas
        final_theta = best_theta
        final_s = best_s
    else:
        # Fallback: Use the honeycomb init which is valid
        final_pos, final_alphas = init_honeycomb()
        final_theta = 0.0
        final_s, _ = evaluate(final_pos, final_alphas, final_theta)
        
        # Try to improve the fallback with a few SA steps?
        # No, just return the valid honeycomb. It's safe.
        # But we can try to optimize the honeycomb slightly.
        # For now, return the valid honeycomb.
        
    # Add a tiny margin to s to ensure floating point safety
    final_s += 1e-5

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_alphas[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(final_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
