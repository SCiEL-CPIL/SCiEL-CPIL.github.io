# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals for a hexagon (6, 2) array."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _check_separation(v1, v2, normals1, normals2, tol=1e-6):
    """Check if two hexagons are separated. Returns True if separated."""
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        gap = min(max1, max2) - max(min1, min2)
        if gap < -tol:
            return False
    return True

def _check_containment(inner_verts, outer_verts, tol=1e-6):
    """Check if all inner vertices are inside the outer hexagon."""
    for v in inner_verts:
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k + 1) % 6]
            edge = p2 - p1
            to_point = v - p1
            cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
            if cross_val < -tol:
                return False
    return True

def _compute_penalty(centers, s, angle_inner, angle_outer, tol=1e-6):
    """
    Compute total penalty for a configuration.
    Returns (penalty, is_feasible)
    """
    penalty = 0.0
    feasible = True
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
    
    # Pairwise separation check
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            max_overlap = 0.0
            for n in all_normals:
                p1 = inner_verts[i] @ n
                p2 = inner_verts[j] @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    max_overlap = max(max_overlap, -gap)
                    feasible = False
            penalty += max_overlap * max_overlap
    
    # Boundary check
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                to_point = v - p1
                cross_val = edge[0] * to_point[1] - edge[1] * to_point[0]
                if cross_val < -tol:
                    penalty += (-cross_val) ** 2
                    feasible = False
    
    return penalty, feasible

def _is_valid(centers, s, angle_inner, angle_outer, tol=1e-6):
    """Check if configuration is fully valid."""
    penalty, feasible = _compute_penalty(centers, s, angle_inner, angle_outer, tol)
    return feasible

def hexagon_packing_12():
    """
    Uses Simulated Annealing to pack 12 unit hexagons in a regular hexagon,
    minimizing the outer hexagon side length.
    """
    start_time = time.time()
    deadline = 625.0  # safety margin before 630s kill
    
    best_score = 0.0
    best_result = None
    
    # Initial configuration: hexagonal grid pattern
    def make_initial_config():
        centers = np.zeros((12, 2))
        # Hex grid with flat-top orientation
        # Row-based arrangement
        sqrt3 = np.sqrt(3)
        # Row 0 (top): 3 hexagons
        centers[0] = [-1.5, 2 * sqrt3 / 2 * 2]  # y = 2*sqrt(3)
        centers[1] = [0, 2 * sqrt3 / 2 * 2]
        centers[2] = [1.5, 2 * sqrt3 / 2 * 2]
        # Row 1: 4 hexagons
        centers[3] = [-3.0, sqrt3]
        centers[4] = [-1.5, sqrt3]
        centers[5] = [0, sqrt3]
        centers[6] = [1.5, sqrt3]
        # Row 2: 3 hexagons
        centers[7] = [-1.5, 0]
        centers[8] = [0, 0]
        centers[9] = [1.5, 0]
        # Row 3 (bottom): 2 hexagons
        centers[10] = [-0.75, -sqrt3 / 2]
        centers[11] = [0.75, -sqrt3 / 2]
        return centers
    
    # Try different angle combinations
    angle_options = [0, 30]
    
    for angle_outer in angle_options:
        for angle_inner in angle_options:
            if time.time() > deadline:
                break
            
            # Start with a reasonable s estimate
            # For 12 hexagons in hex grid, bounding hex side is roughly 4-5
            s = 5.0
            
            centers = make_initial_config()
            
            # First, find a feasible s by optimizing at fixed s
            # Use SA to minimize penalty at fixed s
            current_centers = centers.copy()
            current_penalty, current_feasible = _compute_penalty(
                current_centers, s, angle_inner, angle_outer)
            
            # SA parameters
            T0 = 1.0  # initial temperature
            T_end = 1e-8  # final temperature
            cooling_rate = 0.9995
            T = T0
            
            best_centers = current_centers.copy()
            best_penalty = current_penalty
            best_s = s
            
            # SA loop
            max_iter = 500000
            iter_count = 0
            
            while T > T_end and time.time() < deadline:
                # Generate neighbor: perturb one hexagon's center
                idx = random.randint(0, 11)
                dx = random.gauss(0, 0.5)
                dy = random.gauss(0, 0.5)
                
                new_centers = current_centers.copy()
                new_centers[idx] = [current_centers[idx][0] + dx, 
                                     current_centers[idx][1] + dy]
                
                new_penalty, new_feasible = _compute_penalty(
                    new_centers, s, angle_inner, angle_outer)
                
                delta = new_penalty - current_penalty
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_centers = new_centers
                    current_penalty = new_penalty
                    
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_centers = current_centers.copy()
                
                # Cool down
                T *= cooling_rate
                iter_count += 1
                
                # Periodically try to decrease s
                if iter_count % 1000 == 0 and best_penalty < 1e-4:
                    # Try slightly smaller s
                    s_trial = s * 0.995
                    trial_penalty, trial_feasible = _compute_penalty(
                        best_centers, s_trial, angle_inner, angle_outer)
                    if trial_feasible:
                        s = s_trial
                        best_s = s
                        best_penalty = trial_penalty
                    else:
                        # Try even smaller step
                        s_trial = s * 0.999
                        trial_penalty, trial_feasible = _compute_penalty(
                            best_centers, s_trial, angle_inner, angle_outer)
                        if trial_feasible:
                            s = s_trial
                            best_s = s
                            best_penalty = trial_penalty
                        # Reset SA with smaller s
                        current_centers = best_centers.copy()
                        current_penalty, _ = _compute_penalty(
                            current_centers, s, angle_inner, angle_outer)
                        best_penalty = current_penalty
                        T = max(T, 0.1)  # Reheat slightly
                        iter_count = 0
            
            # Final verification
            if best_penalty < 1e-4:
                # Verify with tighter tolerance
                final_penalty, final_feasible = _compute_penalty(
                    best_centers, best_s, angle_inner, angle_outer, tol=1e-8)
                if final_feasible:
                    score = 1.0 / best_s
                    if score > best_score:
                        best_score = score
                        inner_data = np.column_stack([best_centers, np.full(12, angle_inner)])
                        outer_data = np.array([0, 0, angle_outer])
                        best_result = (inner_data, outer_data, best_s)
    
    if best_result is None:
        # Fallback: return a valid seed solution
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result

# EVOLVE-BLOCK-END
