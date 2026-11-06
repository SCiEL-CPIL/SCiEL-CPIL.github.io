import time
import random
import math
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n_pts = 30
    
    # Precompute indices for all triples
    idxs = list(itertools.combinations(range(n_pts), 3))
    idx_array = np.array(idxs)
    pre_i = idx_array[:, 0]
    pre_j = idx_array[:, 1]
    pre_k = idx_array[:, 2]

    def evaluate_fast(points):
        """
        points: list of 30 (x, y) tuples
        Returns max multiplicity of non-zero doubled area.
        """
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        
        xi = x[pre_i]
        yi = y[pre_i]
        xj = x[pre_j]
        yj = y[pre_j]
        xk = x[pre_k]
        yk = y[pre_k]
        
        # Determinant: xi(yj - yk) + xj(yk - yi) + xk(yi - yj)
        dets = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
        abs_dets = np.abs(dets)
        
        # Filter zeros
        non_zero = abs_dets[abs_dets > 0]
        if len(non_zero) == 0:
            return 0
            
        # Count multiplicities
        # np.unique is efficient for this size
        _, counts = np.unique(non_zero, return_counts=True)
        return int(np.max(counts))

    def make_grid(w, h):
        ox = -w // 2
        oy = -h // 2
        return [(ox + i, oy + j) for i in range(w) for j in range(h)]

    candidates = []
    
    # Grids
    grid_shapes = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]
    for w, h in grid_shapes:
        pts = make_grid(w, h)
        # Verify bounds and distinctness
        if len(set(pts)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts)

    # Random perturbations of the best likely grid (6x5)
    base_grid = make_grid(6, 5)
    for _ in range(10):
        perturbed = base_grid[:]
        for _ in range(10):
            idx = random.randint(0, 29)
            x, y = perturbed[idx]
            new_x = x + random.randint(-2, 2)
            new_y = y + random.randint(-2, 2)
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                if (new_x, new_y) not in perturbed:
                    perturbed[idx] = (new_x, new_y)
        candidates.append(perturbed)

    best_points = None
    best_score = -1
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        best_points = make_grid(6, 5)
        best_score = evaluate_fast(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    max_iter = 100000
    
    point_set = set(current_points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 29)
        x, y = current_points[idx]
        
        # Generate move
        # Bias towards small steps
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = x + dx
        new_y = y + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        
        # Collision check
        if new_pt in point_set:
            continue
            
        # Apply
        point_set.remove((x, y))
        point_set.add(new_pt)
        current_points[idx] = new_pt
        
        new_score = evaluate_fast(current_points)
        
        # Acceptance
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                point_set.remove(new_pt)
                point_set.add((x, y))
                current_points[idx] = (x, y)
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points
