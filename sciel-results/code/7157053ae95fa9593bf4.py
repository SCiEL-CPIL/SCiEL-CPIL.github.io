import time
import random
import numpy as np
from collections import Counter

def _score_fast(points):
    """
    Fast scoring using NumPy.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points, dtype=np.int32)
    n = len(P)
    
    if n < 3:
        return 0
    
    hist_size = 4001
    counts = np.zeros(hist_size, dtype=np.int32)
    
    for i in range(n - 2):
        pi = P[i]
        Pj = P[i+1:]
        m = len(Pj)
        if m < 2:
            continue
        
        j_idx, k_idx = np.triu_indices(m, k=1)
        
        v1 = Pj[j_idx] - pi
        v2 = Pj[k_idx] - pi
        
        dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        mask = (dets > 0) & (dets < hist_size)
        dets = dets[mask]
        
        if len(dets) > 0:
            temp = np.bincount(dets, minlength=hist_size)
            counts += temp
    
    return int(np.max(counts)) if n >= 3 else 0

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_score = 0
    best_points = None
    
    def _try_config(points):
        nonlocal best_score, best_points
        score = _score_fast(points)
        if score > best_score:
            best_score = score
            best_points = points[:]
        return score
    
    # 1. Systematic Grid Search
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    
    for W, H in factors:
        max_sx = 60 // (W - 1) if W > 1 else 60
        max_sy = 60 // (H - 1) if H > 1 else 60
        
        for sx in range(1, max_sx + 1):
            for sy in range(1, max_sy + 1):
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                if grid_width <= 60 and grid_height <= 60:
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    _try_config(points)
                    
                    if W > 1 and H > 1:
                        cx = -grid_width // 2
                        cy = -grid_height // 2
                        points_c = [(x * sx + cx, y * sy + cy) for x in range(W) for y in range(H)]
                        if all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in points_c):
                            _try_config(points_c)
    
    if best_points is None:
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
    
    # 2. Graph-based greedy selection for target areas
    # Precompute all candidate points and their pairwise area relationships
    
    all_candidates = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    cand_idx = {pt: i for i, pt in enumerate(all_candidates)}
    n_cands = len(all_candidates)
    
    # For each target area A, build adjacency: for each pair (i,j), 
    # which other points k form a triangle of area A with i,j
    # This is expensive, so we'll compute on-the-fly per target area
    
    def _greedy_graph_select(target_A, n_points=30, max_time_budget=10.0):
        """
        Greedy selection using graph structure.
        For target area A, at each step select the point that creates
        the most new triangles of area A with already-selected pairs.
        """
        if time.time() > deadline - 5:
            return None
        
        # Start with a few well-separated seed points
        random.seed(42)
        seeds = []
        attempts = 0
        while len(seeds) < 4 and attempts < 1000:
            p = random.choice(all_candidates)
            if all(abs(p[0]-s[0]) + abs(p[1]-s[1]) > 10 for s in seeds):
                seeds.append(p)
            attempts += 1
        
        selected = list(seeds)
        selected_set = set(selected)
        
        # For efficiency, precompute for each candidate point p and each pair (i,j) in selected,
        # whether det(p, i, j) == target_A
        # We'll maintain a count: for each candidate p, how many pairs in selected form area A with p
        
        # Initialize contribution counts
        contrib = np.zeros(n_cands, dtype=np.int32)
        
        # Compute initial contributions
        for idx, s in enumerate(selected):
            for jdx in range(idx + 1, len(selected)):
                sj = selected[jdx]
                # For each candidate p, check if det(p, s, sj) == target_A
                # det = |px*(sy-sjy) + sx*(sjy-py) + sjx*(py-sy)|
                # This is linear in px, py
                # We can vectorize over all candidates
                
                px = np.array([c[0] for c in all_candidates])
                py = np.array([c[1] for c in all_candidates])
                
                det_vals = np.abs(px * (s[1] - sj[1]) + s[0] * (sj[1] - py) + sj[0] * (py - s[1]))
                
                # Add to contribution where det == target_A
                mask = (det_vals == target_A) & (det_vals > 0)
                contrib[mask] += 1
        
        # Greedily select remaining points
        for step in range(n_points - len(selected)):
            if time.time() > deadline - 5:
                break
            
            # Exclude already selected points
            for s in selected:
                contrib[cand_idx[s]] = -1  # Mark as unavailable
            
            # Find best candidate
            best_idx = np.argmax(contrib)
            best_contrib = contrib[best_idx]
            
            if best_contrib <= 0:
                # No improvement possible with this target area, pick random
                available = [i for i in range(n_cands) if contrib[i] >= 0]
                if not available:
                    break
                best_idx = random.choice(available)
            
            best_pt = all_candidates[best_idx]
            selected.append(best_pt)
            selected_set.add(best_pt)
            
            # Update contributions for new pairs involving best_pt
            for idx in range(len(selected) - 2):
                s = selected[idx]
                # det(p, s, best_pt) for all p
                px = np.array([c[0] for c in all_candidates])
                py = np.array([c[1] for c in all_candidates])
                
                det_vals = np.abs(px * (s[1] - best_pt[1]) + s[0] * (best_pt[1] - py) + best_pt[0] * (py - s[1]))
                
                mask = (det_vals == target_A) & (det_vals > 0)
                contrib[mask] += 1
            
            # Mark new point as unavailable
            contrib[cand_idx[best_pt]] = -1
        
        if len(selected) == n_points:
            return selected
        return None
    
    # Try multiple target areas
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
    
    for A in target_areas:
        if time.time() > deadline - 5:
            break
        
        points = _greedy_graph_select(A)
        if points is not None:
            _try_config(points)
    
    # 3. Local search / hill climbing
    if time.time() > deadline - 5:
        return best_points
    
    current_points = best_points[:]
    current_score = best_score
    n = 30
    current_set = set(current_points)
    
    for round_num in range(20):
        if time.time() > deadline - 2:
            break
        
        improved = False
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            ox, oy = current_points[idx]
            
            for r in range(1, 4):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx = ox + dx
                        ny = oy + dy
                        
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        
                        if (nx, ny) in current_set:
                            continue
                        
                        current_points[idx] = (nx, ny)
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_set.discard((ox, oy))
                            current_set.add((nx, ny))
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                        else:
                            current_points[idx] = (ox, oy)
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_points
