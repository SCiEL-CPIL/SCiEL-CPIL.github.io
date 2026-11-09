import itertools
import random
import time
import math
from collections import defaultdict

N = 16
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin (evaluator kills at 330s, 300s limit)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    idx_to_point = all_points
    
    # Precompute isosceles conflict: 
    # For each pair (i, j), find all k such that (i, j, k) is isosceles
    # This gives us: when we have chosen set S, point p can be added iff
    # for all pairs (a, b) in S, p is NOT in iso_conflict[a][b]
    
    # iso_conflict[i][j] = bitmask of points k such that (i, j, k) is isosceles
    # Use Python integers as bitmasks for efficiency
    iso_conflict = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    
    # Precompute squared distances
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = all_points[i]
        for j in range(NUM_POINTS):
            xj, yj = all_points[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2
    
    # Precompute isosceles conflicts
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            for k in range(NUM_POINTS):
                if k == i or k == j:
                    continue
                d1 = dist2[i][j]
                d2 = dist2[i][k]
                d3 = dist2[j][k]
                if d1 == d2 or d1 == d3 or d2 == d3:
                    mask |= (1 << k)
            iso_conflict[i][j] = mask
            iso_conflict[j][i] = mask
    
    # Helper: check if point p_idx can be added to chosen set (represented as bitmask)
    def can_add(p_idx, chosen_mask, chosen_indices):
        if (chosen_mask >> p_idx) & 1:
            return False
        # Check all pairs in chosen
        for i in range(len(chosen_indices)):
            a = chosen_indices[i]
            for j in range(i + 1, len(chosen_indices)):
                b = chosen_indices[j]
                if (iso_conflict[a][b] >> p_idx) & 1:
                    return False
        return True
    
    # Helper: check if set is maximal (no point can be added)
    def is_maximal(chosen_mask, chosen_indices):
        for p_idx in range(NUM_POINTS):
            if (chosen_mask >> p_idx) & 1:
                continue
            if can_add(p_idx, chosen_mask, chosen_indices):
                return False
        return True
    
    # Helper: prune set by removing redundant points while maintaining maximality
    def try_prune(chosen_indices):
        if not chosen_indices:
            return chosen_indices
        chosen_set = set(chosen_indices)
        chosen_mask = 0
        for idx in chosen_indices:
            chosen_mask |= (1 << idx)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(chosen_indices)):
                p = chosen_indices[i]
                s_temp = [x for j, x in enumerate(chosen_indices) if j != i]
                s_temp_set = set(s_temp)
                s_temp_mask = 0
                for idx in s_temp:
                    s_temp_mask |= (1 << idx)
                if is_maximal(s_temp_mask, s_temp):
                    chosen_indices = s_temp
                    chosen_set = s_temp_set
                    chosen_mask = s_temp_mask
                    improved = True
                    break
        return chosen_indices
    
    best_set = None
    best_size = float('inf')
    
    # Approach A: Randomized greedy with coverage heuristic
    num_greedy_trials = 300
    
    for trial in range(num_greedy_trials):
        if time.time() > deadline - 15:
            break
            
        random.seed(42 + trial)
        
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        
        points_order = list(range(NUM_POINTS))
        random.shuffle(points_order)
        
        while True:
            if time.time() > deadline - 15:
                break
                
            valid_points = []
            for p_idx in points_order:
                if (chosen_mask >> p_idx) & 1:
                    continue
                if can_add(p_idx, chosen_mask, chosen):
                    valid_points.append(p_idx)
            
            if not valid_points:
                break
            
            if len(chosen) < 2:
                p_idx = random.choice(valid_points)
            else:
                if len(valid_points) > 40:
                    sample = random.sample(valid_points, 40)
                else:
                    sample = valid_points
                
                max_cov = -1
                best_p = sample[0]
                for p_idx in sample:
                    cov = 0
                    for q_idx in valid_points:
                        if p_idx == q_idx:
                            continue
                        blocked = False
                        for a in chosen:
                            if (iso_conflict[a][p_idx] >> q_idx) & 1:
                                blocked = True
                                break
                        if blocked:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_p = p_idx
                p_idx = best_p
            
            chosen.append(p_idx)
            chosen_set.add(p_idx)
            chosen_mask |= (1 << p_idx)
        
        if chosen:
            pruned = try_prune(chosen)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_set = pruned.copy()
    
    # Approach B: Branch and Bound for minimum maximal set
    # We search for the smallest set S such that:
    # 1. S is no-isosceles (no three points form isosceles triangle)
    # 2. S is maximal (no point can be added)
    
    # Key insight: A set S is maximal iff for every point p not in S, 
    # there exist a, b in S such that (a, b, p) is isosceles.
    
    # B&B strategy:
    # - Maintain current set S (chosen)
    # - Maintain "blocked" set: points that are already blocked by S
    # - S is maximal iff blocked covers all points not in S
    # - Lower bound: |S| + (number of unblocked points not in S that need to be covered)
    #   But this is tricky. Better: use the fact that each point in S can block at most K points.
    
    # Simpler B&B: try to build a small maximal set
    # State: (current_set, remaining_candidates)
    # Prune if |current_set| >= best_size
    
    # Let's do a different B&B: start from a candidate set and try to minimize
    # Use local search with bitset acceleration
    
    if best_set and time.time() < deadline - 30:
        # Local search with swap operations
        current_set = best_set.copy()
        current_mask = 0
        for idx in current_set:
            current_mask |= (1 << idx)
        
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            found_improvement = False
            
            # Try removing each point
            for i in range(len(current_set)):
                p = current_set[i]
                s_temp = [x for j, x in enumerate(current_set) if j != i]
                s_temp_mask = 0
                for idx in s_temp:
                    s_temp_mask |= (1 << idx)
                if is_maximal(s_temp_mask, s_temp):
                    current_set = s_temp
                    current_mask = s_temp_mask
                    found_improvement = True
                    break
            
            if found_improvement:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                continue
            
            # Try swap: remove p, add q
            if len(current_set) > 0:
                p = random.choice(current_set)
                s_without_p = [x for x in current_set if x != p]
                s_without_p_mask = 0
                for idx in s_without_p:
                    s_without_p_mask |= (1 << idx)
                
                # Find q that can be added to s_without_p
                candidates = []
                for q_idx in range(NUM_POINTS):
                    if (s_without_p_mask >> q_idx) & 1:
                        continue
                    if can_add(q_idx, s_without_p_mask, s_without_p):
                        candidates.append(q_idx)
                
                if candidates:
                    q = random.choice(candidates)
                    s_new = s_without_p + [q]
                    s_new_mask = s_without_p_mask | (1 << q)
                    
                    # Try to prune s_new
                    s_pruned = try_prune(s_new)
                    if len(s_pruned) < len(current_set):
                        current_set = s_pruned
                        current_mask = 0
                        for idx in current_set:
                            current_mask |= (1 << idx)
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = current_set.copy()
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        chosen_mask = 0
        for p_idx in range(NUM_POINTS):
            if can_add(p_idx, chosen_mask, chosen):
                chosen.append(p_idx)
                chosen_mask |= (1 << p_idx)
        return [all_points[idx] for idx in chosen]
    
    return [all_points[idx] for idx in best_set]
