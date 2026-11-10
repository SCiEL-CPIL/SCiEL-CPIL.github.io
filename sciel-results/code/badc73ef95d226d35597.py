import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill
    
    d = 11
    # Generate all vectors with exactly 3 non-zero entries, each ±2
    # Norm squared = 3 * 4 = 12
    candidates = []
    for combo in itertools.combinations(range(d), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * d
            for idx, s in zip(combo, signs):
                v[idx] = 2 * s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)  # Should be 1320
    
    if time.time() > deadline - 10:
        return _get_norm2_points()
    
    # Compute dot products
    dot_prods = candidates @ candidates.T
    
    # Conflict if dot > 6 (since dist_sq = 24 - 2*dot < 12 => dot > 6)
    # dot is multiple of 4, so conflict if dot >= 8
    conflict = dot_prods >= 8
    np.fill_diagonal(conflict, False)
    
    # Build adjacency lists
    adj = [np.where(conflict[i])[0].tolist() for i in range(n)]
    adj_sets = [set(a) for a in adj]
    degrees = np.array([len(a) for a in adj])
    
    if time.time() > deadline - 5:
        return _get_norm2_points()
    
    # Strategy: Greedy + Local Search + BB with coloring
    # Step 1: Greedy coloring-based ordering for initial solution
    # Use a randomized greedy multiple times to get good initial solutions
    
    best_size = 0
    best_set = []
    
    # Multiple greedy passes with different orderings
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation of ordering
        order = np.argsort(-degrees + np.random.randn(n) * 0.1)
        
        selected = []
        available = np.ones(n, dtype=bool)
        
        for idx in order:
            if available[idx]:
                selected.append(idx)
                available[idx] = False
                if adj[idx]:
                    available[adj[idx]] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Step 2: Local search - try to add vertices by removing few others
    # 1-add: find a vertex not in selected that can be added
    selected_set = set(best_set)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 20:
            break
        
        # Try adding each non-selected vertex
        for v in range(n):
            if v in selected_set:
                continue
            
            # Check if v can be added directly
            can_add = True
            for u in adj[v]:
                if u in selected_set:
                    can_add = False
                    break
            
            if can_add:
                selected_set.add(v)
                improved = True
                continue
            
            # Try 1-swap: remove one neighbor u, add v
            # Only beneficial if we can add v and the net effect is +0 or we're optimizing for something else
            # For MIS, 1-swap doesn't increase size. Skip.
    
    best_size = len(selected_set)
    best_set = list(selected_set)
    
    # Step 3: Branch and Bound with greedy coloring
    # Tomita-style algorithm
    
    def greedy_color_bound(cands, adj_list, adj_sets, degrees_arr):
        """Fast greedy coloring to get upper bound on MIS size in subgraph."""
        if not cands:
            return 0
        
        cands_list = list(cands)
        # Order by degree in subgraph (approximate with global degree)
        cands_list.sort(key=lambda x: -degrees_arr[x])
        
        colors = {}
        color_count = 0
        for v in cands_list:
            used_colors = set()
            for u in adj_list[v]:
                if u in colors:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        return color_count
    
    # BB state
    global_best_size = best_size
    global_best_set = best_set[:]
    
    def bb_mis(cands, adj_list, adj_sets, degrees_arr, current_set, time_limit):
        nonlocal global_best_size, global_best_set
        
        if time.time() > time_limit:
            return
        
        if not cands:
            if len(current_set) > global_best_size:
                global_best_size = len(current_set)
                global_best_set = current_set[:]
            return
        
        # Upper bound via greedy coloring
        ub = greedy_color_bound(cands, adj_list, adj_sets, degrees_arr)
        if len(current_set) + ub <= global_best_size:
            return
        
        # Pick branch vertex: highest degree in subgraph
        # Use global degree as proxy
        cands_list = list(cands)
        cands_list.sort(key=lambda x: -degrees_arr[x])
        v = cands_list[-1]
        
        # Branch 1: Include v
        new_cands = [u for u in cands if u != v and u not in adj_sets[v]]
        bb_mis(new_cands, adj_list, adj_sets, degrees_arr, current_set + [v], time_limit)
        
        # Branch 2: Exclude v
        cands_without_v = [u for u in cands if u != v]
        bb_mis(cands_without_v, adj_list, adj_sets, degrees_arr, current_set, time_limit)
    
    # Run BB with time limit
    try:
        bb_mis(list(range(n)), adj, adj_sets, degrees, best_set[:], deadline - 10)
    except RecursionError:
        pass
    except Exception:
        pass
    
    # Use the best found
    final_set = global_best_set if global_best_size > best_size else best_set
    
    if len(final_set) < 2:
        return _get_norm2_points()
    
    points = candidates[final_set]
    
    # Verify
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    min_dist_sq = float('inf')
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq >= max_norm_sq:
        return np.array(points, dtype=np.int64)
    
    return _get_norm2_points()

def _get_norm2_points():
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points, dtype=np.int64)
