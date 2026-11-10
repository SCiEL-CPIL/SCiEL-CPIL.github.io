import numpy as np
import time
from itertools import combinations
from functools import lru_cache

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 570
    
    # Generate all 4-subsets of {0, ..., 10}
    four_subsets = list(combinations(range(d), 4))
    n = len(four_subsets)  # 330
    
    # Precompute incompatibility: edge if |S_i ∩ S_j| >= 3
    # Build adjacency as bitmasks for fast operations
    adj = [0] * n  # adj[i] is bitmask of vertices incompatible with i
    
    for i in range(n):
        si = set(four_subsets[i])
        for j in range(i + 1, n):
            sj = set(four_subsets[j])
            if len(si & sj) >= 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute degree
    degree = [bin(adj[i]).count('1') for i in range(n)]
    
    # Greedy initialization with multiple orderings
    def greedy_init(order):
        selected = set()
        for v in order:
            if all(not (adj[v] & (1 << s)) for s in selected):
                selected.add(v)
        return selected
    
    # Try several greedy orderings
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Degree-based greedy (pick lowest degree first)
    order_deg = sorted(range(n), key=lambda x: degree[x])
    s = greedy_init(order_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Random orderings
    for _ in range(50):
        order = rng.permutation(n)
        s = greedy_init(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add vertices
    def local_search(selected, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any vertex
            for v in range(n):
                if v in selected:
                    continue
                if all(not (adj[v] & (1 << s)) for s in selected):
                    selected.add(v)
                    improved = True
                    break
            # Try 1-swap: remove one, add up to 2
            if time.time() > time_limit:
                break
            for rem in list(selected):
                if time.time() > time_limit:
                    break
                new_sel = selected - {rem}
                added = 0
                for v in range(n):
                    if v in new_sel or v == rem:
                        continue
                    if all(not (adj[v] & (1 << s)) for s in new_sel):
                        new_sel.add(v)
                        added += 1
                        if added >= 2:
                            break
                if len(new_sel) > len(selected):
                    selected = new_sel
                    improved = True
                    break
        return selected
    
    # Branch and bound for maximum independent set
    # State: (available_mask, selected_mask, current_size)
    # Use folding: if vertex v has all neighbors in a clique, we can fold
    
    def branch_and_bound(available, selected, current_size, time_limit):
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        # Upper bound: current_size + |available|
        if current_size + bin(available).count('1') <= best_size:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(selected)
            return
        
        # Pick a vertex with minimum degree in available subgraph
        # Try to find a good pivot
        best_v = -1
        best_deg = n + 1
        for v in range(n):
            if available & (1 << v):
                deg_in_avail = bin(adj[v] & available).count('1')
                if deg_in_avail < best_deg:
                    best_deg = deg_in_avail
                    best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_selected = selected | (1 << best_v)
        new_available = available & ~(1 << best_v) & ~adj[best_v]
        branch_and_bound(new_available, new_selected, current_size + 1, time_limit)
        
        if time.time() > time_limit:
            return
        
        # Branch 2: exclude best_v
        new_available = available & ~(1 << best_v)
        branch_and_bound(new_available, selected, current_size, time_limit)
    
    # Run branch and bound with the best greedy solution as initial bound
    initial_selected_mask = 0
    for v in best_set:
        initial_selected_mask |= (1 << v)
    
    # Start B&B from a reduced problem
    # Available: all vertices not in best_set and not adjacent to any in best_set
    available = (1 << n) - 1
    for v in best_set:
        available &= ~(1 << v)
        available &= ~adj[v]
    
    # But we can also try to replace some selected vertices
    # For now, just try to add to the current solution
    branch_and_bound(available, initial_selected_mask, best_size, deadline - 60)
    
    # If B&B didn't improve, try more local search
    if time.time() < deadline - 30:
        improved = local_search(set(best_set), deadline - 20)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Construct points from best_set
    # For each selected 4-subset, pick one of 16 sign patterns
    # We need to ensure no two vectors have Hamming distance < 4
    # Since we picked at most one per 4-subset, and subsets share ≤ 2 elements,
    # any two vectors differ in at least 4 positions (the 4 positions of one subset 
    # that are not in the other, or vice versa)
    # Actually, if |S_i ∩ S_j| ≤ 2, then |S_i Δ S_j| ≥ 4, so they differ in ≥ 4 positions
    # regardless of signs. So any sign pattern works!
    
    points = []
    for idx in sorted(best_set):
        support = four_subsets[idx]
        # Use all +1 for simplicity (or any fixed pattern)
        v = np.zeros(d, dtype=np.int64)
        for pos in support:
            v[pos] = 1
        points.append(v)
    
    # Verify
    if len(points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    pts_array = np.array(points, dtype=np.int64)
    
    # Quick validation
    n_pts = len(pts_array)
    if n_pts > 0:
        norms = np.sum(pts_array ** 2, axis=1)
        max_norm = int(norms.max())
        
        min_dist = float('inf')
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = pts_array[i] - pts_array[j]
                dist_sq = int(np.sum(diff ** 2))
                if dist_sq < min_dist:
                    min_dist = dist_sq
        
        if min_dist < max_norm:
            # Invalid - fallback
            points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
            pts_array = np.array(points, dtype=np.int64)
    
    return np.around(pts_array).astype(np.int64)
