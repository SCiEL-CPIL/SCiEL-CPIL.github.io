"""Improved no-isosceles set solver using branch-and-bound with bitset pruning."""
import random
import numpy as np
from collections import defaultdict

N = 64

def construct():
    # Generate all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    # Precompute squared distances between all pairs
    # For each point i, group other points by squared distance
    neighbors_by_dist = [defaultdict(list) for _ in range(num_points)]
    
    for i in range(num_points):
        x1, y1 = points[i]
        for j in range(num_points):
            if i == j:
                continue
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + dy * dy
            neighbors_by_dist[i][d2].append(j)
    
    # Build conflict graph as bitsets
    # adj[i] is an integer where bit j is set if i and j conflict
    # Two points conflict if there exists a third point forming an isosceles triangle
    adj = [0] * num_points
    
    def set_edge(u, v):
        if u != v:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Type 1: For each apex w, all pairs of points at the same distance from w conflict
    for w in range(num_points):
        for d2, lst in neighbors_by_dist[w].items():
            k = len(lst)
            if k < 2:
                continue
            for a in range(k):
                u = lst[a]
                for b in range(a + 1, k):
                    v = lst[b]
                    set_edge(u, v)
    
    # Type 2: For each apex u, and each point w at distance d2 from u,
    # w conflicts with all other points v at the same distance d2 from u
    # This is actually covered by Type 1 above when we consider u as the apex
    # Let me verify: In Type 1, for apex u, all pairs in neighbors_by_dist[u][d2] conflict.
    # That means w and v both in neighbors_by_dist[u][d2] will have an edge between them.
    # So Type 2 is redundant with Type 1. Good.
    
    # Precompute static degrees
    static_deg = [bin(adj[i]).count('1') for i in range(num_points)]
    
    # Sort points by degree (ascending) for better branching
    order = sorted(range(num_points), key=lambda i: static_deg[i])
    
    # Branch and bound solver
    best_size = 0
    best_set = set()
    
    # Use a timeout-aware approach
    import time
    start_time = time.time()
    TIMEOUT = 280  # Leave some margin for safety
    
    # Heuristic: greedy initial solution
    def greedy_solution():
        remaining = set(range(num_points))
        selected = set()
        while remaining:
            # Pick node with minimum degree in remaining
            min_deg_node = None
            min_deg = float('inf')
            for u in remaining:
                # Count neighbors in remaining
                deg = 0
                mask = adj[u]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx in remaining:
                        deg += 1
                    m ^= lsb
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
            selected.add(min_deg_node)
            # Remove min_deg_node and its neighbors
            to_remove = {min_deg_node}
            m = adj[min_deg_node]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in remaining:
                    to_remove.add(idx)
                m ^= lsb
            remaining -= to_remove
        return selected
    
    # Get initial greedy solution
    init_sol = greedy_solution()
    best_size = len(init_sol)
    best_set = init_sol.copy()
    
    # Local search improvement
    def local_search(start_set):
        sel = set(start_set)
        sel_list = list(sel)
        random.shuffle(sel_list)
        
        # Build conflict count for each node
        conflict_count = [0] * num_points
        for u in sel:
            m = adj[u]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in sel:
                    conflict_count[idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            improved = False
            # Try to add a node
            for u in range(num_points):
                if u not in sel and conflict_count[u] == 0:
                    sel.add(u)
                    conflict_count[u] = 0
                    m = adj[u]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in sel:
                            conflict_count[idx] += 1
                        m ^= lsb
                    improved = True
                    if time.time() - start_time > TIMEOUT:
                        return sel
            
            # Try swaps: remove one node, add another
            for u in sel_list:
                if u not in sel:
                    continue
                # Count how many nodes in sel conflict with u
                m = adj[u]
                conflicts = 0
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx in sel:
                        conflicts += 1
                    m ^= lsb
                
                if conflicts == 0:
                    continue
                
                # Try removing u and adding a node that was previously blocked
                # Find nodes that only conflict with u (among sel)
                candidates = []
                for v in range(num_points):
                    if v in sel or v == u:
                        continue
                    # Check if v only conflicts with u in sel
                    m = adj[v]
                    only_u = True
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx in sel and idx != u:
                            only_u = False
                            break
                        m ^= lsb
                    if only_u:
                        candidates.append(v)
                
                for v in candidates:
                    # Remove u, add v
                    sel.remove(u)
                    sel.add(v)
                    # Update conflict counts
                    m = adj[u]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in sel:
                            conflict_count[idx] -= 1
                        m ^= lsb
                    m = adj[v]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if idx not in sel:
                            conflict_count[idx] += 1
                        m ^= lsb
                    conflict_count[v] = 0
                    improved = True
                    if time.time() - start_time > TIMEOUT:
                        return sel
                    break
                if improved:
                    break
        
        return sel
    
    # Run local search with multiple restarts
    for seed in range(20):
        if time.time() - start_time > TIMEOUT:
            break
        random.seed(seed)
        
        # Create a perturbed initial solution
        sol = set(init_sol)
        # Randomly remove some points
        to_remove = random.sample(list(sol), min(5, len(sol)))
        for u in to_remove:
            sol.remove(u)
        
        improved_sol = local_search(sol)
        if len(improved_sol) > best_size:
            best_size = len(improved_sol)
            best_set = improved_sol.copy()
    
    # Additional greedy with different heuristics
    for seed in range(10):
        if time.time() - start_time > TIMEOUT:
            break
        random.seed(seed + 100)
        
        # Randomized greedy
        remaining = set(range(num_points))
        selected = set()
        while remaining:
            # Pick randomly from low-degree nodes
            candidates = list(remaining)
            random.shuffle(candidates)
            best_u = None
            best_deg = float('inf')
            for u in candidates[:20]:  # Only consider first 20 for speed
                deg = 0
                m = adj[u]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if idx in remaining:
                        deg += 1
                    m ^= lsb
                if deg < best_deg:
                    best_deg = deg
                    best_u = u
            
            if best_u is None:
                best_u = random.choice(list(remaining))
            
            selected.add(best_u)
            to_remove = {best_u}
            m = adj[best_u]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in remaining:
                    to_remove.add(idx)
                m ^= lsb
            remaining -= to_remove
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    return [points[i] for i in best_set]
