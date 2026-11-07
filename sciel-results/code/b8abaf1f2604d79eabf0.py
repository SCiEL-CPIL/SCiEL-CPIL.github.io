"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from functools import lru_cache

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _score_subset(vertices, adj):
    """Score: number of distinct degrees in induced subgraph."""
    vset = set(vertices)
    degs = []
    for v in vertices:
        d = 0
        for u in adj[v]:
            if u in vset:
                d += 1
        degs.append(d)
    return len(set(degs))

def _greedy_construct(adj, n, min_vertices=8, target_size=None):
    """Greedy construction: add vertices that maximize distinct degree count."""
    if target_size is None:
        target_size = min_vertices
    
    current = []
    current_set = set()
    
    # Try all starting vertices and pick the best
    best_initial = None
    best_initial_score = -1
    
    for v in range(n):
        trial = [v]
        trial_set = {v}
        degs = {v: 0}
        score = len(set(degs.values()))
        if score > best_initial_score:
            best_initial_score = score
            best_initial = v
    
    if best_initial is None:
        return list(range(min_vertices))
    
    current = [best_initial]
    current_set = {best_initial}
    
    for _ in range(target_size - 1):
        best_v = None
        best_score = -1
        for v in range(n):
            if v in current_set:
                continue
            new_set = current_set | {v}
            new_vertices = list(new_set)
            # Compute degrees
            degs = {}
            for x in new_vertices:
                d = sum(1 for u in adj[x] if u in new_set)
                degs[x] = d
            score = len(set(degs.values()))
            if score > best_score:
                best_score = score
                best_v = v
        if best_v is not None:
            current.append(best_v)
            current_set.add(best_v)
    
    return current

def _local_search_1swap(current, adj, n, deadline):
    """Local search with 1-swap moves."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current)
        best_swap = None
        best_swap_score = _score_subset(current, adj)
        
        for out_v in current:
            for in_v in range(n):
                if in_v in current_set:
                    continue
                new_set = (current_set - {out_v}) | {in_v}
                new_vertices = list(new_set)
                score = _score_subset(new_vertices, adj)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (out_v, in_v)
        
        if best_swap is not None:
            out_v, in_v = best_swap
            current = [v for v in current if v != out_v]
            current.append(in_v)
            improved = True
    
    return current

def _local_search_2swap(current, adj, n, deadline):
    """Local search with 2-swap moves to escape local optima."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current)
        best_swap = None
        best_swap_score = _score_subset(current, adj)
        
        # 1-swap
        for out_v in current:
            for in_v in range(n):
                if in_v in current_set:
                    continue
                new_set = (current_set - {out_v}) | {in_v}
                new_vertices = list(new_set)
                score = _score_subset(new_vertices, adj)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (out_v, in_v, None, None)
        
        # 2-swap
        for out1_idx, out1 in enumerate(current):
            for out2_idx in range(out1_idx + 1, len(current)):
                out2 = current[out2_idx]
                for in1 in range(n):
                    if in1 in current_set:
                        continue
                    for in2 in range(n):
                        if in2 in current_set or in2 == in1:
                            continue
                        new_set = (current_set - {out1, out2}) | {in1, in2}
                        new_vertices = list(new_set)
                        score = _score_subset(new_vertices, adj)
                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap = (out1, out2, in1, in2)
        
        if best_swap is not None:
            out1, out2, in1, in2 = best_swap
            current = [v for v in current if v != out1 and v != out2]
            current.append(in1)
            current.append(in2)
            improved = True
    
    return current

def _exhaustive_best_size_k(adj, n, k, deadline=None):
    """Exhaustive search over all C(n,k) subsets to find best distinct degree count."""
    best_vertices = None
    best_score = -1
    
    for subset in itertools.combinations(range(n), k):
        if deadline is not None and time.time() > deadline:
            break
        score = _score_subset(subset, adj)
        if score > best_score:
            best_score = score
            best_vertices = list(subset)
    
    return best_vertices, best_score

def _multi_start_local_search(adj, n, min_vertices, deadline, n_starts=20, seed=42):
    """Multi-start local search with diverse initializations."""
    rng = random.Random(seed)
    all_vertices = list(range(n))
    best_overall = None
    best_overall_score = -1
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Diverse initialization
        if start_idx == 0:
            # Greedy start
            current = _greedy_construct(adj, n, min_vertices)
        elif start_idx == 1:
            # Random start
            current = rng.sample(all_vertices, min_vertices)
        elif start_idx == 2:
            # Random start with size min_vertices+1
            current = rng.sample(all_vertices, min(min_vertices+1, n))
        elif start_idx == 3:
            # Random start with size min_vertices+2
            current = rng.sample(all_vertices, min(min_vertices+2, n))
        else:
            # Random start with varying sizes
            size = min_vertices + (start_idx % 4)
            size = min(size, n)
            current = rng.sample(all_vertices, size)
        
        # Local search with 1-swap
        current = _local_search_1swap(current, adj, n, deadline)
        score = _score_subset(current, adj)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = list(current)
        
        # Try 2-swap if time permits
        if time.time() < deadline - 5:
            current2 = _local_search_2swap(current, adj, n, deadline - 5)
            score2 = _score_subset(current2, adj)
            if score2 > best_overall_score:
                best_overall_score = score2
                best_overall = list(current2)
    
    return best_overall, best_overall_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    adj = _build_adj(n, edges)
    
    # Start with greedy construction
    current = _greedy_construct(adj, n, min_vertices)
    current_score = _score_subset(current, adj)
    
    # Try exhaustive search for size 8 (C(17,8) = 24310, fast)
    if time.time() < deadline:
        best8, score8 = _exhaustive_best_size_k(adj, n, 8, deadline)
        if best8 is not None and score8 > current_score:
            current = best8
            current_score = score8
    
    # Try exhaustive search for size 9 (C(17,9) = 24310, same as C(17,8))
    if time.time() < deadline - 5:
        best9, score9 = _exhaustive_best_size_k(adj, n, 9, deadline - 5)
        if best9 is not None and score9 > current_score:
            current = best9
            current_score = score9
    
    # Multi-start local search with diverse initializations
    if time.time() < deadline - 10:
        ms_best, ms_score = _multi_start_local_search(adj, n, min_vertices, deadline - 10, n_starts=30, seed=42)
        if ms_best is not None and ms_score > current_score:
            current = ms_best
            current_score = ms_score
    
    # Final local search polish with 1-swap
    if time.time() < deadline:
        current = _local_search_1swap(current, adj, n, deadline)
        current_score = _score_subset(current, adj)
    
    # Try 2-swap polish if time permits
    if time.time() < deadline - 3:
        current2 = _local_search_2swap(current, adj, n, deadline - 3)
        if _score_subset(current2, adj) > current_score:
            current = current2
    
    # Ensure we have at least min_vertices
    if len(current) < min_vertices:
        current_set = set(current)
        for v in range(n):
            if v not in current_set and len(current) < min_vertices:
                current.append(v)
                current_set.add(v)
    
    # Return as list of Python ints
    return [int(v) for v in current]

# EVOLVE-BLOCK-END
