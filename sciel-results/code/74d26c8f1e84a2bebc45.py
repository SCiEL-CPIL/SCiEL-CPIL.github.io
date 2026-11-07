"""Simulated annealing + local search for maximum distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _compute_distinct_degrees(combo, adj, n):
    """Compute number of distinct degrees in induced subgraph on combo."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    degs = set()
    for v in combo:
        x = adj[v] & mask
        c = 0
        while x:
            x &= x - 1
            c += 1
        degs.add(c)
    return len(degs)


def _local_search(combo, adj, n, minv, deadline, rng):
    """Hill-climbing local search: try single swaps to improve distinct degree count."""
    combo_set = set(combo)
    current_score = _compute_distinct_degrees(combo, adj, n)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_in in combo:
            for v_out in range(n):
                if v_out in combo_set:
                    continue
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _compute_distinct_degrees(new_combo, adj, n)
                if new_score > current_score:
                    combo = new_combo
                    combo_set = set(combo)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    return combo, current_score


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s wall-clock
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    rng = random.Random(42)  # deterministic for reproducibility
    
    best_vertices = None
    best_score = -1
    
    # Try multiple subset sizes, starting from larger (more room for distinct degrees)
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        if best_score >= size:
            # Can't do better than size distinct degrees with size vertices
            break
        
        # Multiple restarts with simulated annealing
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial subset of given size
            combo = rng.sample(range(n), size)
            current_score = _compute_distinct_degrees(combo, adj, n)
            
            if current_score > best_score:
                best_score = current_score
                best_vertices = list(combo)
            
            # Simulated annealing
            T = 1.0
            T_min = 0.001
            cooling = 0.95
            max_iters = 200
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if T < T_min:
                    break
                
                # Generate neighbor: swap one vertex in with one vertex out
                v_in = rng.choice(combo)
                candidates_out = [v for v in range(n) if v not in combo]
                if not candidates_out:
                    break
                v_out = rng.choice(candidates_out)
                
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _compute_distinct_degrees(new_combo, adj, n)
                
                delta = new_score - current_score
                if delta > 0:
                    combo = new_combo
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_vertices = list(combo)
                else:
                    # Accept with probability exp(delta/T)
                    if delta < 0:
                        import math
                        if rng.random() < math.exp(delta / T):
                            combo = new_combo
                            current_score = new_score
                
                T *= cooling
            
            # Local search from current position
            combo, local_score = _local_search(combo, adj, n, minv, deadline, rng)
            if local_score > best_score:
                best_score = local_score
                best_vertices = list(combo)
            
            # If we hit the theoretical max for this size, stop
            if best_score == size:
                break
    
    # Fallback if nothing found
    if best_vertices is None:
        best_vertices = list(range(minv))
    
    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
