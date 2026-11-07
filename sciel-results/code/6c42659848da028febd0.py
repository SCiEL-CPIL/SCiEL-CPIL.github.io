"""Combined exhaustive search + simulated annealing + local search for maximum distinct degrees."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _popcount(x):
    """Fast popcount using bit manipulation."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _compute_distinct_degrees(combo, adj):
    """Compute number of distinct degrees in induced subgraph on combo."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    degs = set()
    for v in combo:
        x = adj[v] & mask
        degs.add(_popcount(x))
    return len(degs)


def _local_search_2swap(combo, adj, n, minv, deadline, rng, tabu_len=10):
    """Enhanced local search with 1-swap and 2-swap moves plus tabu list."""
    combo_set = set(combo)
    current_score = _compute_distinct_degrees(combo, adj)
    tabu = {}  # vertex -> iteration when it becomes non-tabu
    it_count = 0
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        it_count += 1
        
        # Try 1-swap moves
        for v_in in combo:
            for v_out in range(n):
                if v_out in combo_set:
                    continue
                # Tabu check
                if v_out in tabu and it_count < tabu[v_out]:
                    continue
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _compute_distinct_degrees(new_combo, adj)
                if new_score > current_score:
                    combo = new_combo
                    combo_set = set(combo)
                    current_score = new_score
                    tabu[v_in] = it_count + tabu_len
                    improved = True
                    break
            if improved:
                break
        
        # If no 1-swap improvement, try 2-swap moves
        if not improved:
            best_2swap = None
            best_2score = current_score
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    v1, v2 = combo[i], combo[j]
                    for v3 in range(n):
                        if v3 in combo_set:
                            continue
                        if v3 in tabu and it_count < tabu[v3]:
                            continue
                        for v4 in range(n):
                            if v4 in combo_set or v4 == v3:
                                continue
                            if v4 in tabu and it_count < tabu[v4]:
                                continue
                            new_combo = [v for k, v in enumerate(combo) if k not in (i, j)]
                            new_combo.append(v3)
                            new_combo.append(v4)
                            new_score = _compute_distinct_degrees(new_combo, adj)
                            if new_score > best_2score:
                                best_2score = new_score
                                best_2swap = (v1, v2, v3, v4, new_combo)
            
            if best_2swap and best_2score > current_score:
                v1, v2, v3, v4, new_combo = best_2swap
                combo = new_combo
                combo_set = set(combo)
                current_score = best_2score
                tabu[v1] = it_count + tabu_len
                tabu[v2] = it_count + tabu_len
                improved = True
    
    return combo, current_score


def _sa_search(combo, adj, n, minv, deadline, rng, best_score, best_vertices):
    """Simulated annealing with enhanced neighborhood."""
    combo_set = set(combo)
    current_score = _compute_distinct_degrees(combo, adj)
    
    if current_score > best_score:
        best_score = current_score
        best_vertices = list(combo)
    
    # Initial local search
    combo, current_score = _local_search_2swap(combo, adj, n, minv, deadline, rng)
    if current_score > best_score:
        best_score = current_score
        best_vertices = list(combo)
    
    # Simulated annealing
    T = 1.5
    T_min = 0.005
    cooling = 0.985
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        # Generate neighbor: random swap
        v_in = rng.choice(combo)
        candidates_out = [v for v in range(n) if v not in combo_set]
        if not candidates_out:
            break
        v_out = rng.choice(candidates_out)
        
        new_combo = list(combo)
        new_combo.remove(v_in)
        new_combo.append(v_out)
        new_score = _compute_distinct_degrees(new_combo, adj)
        
        delta = new_score - current_score
        if delta > 0:
            combo = new_combo
            combo_set = set(combo)
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vertices = list(combo)
        else:
            if delta < 0:
                if rng.random() < math.exp(delta / T):
                    combo = new_combo
                    combo_set = set(combo)
                    current_score = new_score
        
        T *= cooling
    
    # Final local search after SA
    combo, current_score = _local_search_2swap(combo, adj, n, minv, deadline, rng)
    if current_score > best_score:
        best_score = current_score
        best_vertices = list(combo)
    
    return best_score, best_vertices


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
    
    vertices = list(range(n))
    best_score = -1
    best_vertices = list(range(minv))  # fallback
    
    # Phase 1: Exhaustive search for sizes 8-10
    for size in range(minv, 11):
        if time.time() > deadline:
            break
        if best_score >= size:
            break
        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                return [int(v) for v in best_vertices]
            score = _compute_distinct_degrees(combo, adj)
            if score > best_score:
                best_score = score
                best_vertices = list(combo)
                if score == size:
                    break
    
    # Phase 2: For larger sizes (11-17), use enhanced SA + local search
    for size in range(11, n + 1):
        if time.time() > deadline:
            break
        if best_score >= size:
            break
        
        # Multiple restarts with different seeds
        for restart in range(50):
            if time.time() > deadline:
                break
            
            rng = random.Random(42 + size * 100 + restart)
            
            # Smart initialization: start from best known or random
            if restart == 0:
                # Start from a greedy construction
                combo = list(range(minv))
                # Try to extend greedily
                for v in range(minv, n):
                    if len(combo) >= size:
                        break
                    combo.append(v)
                if len(combo) > size:
                    combo = combo[:size]
            else:
                combo = rng.sample(range(n), size)
            
            best_score, best_vertices = _sa_search(combo, adj, n, minv, deadline, rng, best_score, best_vertices)
    
    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
