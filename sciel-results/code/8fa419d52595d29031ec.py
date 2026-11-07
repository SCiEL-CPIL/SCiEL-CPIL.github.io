"""Combined approach: exhaustive bitmask search + local search polish for maximum distinct induced degrees."""
import time
import itertools
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _build_adjacency_bitmasks():
    """Build adjacency as bitmasks for fast degree computation."""
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    return adj


def _popcount(x):
    """Count bits set in integer x."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _distinct_degree_count_bitmask(subset, adj):
    """Count distinct degrees in induced subgraph using bitmask adjacency."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
    degs = set()
    for v in subset:
        degs.add(_popcount(adj[v] & mask))
    return len(degs)


def _local_search_polish(combo, adj, n, minv, deadline, rng):
    """Hill-climbing local search: try single swaps to improve distinct degree count."""
    combo_set = set(combo)
    current_score = _distinct_degree_count_bitmask(combo, adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_in in list(combo):
            for v_out in range(n):
                if v_out in combo_set:
                    continue
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _distinct_degree_count_bitmask(new_combo, adj)
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
    deadline = time.time() + 82.0  # safety margin under 90s wall-clock
    n = P['n']
    minv = P['minimum_vertices']
    adj = _build_adjacency_bitmasks()
    vertices = list(range(n))
    
    best_score = 0
    best_subset = list(range(minv))
    
    # Phase 1: Exhaustive search over all subsets of size >= minv
    # Enumerate by size from largest to smallest
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        # Theoretical max for this size is 'size'
        if best_score >= size:
            break
        for subset in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            score = _distinct_degree_count_bitmask(subset, adj)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                # If we achieved theoretical max for this size, stop
                if best_score == size:
                    return [int(v) for v in best_subset]
    
    # Phase 2: Local search polish on the best subset found
    # Try multiple random restarts around the best solution to see if we can improve
    rng = random.Random(12345)
    polish_deadline = time.time() + 78.0  # leave 12s margin
    
    if time.time() < polish_deadline:
        # Try local search from the best subset
        polished, polished_score = _local_search_polish(list(best_subset), adj, n, minv, polish_deadline, rng)
        if polished_score > best_score:
            best_score = polished_score
            best_subset = polished
        
        # Try a few random restarts with local search
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            random_combo = rng.sample(range(n), len(best_subset))
            polished, polished_score = _local_search_polish(random_combo, adj, n, minv, polish_deadline, rng)
            if polished_score > best_score:
                best_score = polished_score
                best_subset = polished
    
    # Ensure exact Python int types
    return [int(v) for v in best_subset]
