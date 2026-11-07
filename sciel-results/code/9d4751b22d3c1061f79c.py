"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _induced_degrees(subset, adj):
    """Compute degrees of each vertex in subset within the induced subgraph."""
    s = set(subset)
    degs = []
    for v in subset:
        d = sum(1 for u in s if u != v and u in adj[v])
        degs.append(d)
    return degs

def _score(subset, adj):
    degs = _induced_degrees(subset, adj)
    return len(set(degs))

def _greedy_construct(adj, deadline):
    n = P['n']
    min_v = P['minimum_vertices']
    best_score = 0
    best_subset = list(range(min_v))
    
    # Start with all vertices, try to remove to maximize distinct degrees
    # Actually, let's try both: start with min_v and add, or start with all and remove
    
    # Strategy 1: Start with min_v vertices, greedily add
    current = list(range(min_v))
    for step in range(n - min_v):
        if time.time() > deadline:
            break
        best_add = -1
        best_s = _score(current, adj)
        for v in range(n):
            if v in current:
                continue
            trial = current + [v]
            s = _score(trial, adj)
            if s > best_s:
                best_s = s
                best_add = v
        if best_add >= 0:
            current.append(best_add)
        if _score(current, adj) > best_score:
            best_score = _score(current, adj)
            best_subset = list(current)
        if best_score >= n:  # Can't do better than n distinct degrees
            break
    
    # Strategy 2: Start with all vertices, try removing
    current = list(range(n))
    for step in range(n - min_v):
        if time.time() > deadline:
            break
        best_remove = -1
        best_s = _score(current, adj)
        for v in range(n):
            trial = [x for x in current if x != v]
            if len(trial) < min_v:
                continue
            s = _score(trial, adj)
            if s > best_s:
                best_s = s
                best_remove = v
        if best_remove >= 0:
            current.remove(best_remove)
        if _score(current, adj) > best_score:
            best_score = _score(current, adj)
            best_subset = list(current)
        if best_score >= n:
            break
    
    # Strategy 3: Random restarts with local search
    import random
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        # Random initial subset of size min_v to min_v+3
        size = min_v + random.randint(0, 3)
        size = min(size, n)
        current = random.sample(range(n), size)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            cur_s = _score(current, adj)
            # Try adding a vertex
            for v in range(n):
                if v in current:
                    continue
                trial = current + [v]
                s = _score(trial, adj)
                if s > cur_s:
                    current = trial
                    cur_s = s
                    improved = True
                    if _score(current, adj) > best_score:
                        best_score = _score(current, adj)
                        best_subset = list(current)
                    if best_score >= n:
                        return best_subset
                    break
            if time.time() > deadline:
                break
            # Try removing a vertex (if size > min_v)
            if len(current) > min_v:
                for v in list(current):
                    trial = [x for x in current if x != v]
                    s = _score(trial, adj)
                    if s > cur_s:
                        current = trial
                        cur_s = s
                        improved = True
                        if _score(current, adj) > best_score:
                            best_score = _score(current, adj)
                            best_subset = list(current)
                        if best_score >= n:
                            return best_subset
                        break
            # Try swapping
            if time.time() > deadline:
                break
            cur_s = _score(current, adj)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for v in range(n):
                    if v in current:
                        continue
                    trial = current[:i] + [v] + current[i+1:]
                    s = _score(trial, adj)
                    if s > cur_s:
                        current = trial
                        cur_s = s
                        improved = True
                        if _score(current, adj) > best_score:
                            best_score = _score(current, adj)
                            best_subset = list(current)
                        if best_score >= n:
                            return best_subset
                        break
                if time.time() > deadline:
                    break
    
    return best_subset

def _exhaustive_small(adj, deadline):
    """For small n, do more exhaustive search."""
    n = P['n']
    min_v = P['minimum_vertices']
    best_score = 0
    best_subset = list(range(min_v))
    
    # Try all subsets of size min_v to min_v+4
    for size in range(min_v, min(min_v + 5, n + 1)):
        if time.time() > deadline:
            break
        for subset in combinations(range(n), size):
            s = _score(list(subset), adj)
            if s > best_score:
                best_score = s
                best_subset = list(subset)
            if best_score >= n:
                return best_subset
    return best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin for 90s limit
    
    adj = _build_adj()
    
    # First, try exhaustive for smaller sizes
    best = _exhaustive_small(adj, deadline)
    best_s = _score(best, adj)
    
    # Then greedy refinement
    best2 = _greedy_construct(adj, deadline)
    s2 = _score(best2, adj)
    
    if s2 > best_s:
        best = best2
        best_s = s2
    
    # Ensure it's a list of Python ints
    result = [int(v) for v in best]
    return result

# EVOLVE-BLOCK-END
