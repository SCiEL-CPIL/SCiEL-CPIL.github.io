import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

# Known good solution with 10 edges
_KNOWN_GOOD = [
    [11, 0], [11, 2], [11, 4], [11, 6], [11, 8],
    [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]
]

def _validate(P, added_edges):
    n = P['n']
    # 1. Check no duplicates and not in base
    base_set = set(tuple(sorted(e)) for e in P['edges'])
    edge_set = set()
    for e in added_edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            return False
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int) or isinstance(u, bool) or isinstance(v, bool):
            return False
        if u == v:
            return False
        t = tuple(sorted((u, v)))
        if t in base_set:
            return False
        if t in edge_set:
            return False
        edge_set.add(t)

    # 2. Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)

    # 3. Check triangle-free
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    return False

    # 4. Check diameter <= 4
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    if dist[neighbor] > 4:
                        return False
                    queue.append(neighbor)
    return True

def _get_candidate_edges(P):
    """Generate all valid bipartite edges not in base path."""
    n = P['n']
    base_set = set(tuple(sorted(e)) for e in P['edges'])
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            # Must be bipartite: one even, one odd
            if (u % 2) == (v % 2):
                continue
            t = (u, v)
            if t not in base_set:
                candidates.append([u, v])
    return candidates

def _local_search(P, candidates, initial_edges, max_iter=5000):
    """Try to reduce number of edges while maintaining validity."""
    n = P['n']
    current_edges = list(initial_edges)
    current_set = set(tuple(sorted(e)) for e in current_edges)
    
    if not _validate(P, current_edges):
        return None

    best_edges = current_edges
    best_len = len(current_edges)
    
    for _ in range(max_iter):
        if not current_edges:
            break
            
        # Try to remove a random edge
        idx = random.randint(0, len(current_edges) - 1)
        removed_edge = current_edges[idx]
        new_edges = [e for i, e in enumerate(current_edges) if i != idx]
        
        if _validate(P, new_edges):
            current_edges = new_edges
            best_edges = current_edges
            best_len = len(current_edges)
            continue
            
        # If removal fails, try swapping with a random candidate
        # Pick a random candidate not in current edges
        avail_candidates = [c for c in candidates if tuple(sorted(c)) not in current_set]
        if avail_candidates:
            swap_edge = random.choice(avail_candidates)
            new_edges = [e for i, e in enumerate(current_edges) if i != idx]
            new_edges.append(swap_edge)
            
            if _validate(P, new_edges):
                current_edges = new_edges
                best_edges = current_edges
                best_len = len(current_edges)
                
    return best_edges

def construct():
    start_time = time.time()
    deadline = 80.0
    
    # 1. Check known good (10 edges)
    if _validate(P, _KNOWN_GOOD):
        best_solution = _KNOWN_GOOD
    else:
        best_solution = None

    candidates = _get_candidate_edges(P)
    
    # 2. Try to find a solution with fewer than 10 edges
    # We will run local search starting from the known good solution
    # and also try random starts if time permits.
    
    if best_solution:
        # Run local search from known good
        improved = _local_search(P, candidates, best_solution, max_iter=2000)
        if improved and len(improved) < len(best_solution):
            best_solution = improved

    # Try to find solutions with k=9, 8, etc.
    # We can do this by random restarts
    target_len = len(best_solution) - 1 if best_solution else 9
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random start: pick random number of edges
        # We want to find a valid configuration with few edges.
        # Start with a random set of size 9
        if target_len > 0:
            random_edges = random.sample(candidates, target_len)
            if _validate(P, random_edges):
                best_solution = random_edges
                target_len -= 1
                continue
                
            # If random start fails, try local search from it?
            # Or just try to improve a random valid start?
            # Better: Start from a valid config (known good) and try to reduce.
            # The _local_search already does this.
            # Let's try a different heuristic: 
            # Start with 9 random edges, then try to fix validity by swapping.
            # This is complex. Let's stick to reducing from 10.
            
        # If we have time, try more iterations on the best solution
        if best_solution:
            improved = _local_search(P, candidates, best_solution, max_iter=1000)
            if improved and len(improved) < len(best_solution):
                best_solution = improved

    if best_solution is None:
        return _KNOWN_GOOD
        
    return best_solution
# EVOLVE-BLOCK-END
