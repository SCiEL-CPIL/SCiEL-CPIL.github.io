"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import random
from collections import defaultdict

def _build_adjacency():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_ham_cycle(adj, used_edges, start, deadline, n):
    """Find a Hamiltonian cycle starting/ending at `start` avoiding used_edges."""
    path = [start]
    visited = [False] * n
    visited[start] = True
    path_set = {start}
    
    # Precompute available neighbors (undirected edge keys)
    used_keys = set()
    for u, v in used_edges:
        used_keys.add(_edge_key(u, v))
    
    avail = {}
    for v in range(n):
        avail[v] = [u for u in adj[v] if _edge_key(v, u) not in used_keys]
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_keys and start in adj[last]:
                return True
            return False
        
        current = path[-1]
        # Warnsdorff: prefer neighbors with fewer available unvisited neighbors
        candidates = []
        for nb in avail[current]:
            if visited[nb]:
                continue
            if nb == start and len(path) < n:
                continue
            # Count unvisited neighbors of nb
            unvisited_count = 0
            for nn in avail[nb]:
                if not visited[nn] and nn != current:
                    unvisited_count += 1
            candidates.append((unvisited_count, nb))
        
        candidates.sort()
        
        for _, nb in candidates:
            # Dead-end check: if nb has no unvisited neighbors (and we're not at n-1), skip
            if len(path) < n - 1:
                has_unvisited = False
                for nn in avail[nb]:
                    if not visited[nn] and nn != current:
                        has_unvisited = True
                        break
                if not has_unvisited:
                    continue
            
            path.append(nb)
            visited[nb] = True
            path_set.add(nb)
            
            if backtrack():
                return True
            
            path.pop()
            visited[nb] = False
            path_set.discard(nb)
        
        return False
    
    if backtrack():
        return path
    return None

def _try_find_cycles(adj, n, used_edges, deadline, max_cycles, start_order):
    """Try to find up to max_cycles edge-disjoint Hamiltonian cycles."""
    cycles = []
    local_used = set(used_edges)
    
    for start in start_order:
        if time.time() > deadline:
            break
        if len(cycles) >= max_cycles:
            break
        
        cycle = _find_ham_cycle(adj, local_used, start, deadline, n)
        if cycle is not None:
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                key = _edge_key(u, v)
                if key in local_used:
                    valid = False
                    break
                cycle_edges.add(key)
            
            if valid:
                cycles.append(cycle)
                local_used.update(cycle_edges)
    
    return cycles, local_used

def construct():
    t0 = time.time()
    deadline = t0 + 55
    adj = _build_adjacency()
    n = P['n']
    
    # Strategy 1: Natural order
    cycles1, _ = _try_find_cycles(adj, n, set(), deadline, 10, list(range(n)))
    best = cycles1
    
    # Strategy 2: Reversed order
    if time.time() < deadline and len(best) < 10:
        cycles2, _ = _try_find_cycles(adj, n, set(), deadline, 10, list(range(n - 1, -1, -1)))
        if len(cycles2) > len(best):
            best = cycles2
    
    # Strategy 3: Randomized restarts
    for seed in range(1, 100):
        if time.time() > deadline:
            break
        if len(best) >= 10:
            break
        
        random.seed(seed)
        start_order = list(range(n))
        random.shuffle(start_order)
        
        cycles_try, _ = _try_find_cycles(adj, n, set(), deadline, 10, start_order)
        if len(cycles_try) > len(best):
            best = cycles_try
    
    # Ensure types are correct
    result = [list(c) for c in best]
    for c in result:
        for v in c:
            if not isinstance(v, int) or isinstance(v, bool):
                return [list(range(n))]
    
    if not result:
        return [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
