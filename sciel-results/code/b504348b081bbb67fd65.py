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

def _find_ham_cycle_fast(adj, used_edges, start, deadline):
    """Find a Hamiltonian cycle starting/ending at `start` avoiding used_edges."""
    n = P['n']
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute available neighbors for each vertex (excluding used edges)
    avail = {}
    for v in range(n):
        avail[v] = [u for u in adj[v] if (v, u) not in used_edges and (u, v) not in used_edges]
    
    # Sort neighbors by remaining degree for better pruning
    for v in range(n):
        avail[v].sort(key=lambda x: len(avail[x]))
    
    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            if (last, start) not in used_edges and (start, last) not in used_edges:
                if start in adj[last]:
                    return True
            return False
        
        current = path[-1]
        for nb in avail[current]:
            if nb == start and len(path) < n:
                continue
            if visited[nb]:
                continue
            path.append(nb)
            visited[nb] = True
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        
        return False
    
    if backtrack():
        return path
    return None

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    adj = _build_adjacency()
    n = P['n']
    cycles = []
    used_edges = set()
    
    # Strategy 1: Try all start vertices in natural order
    for start in range(n):
        if time.time() > deadline:
            break
        if len(cycles) >= 10:
            break
        
        cycle = _find_ham_cycle_fast(adj, used_edges, start, deadline)
        if cycle is not None:
            valid = True
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if (u, v) in used_edges or (v, u) in used_edges:
                    valid = False
                    break
                cycle_edges.add((u, v))
                cycle_edges.add((v, u))
            
            if valid:
                cycles.append(cycle)
                used_edges.update(cycle_edges)
    
    # Strategy 2: If fewer than 5 found, try reversed vertex ordering
    if len(cycles) < 5:
        # Reset and try different ordering
        cycles2 = []
        used_edges2 = set()
        for start in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            if len(cycles2) >= 10:
                break
            
            cycle = _find_ham_cycle_fast(adj, used_edges2, start, deadline)
            if cycle is not None:
                valid = True
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if (u, v) in used_edges2 or (v, u) in used_edges2:
                        valid = False
                        break
                    cycle_edges.add((u, v))
                    cycle_edges.add((v, u))
                
                if valid:
                    cycles2.append(cycle)
                    used_edges2.update(cycle_edges)
        
        if len(cycles2) > len(cycles):
            cycles = cycles2
            used_edges = used_edges2
    
    # Strategy 3: Randomized restarts with different seeds
    best_count = len(cycles)
    for seed in range(1, 50):
        if time.time() > deadline:
            break
        if best_count >= 10:
            break
        
        random.seed(seed)
        start_order = list(range(n))
        random.shuffle(start_order)
        
        cycles_try = []
        used_edges_try = set()
        
        for start in start_order:
            if time.time() > deadline:
                break
            if len(cycles_try) >= 10:
                break
            
            cycle = _find_ham_cycle_fast(adj, used_edges_try, start, deadline)
            if cycle is not None:
                valid = True
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if (u, v) in used_edges_try or (v, u) in used_edges_try:
                        valid = False
                        break
                    cycle_edges.add((u, v))
                    cycle_edges.add((v, u))
                
                if valid:
                    cycles_try.append(cycle)
                    used_edges_try.update(cycle_edges)
        
        if len(cycles_try) > best_count:
            cycles = cycles_try
            used_edges = used_edges_try
            best_count = len(cycles)
    
    # Ensure types are correct
    result = [list(c) for c in cycles]
    for c in result:
        for v in c:
            if not isinstance(v, int) or isinstance(v, bool):
                return [list(range(n))]
    
    if not result:
        return [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
