"""Improved program: prioritize small cycles for higher count via greedy set packing."""
# EVOLVE-BLOCK-START
import time
import sys
import math
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_circulant_cycles(n, all_edges, used_edges, deadline):
    """Find all valid circulant Hamiltonian cycles with unused edges."""
    cycles = []
    for k in range(1, n):
        if time.time() > deadline:
            break
        if _gcd(k, n) != 1:
            continue
        
        valid = True
        for i in range(n):
            u = i
            v = (i + k) % n
            ek = _edge_key(u, v)
            if ek not in all_edges or ek in used_edges:
                valid = False
                break
        
        if valid:
            cycle = []
            v = 0
            for _ in range(n):
                cycle.append(v)
                v = (v + k) % n
            cycles.append(cycle)
    
    return cycles

def _find_cycle_of_length(avail_adj, n, start, target_len, avail_edges, deadline):
    """Find a simple cycle of exactly target_len starting from start."""
    
    def dfs(current, path, visited):
        if time.time() > deadline:
            return None
        
        if len(path) == target_len:
            # Check if we can close back to start
            if start in avail_adj[current]:
                ek = _edge_key(current, start)
                if ek in avail_edges:
                    return path[:]
            return None
        
        # Pruning: if remaining vertices needed > available neighbors, fail
        remaining = target_len - len(path)
        if remaining < 2:
            return None
        
        for neighbor in sorted(avail_adj[current]):
            if neighbor == start:
                continue
            if neighbor not in visited:
                ek = _edge_key(current, neighbor)
                if ek in avail_edges:
                    visited.add(neighbor)
                    path.append(neighbor)
                    result = dfs(neighbor, path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(neighbor)
        return None
    
    visited = {start}
    path = [start]
    result = dfs(start, path, visited)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = _build_adj()
    n = P['n']
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    used_edges = set()
    avail_edges = set(all_edges)
    
    # Build initial avail_adj
    avail_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        avail_adj[u].add(v)
        avail_adj[v].add(u)
    
    cycles = []
    
    # Strategy: Greedily find cycles, prioritizing shorter lengths.
    # This maximizes the number of cycles.
    # We try lengths 3, 4, ..., 21.
    # For each length, we try to find as many disjoint cycles as possible.
    
    # To improve greedy performance, we can try multiple starting vertices and pick the "best" cycle
    # (e.g., one that leaves the most edges for others, or just the first one found).
    # Given the small size, we'll use a simple greedy approach:
    # 1. Iterate lengths L from 3 to 21.
    # 2. For each L, repeatedly find a cycle of length L using available edges.
    # 3. Remove its edges.
    # 4. Stop if we have 10 cycles or no more cycles of length L.
    # 5. Move to next L.
    
    # However, greedy can get stuck. A better heuristic is to try to find ALL cycles of length 3 first,
    # then 4, etc.
    
    # Let's refine:
    # We will try to find cycles of length L.
    # We will use a backtracking search that tries to find a cycle.
    # To avoid getting stuck in a bad greedy choice, we could use a limited backtracking
    # that tries a few alternatives if the first choice leads to a dead end later?
    # That's complex. Let's stick to robust greedy first.
    
    # Optimization: Pre-compute all triangles (3-cycles) and 4-cycles?
    # 21 vertices, ~1000 edges. Triangles are fast to find.
    
    for target_len in range(3, n + 1):
        if len(cycles) >= 10:
            break
        if time.time() > deadline - 5:
            break
            
        # Try to find cycles of this length
        found_any = True
        while found_any and len(cycles) < 10:
            found_any = False
            for start in range(n):
                if time.time() > deadline - 5:
                    break
                if len(cycles) >= 10:
                    break
                    
                # Check if start has enough degree to form a cycle of this length
                if len(avail_adj[start]) < 2:
                    continue
                
                # Try to find a cycle
                # To make it deterministic and fast, we just call _find_cycle_of_length
                # But _find_cycle_of_length uses DFS which might be slow if it explores many paths.
                # For small L, it's fast.
                
                cycle = _find_cycle_of_length(avail_adj, n, start, target_len, avail_edges, deadline)
                
                if cycle is not None:
                    # Verify and add
                    valid = True
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        ek = _edge_key(u, v)
                        if ek not in avail_edges:
                            valid = False
                            break
                    
                    if valid:
                        cycles.append(cycle)
                        # Remove edges
                        for j in range(len(cycle)):
                            u = cycle[j]
                            v = cycle[(j + 1) % len(cycle)]
                            ek = _edge_key(u, v)
                            avail_edges.discard(ek)
                            avail_adj[u].discard(v)
                            avail_adj[v].discard(u)
                        found_any = True
                        # Break out of start loop to restart with new graph state
                        break
                        
            # If we found a cycle, we restart the start loop for the same target_len
            # This is handled by the while loop
    
    # If we still don't have 10, we might have missed some due to greedy choices.
    # We can try a second pass or a different order.
    # But for now, this should yield many small cycles.
    
    # Ensure at least 1 cycle (fallback to Hamiltonian if greedy failed completely)
    if not cycles and time.time() < deadline:
        # Try circulant
        circulant_cycles = _find_circulant_cycles(n, all_edges, used_edges, deadline)
        for cycle in circulant_cycles:
            if len(cycles) >= 10:
                break
            valid = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                ek = _edge_key(u, v)
                if ek not in avail_edges:
                    valid = False
                    break
            if valid:
                cycles.append(cycle)
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    ek = _edge_key(u, v)
                    avail_edges.discard(ek)
                    avail_adj[u].discard(v)
                    avail_adj[v].discard(u)
    
    # Cap at 10
    cycles = cycles[:10]
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
