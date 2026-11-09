"""Improved program: exploit circulant structure to find all 3 edge-disjoint Hamiltonian cycles."""
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
    # For n=21, coprime steps are: 1,2,4,5,8,10,11,13,16,17,19,20
    for k in range(1, n):
        if time.time() > deadline:
            break
        if _gcd(k, n) != 1:
            continue
        
        # Check if all edges in this circulant cycle are in the graph and unused
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

def _find_hamiltonian_cycle(adj, n, used_edges, deadline, start_vertex=0):
    """Find a Hamiltonian cycle using backtracking with aggressive pruning."""
    
    # Precompute available neighbors (excluding used edges)
    avail = [set() for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if _edge_key(v, nb) not in used_edges:
                avail[v].add(nb)
    
    # Sort neighbors by degree in available graph for better pruning
    deg = [len(avail[v]) for v in range(n)]
    neighbor_order = {v: sorted(avail[v], key=lambda x: (deg[x], x)) for v in range(n)}
    
    # Precompute forced moves: vertices with only 1 available neighbor
    def _get_forced(avail_sets, visited_set, current):
        """Find unvisited vertices with exactly 1 available unvisited/current neighbor."""
        forced = {}
        for v in range(n):
            if v in visited_set:
                continue
            candidates = [nb for nb in avail_sets[v] if nb == current or nb not in visited_set]
            if len(candidates) == 1:
                forced[v] = candidates[0]
        return forced
    
    def _check_connectivity(avail_sets, visited_set, current):
        """Check that all unvisited vertices + current form a connected component."""
        unvisited = [v for v in range(n) if v not in visited_set]
        if not unvisited:
            return True
        
        # BFS from current through unvisited vertices
        visited_bfs = set()
        queue = deque([current])
        visited_bfs.add(current)
        
        while queue:
            node = queue.popleft()
            for nb in avail_sets[node]:
                if nb not in visited_bfs and (nb not in visited_set or nb == current):
                    if nb not in visited_set or nb == current:
                        visited_bfs.add(nb)
                        queue.append(nb)
        
        # All unvisited vertices must be reachable
        for v in unvisited:
            if v not in visited_bfs:
                return False
        return True
    
    def backtrack(visited, path, current, depth):
        if time.time() > deadline:
            return None
        
        if depth == n:
            # Check if we can close the cycle
            if path[0] in avail[current]:
                return path[:]
            return None
        
        # Pruning: check connectivity
        if not _check_connectivity(avail, visited, current):
            return None
        
        # Pruning: forced moves
        forced = _get_forced(avail, visited, current)
        
        # If current has a forced unvisited neighbor, use it
        if current in forced:
            neighbor = forced[current]
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                result = backtrack(visited, path, neighbor, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited.discard(neighbor)
            return None
        
        # Try neighbors in order, but prioritize forced moves
        candidates = []
        for neighbor in neighbor_order[current]:
            if neighbor not in visited:
                candidates.append(neighbor)
        
        # Sort: forced targets first, then by degree
        candidates.sort(key=lambda x: (0 if x in forced else 1, deg[x], x))
        
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            result = backtrack(visited, path, neighbor, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.discard(neighbor)
        return None
    
    # Try starting from the given vertex
    visited = {start_vertex}
    path = [start_vertex]
    result = backtrack(visited, path, start_vertex, 1)
    if result is not None:
        return result
    
    # Try other starting vertices
    for start in range(n):
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        result = backtrack(visited, path, start, 1)
        if result is not None:
            return result
    
    return None

def _find_small_cycles(adj, n, used_edges, deadline, min_size=3, max_size=6, max_cycles=10):
    """Find small edge-disjoint cycles (size min_size to max_size) greedily."""
    found = []
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    # Build available adjacency
    avail = [set() for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if _edge_key(v, nb) not in used_edges:
                avail[v].add(nb)
    
    # Find triangles first (most efficient packing)
    for size in range(min_size, max_size + 1):
        if time.time() > deadline:
            break
        if len(found) >= max_cycles:
            break
        
        # For small sizes, use DFS with pruning
        visited_in_search = set()  # (vertex, depth) pairs to avoid redundant exploration
        
        for start in range(n):
            if time.time() > deadline:
                break
            if len(found) >= max_cycles:
                break
            
            # DFS to find cycle of given size starting from 'start'
            def dfs(path, current, depth):
                if time.time() > deadline:
                    return None
                if len(found) >= max_cycles:
                    return None
                
                if depth == size:
                    # Check if we can close the cycle
                    if start in avail[current]:
                        edge = _edge_key(current, start)
                        if edge not in used_edges:
                            return path[:]
                    return None
                
                # Pruning: remaining vertices must be reachable
                remaining_needed = size - depth
                if remaining_needed < 1:
                    return None
                
                for nb in sorted(avail[current]):
                    if nb in path:
                        continue
                    if nb == start and depth < size:
                        continue
                    path.append(nb)
                    result = dfs(path, nb, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                return None
            
            path = [start]
            cycle = dfs(path, start, 1)
            if cycle is not None:
                # Mark edges as used
                valid = True
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    edge = _edge_key(u, v)
                    if edge in used_edges:
                        valid = False
                        break
                    used_edges.add(edge)
                
                if valid:
                    found.append(cycle)
                
                # Update available adjacency
                for v in range(n):
                    for nb in list(avail[v]):
                        if _edge_key(v, nb) in used_edges:
                            avail[v].discard(nb)
    
    return found

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_edge_key(u, v))
    
    used_edges = set()
    cycles = []
    
    # Step 1: Find all circulant cycles (fast, deterministic)
    circulant_cycles = _find_circulant_cycles(n, all_edges, used_edges, deadline)
    
    # Sort by step size for deterministic order
    circulant_cycles.sort(key=lambda c: c[1] - c[0] if c[1] > c[0] else c[1] - c[0] + n)
    
    for cycle in circulant_cycles:
        if time.time() > deadline - 10:
            break
        # Verify and add
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = _edge_key(u, v)
            if edge in used_edges:
                valid = False
                break
            used_edges.add(edge)
        if valid:
            cycles.append(cycle)
    
    # Calculate theoretical upper bound on number of edge-disjoint Hamiltonian cycles
    # Each cycle uses n edges. Total edges is len(all_edges).
    upper_bound_ham = len(all_edges) // n
    
    # Step 2: Try to find additional Hamiltonian cycles if we haven't hit the bound
    remaining_ham = min(10 - len(cycles), upper_bound_ham - len(cycles))
    if remaining_ham > 0:
        for i in range(remaining_ham):
            if time.time() > deadline - 15:
                break
            
            cycle = None
            # Try multiple starting vertices with time-sliced search
            time_per_start = max(0.5, (deadline - 15 - time.time()) / n)
            for start in range(n):
                if time.time() > deadline - 15:
                    break
                sub_deadline = min(deadline - 15, time.time() + time_per_start)
                cycle = _find_hamiltonian_cycle(adj, n, used_edges, sub_deadline, start)
                if cycle is not None:
                    break
            
            if cycle is None:
                break
            
            # Mark edges as used
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                used_edges.add(edge)
            
            cycles.append(cycle)
    
    # Step 3: If we have room, find small cycles (triangles, 4-cycles, etc.)
    # This can pack more total cycles than Hamiltonian cycles alone
    if len(cycles) < 10 and time.time() < deadline - 5:
        small_deadline = deadline - 2
        small_cycles = _find_small_cycles(adj, n, used_edges, small_deadline, min_size=3, max_size=6, max_cycles=10 - len(cycles))
        cycles.extend(small_cycles)
    
    # Ensure we have at least one cycle (fallback)
    if not cycles and time.time() < deadline:
        cycle = _find_hamiltonian_cycle(adj, n, used_edges, deadline, 0)
        if cycle is not None:
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                edge = _edge_key(u, v)
                used_edges.add(edge)
            cycles.append(cycle)
    
    # Convert to lists of Python ints
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
