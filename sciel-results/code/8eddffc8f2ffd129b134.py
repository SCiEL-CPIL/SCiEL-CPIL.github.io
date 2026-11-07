"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = time.time() + 80.0  # safety margin before 90s kill

def build_adjacency(edges):
    n = P['n']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamiltonian_cycle(adj, n, start=0, deadline=None):
    """Find a Hamiltonian cycle in the graph defined by adj using backtracking."""
    if deadline is not None and time.time() > deadline:
        return None
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    sys.setrecursionlimit(10000)
    
    def dfs(current, path_len):
        if deadline is not None and time.time() > deadline:
            return None
        if path_len == n:
            if start in adj[current]:
                return list(path)
            return None
        for nxt in sorted(adj[current]):
            if not visited[nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(nxt, path_len + 1)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None
    
    return dfs(start, 1)

def check_hamiltonian_necessary_conditions(adj, n):
    """Check necessary conditions for Hamiltonian cycle existence."""
    for v in range(n):
        if len(adj[v]) < 2:
            return False
    # Check connectivity
    visited = [False] * n
    visited[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    if not all(visited):
        return False
    return True

def find_hamiltonian_cycle_with_pruning(adj, n, start=0, deadline=None):
    """Find a Hamiltonian cycle with degree-based pruning and connectivity checks."""
    if deadline is not None and time.time() > deadline:
        return None
    
    if not check_hamiltonian_necessary_conditions(adj, n):
        return None
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    sys.setrecursionlimit(10000)
    
    # Pre-sort neighbors: prefer lower-degree vertices first (more constrained)
    neighbor_order = {}
    for v in range(n):
        neighbor_order[v] = sorted(adj[v], key=lambda x: len(adj[x]))
    
    def dfs(current, path_len):
        if deadline is not None and time.time() > deadline:
            return None
        if path_len == n:
            if start in adj[current]:
                return list(path)
            return None
        
        # Pruning: for unvisited vertices, check if they still have enough
        # potential neighbors to be in a Hamiltonian cycle
        for v in range(n):
            if not visited[v] and v != start:
                count = 0
                for nb in adj[v]:
                    if not visited[nb]:
                        count += 1
                    elif nb == current:
                        count += 1
                    elif nb == start:
                        count += 1
                if count < 2:
                    return None
        
        for nxt in neighbor_order[current]:
            if not visited[nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(nxt, path_len + 1)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None
    
    return dfs(start, 1)

def construct():
    global DEADLINE
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency
    adj = build_adjacency(edges)
    
    cycles = []
    used_edges = set()
    
    max_cycles_to_try = 4  # Try to find up to 4 cycles
    
    for cycle_idx in range(max_cycles_to_try):
        if DEADLINE is not None and time.time() > DEADLINE:
            break
        
        # Build subgraph of unused edges
        remaining_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in used_edges:
                remaining_edges.append(e)
        
        # Check if remaining graph can support a Hamiltonian cycle
        rem_adj = build_adjacency(remaining_edges)
        if not check_hamiltonian_necessary_conditions(rem_adj, n):
            break
        
        # Try multiple starting vertices for better success rate
        deadline = DEADLINE - 2.0 if DEADLINE else None
        cycle = None
        
        for start_v in [0, 1, 2, 3, 4]:
            if deadline is not None and time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle_with_pruning(rem_adj, n, start=start_v, deadline=deadline)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = (min(u, v), max(u, v))
            used_edges.add(key)
        
        cycles.append(cycle)
    
    # Ensure we have at least 1 cycle
    if not cycles:
        # Fallback: return the trivial cycle
        return [list(range(n))]
    
    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
