"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, deadline):
    """Find a Hamiltonian cycle in the graph using backtracking with Warnsdorff heuristic."""
    n = len(adj)
    if n < 3:
        return None
    
    # Try multiple starting vertices
    for start in range(n):
        # Check if start can be part of a cycle (degree >= 2)
        if len(adj[start]) < 2:
            continue
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        
        def dfs():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                if path[-1] in adj[start]:
                    return list(path)
                return None
            
            # Get unvisited neighbors
            curr = path[-1]
            neighbors = [v for v in adj[curr] if not visited[v]]
            
            # Warnsdorff heuristic: prefer vertices with fewer unvisited neighbors
            def warnsdorff_score(v):
                return len([u for u in adj[v] if not visited[u]])
            
            neighbors.sort(key=warnsdorff_score)
            
            for v in neighbors:
                visited[v] = True
                path.append(v)
                
                result = dfs()
                if result is not None:
                    return result
                
                path.pop()
                visited[v] = False
            
            return None
        
        result = dfs()
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    adj = _build_adj()
    n = P['n']
    
    # First, find one Hamiltonian cycle
    cycle1 = _find_ham_cycle(adj, deadline)
    if cycle1 is None:
        # Fallback: return a single cycle if we can find any
        # This shouldn't happen for this graph, but just in case
        return [list(range(n))]
    
    # Remove edges used by cycle1
    used_edges = set()
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        edge = (min(u, v), max(u, v))
        used_edges.add(edge)
    
    # Build remaining graph
    remaining_edges = []
    for u, v in P['edges']:
        edge = (min(u, v), max(u, v))
        if edge not in used_edges:
            remaining_edges.append((u, v))
    
    # Build adjacency for remaining graph
    rem_adj = [set() for _ in range(n)]
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    # Try to find a second Hamiltonian cycle in the remaining graph
    cycle2 = None
    for start in range(n):
        if len(rem_adj[start]) < 2:
            continue
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        
        def dfs2():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                if path[-1] in rem_adj[start]:
                    return list(path)
                return None
            
            curr = path[-1]
            neighbors = [v for v in rem_adj[curr] if not visited[v]]
            
            def score(v):
                return len([u for u in rem_adj[v] if not visited[u]])
            
            neighbors.sort(key=score)
            
            for v in neighbors:
                visited[v] = True
                path.append(v)
                
                result = dfs2()
                if result is not None:
                    return result
                
                path.pop()
                visited[v] = False
            
            return None
        
        result = dfs2()
        if result is not None:
            cycle2 = result
            break
    
    cycles = [cycle1]
    if cycle2 is not None:
        cycles.append(cycle2)
    
    # Convert to proper list format
    result = []
    for c in cycles:
        result.append(list(c))
    
    return result
# EVOLVE-BLOCK-END
