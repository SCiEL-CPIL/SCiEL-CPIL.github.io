import time
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return list(edges)

def find_cycle(adj, edges_left):
    """
    Find a cycle in the graph defined by adj and edges_left.
    Returns a list of vertices forming the cycle, or None.
    adj: dict of vertex -> list of neighbors (static structure, but we check edges_left)
    edges_left: set of (u, v) tuples that are still available.
    """
    n = len(adj)
    visited = [False] * n
    parent = [-1] * n
    in_stack = [False] * n
    
    # We need to find a cycle.
    # We can iterate through all vertices to ensure we find a cycle in any connected component.
    for start in range(n):
        if visited[start]:
            continue
        # DFS
        stack = [(start, -1, iter(adj[start]))]
        visited[start] = True
        in_stack[start] = True
        path = [start]
        
        while stack:
            node, par, it = stack[-1]
            advanced = False
            for neighbor in it:
                # Check if edge (node, neighbor) is in edges_left
                u, v = sorted((node, neighbor))
                if (u, v) not in edges_left:
                    continue
                
                if not visited[neighbor]:
                    visited[neighbor] = True
                    in_stack[neighbor] = True
                    parent[neighbor] = node
                    path.append(neighbor)
                    stack.append((neighbor, node, iter(adj[neighbor])))
                    advanced = True
                    break
                elif in_stack[neighbor]:
                    # Found a cycle
                    # The cycle is from neighbor to node in path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    return cycle
            
            if not advanced:
                stack.pop()
                in_stack[node] = False
                path.pop()
                
    return None

def decompose(edges):
    if not edges:
        return []
    
    # Build adjacency list
    n = max(max(u, v) for u, v in edges) + 1
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    edges_left = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Deadline guard
    start_time = time.time()
    time_limit = 100  # Leave margin for 3 graphs and overhead
    
    while edges_left:
        if time.time() - start_time > time_limit:
            # Fallback: just add remaining edges as singles
            for e in edges_left:
                pieces.append(list(e))
            break
            
        cycle = find_cycle(adj, edges_left)
        
        if cycle:
            pieces.append(cycle)
            # Remove edges of the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                edges_left.discard(edge)
        else:
            # No cycles left, all remaining edges are in a forest
            for e in edges_left:
                pieces.append(list(e))
            break
            
    return pieces

def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
