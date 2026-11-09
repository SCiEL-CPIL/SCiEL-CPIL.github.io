"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_START = None

def _get_cycle_lengths(graph, n):
    """
    Find all simple cycle lengths in the graph.
    graph: list of sets of neighbors
    n: number of vertices
    """
    cycle_lengths = set()
    # Use DFS to find all simple cycles
    # Since n=12, this is feasible.
    # We can use a standard algorithm to enumerate simple cycles.
    
    def dfs(u, visited, path):
        if len(path) > 1:
            # Check if we can close the cycle
            if path[0] in graph[u]:
                cycle_lengths.add(len(path))
        
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, visited, path)
                path.pop()
                visited.remove(v)
    
    # To avoid duplicates and ensure we find all, we can fix a start node
    # and only allow cycles where the start node is the smallest node in the cycle.
    for start in range(n):
        visited = {start}
        dfs(start, visited, [start])
        
    return cycle_lengths

def construct():
    global _START
    if _START is None:
        _START = time.time()
    DEADLINE = _START + 85

    n = 12
    best_edges = None
    best_score = float('inf')
    
    # Fallback: Two K3,3
    fallback_edges = []
    for offset in (0, 6):
        A = [offset, offset + 1, offset + 2]
        B = [offset + 3, offset + 4, offset + 5]
        for a in A:
            for b in B:
                fallback_edges.append((a, b))
    
    # Try to find a better graph via random 3-regular graphs
    # We generate random 3-regular graphs and check cycle lengths.
    
    def generate_random_3_regular(n):
        # Configuration model
        # Each vertex has 3 stubs.
        # Pair stubs randomly.
        # Avoid self-loops and multi-edges.
        for _ in range(1000): # Attempts
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            
            if valid:
                # Check if it's a single component or disconnected?
                # Disconnected is allowed.
                # Check degrees
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == 3 for d in deg):
                    return list(edges)
        return None

    def evaluate_graph(edges):
        if not edges:
            return float('inf')
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check min degree
        if any(len(a) < 3 for a in adj):
            return float('inf')
            
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                if v in adj[u]: # self loop
                    return float('inf')
                for w in adj[v]:
                    if w == u:
                        continue
                    if w in adj[u]:
                        return float('inf') # Triangle u-v-w-u
        
        # Get cycle lengths
        lengths = _get_cycle_lengths(adj, n)
        if not lengths:
            return float('inf') # Must contain a cycle
            
        return len(lengths)

    # Run random search
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < 50 and attempts < 5000:
        edges = generate_random_3_regular(n)
        if edges:
            score = evaluate_graph(edges)
            if score < best_score:
                best_score = score
                best_edges = edges
                if best_score == 1:
                    break
        attempts += 1

    # If we found a better graph, use it. Else fallback.
    if best_edges and best_score < 2:
        final_edges = best_edges
    else:
        final_edges = fallback_edges

    # Normalize and return
    seen = set()
    out = []
    for u, v in final_edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([int(u), int(v)])
    out.sort()
    
    return out

# EVOLVE-BLOCK-END
