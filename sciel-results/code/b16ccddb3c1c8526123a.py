import time
import sys

_START = None

def _get_distinct_cycle_lengths(edges):
    """
    Computes the set of distinct lengths of all simple cycles in the graph.
    """
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited):
        if len(current) > 12:
            return
        
        for neighbor in adj[current[-1]]:
            if neighbor == start and len(current) >= 4:
                length = len(current)
                cycle_lengths.add(length)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, current + [neighbor], visited)
                visited.remove(neighbor)

    for v in range(n):
        dfs(v, [v], {v})

    return cycle_lengths

def construct():
    # Two disjoint K_{3,3} components.
    # Each K_{3,3} has 6 vertices, min degree 3, triangle-free.
    # Cycle lengths in K_{3,3}: 4 and 6.
    # Total distinct cycle lengths: {4, 6}. Count = 2.
    # This is the theoretical minimum for min-degree 3 triangle-free graphs.
    
    def make_k33(offset):
        # Bipartition: A = {offset, offset+1, offset+2}, B = {offset+3, offset+4, offset+5}
        edges = []
        for a in range(3):
            for b in range(3):
                u = offset + a
                v = offset + 3 + b
                edges.append([int(u), int(v)])
        return edges

    raw_edges = make_k33(0) + make_k33(6)
    
    # Deduplicate and sort for consistency
    seen = set()
    result = []
    for u, v in raw_edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
            
    result.sort(key=lambda x: (x[0], x[1]))
    return result
