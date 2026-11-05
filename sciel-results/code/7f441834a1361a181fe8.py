"""
Construct a maximum-cardinality subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.

Approach:
1. Build a conflict graph where an edge exists between i and j if lcm(i, j) > 512.
2. Find a maximum independent set (MIS) in this graph using a branch-and-bound
   solver (Tomita-style) with greedy coloring for upper bounds.
3. The graph has 512 nodes, so the solver is feasible within 60 seconds.
"""

from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _greedy_color(order, adj):
    """
    Greedy coloring of vertices in the given order.
    Returns a list of colors (one per vertex in original order).
    Colors are 1-indexed; color 0 means uncolored.
    """
    n = len(order)
    color = [0] * n
    # Map vertex index in original graph to position in order
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    # For each vertex in order, assign the smallest available color
    for i, v in enumerate(order):
        # Collect colors of already-colored neighbors
        used = set()
        for u in adj[v]:
            if pos_in_order[u] < i:  # already colored
                used.add(color[pos_in_order[u]])
        c = 1
        while c in used:
            c += 1
        color[pos_in_order[v]] = c
    
    # Convert back to original vertex indexing
    result = [0] * n
    for v in range(n):
        result[v] = color[pos_in_order[v]]
    return result

def _max_independent_set(adj):
    """
    Tomita-style branch-and-bound for Maximum Independent Set.
    adj[v] is the set of neighbors of v in the conflict graph.
    Returns the maximum independent set as a set of vertex indices.
    """
    n = len(adj)
    
    # Convert adj to sorted lists for speed
    adj_list = [sorted(adj[v]) for v in range(n)]
    
    best_set = set()
    best_size = 0
    
    # Order vertices by degree (ascending) for initial ordering
    order = sorted(range(n), key=lambda v: len(adj_list[v]))
    
    # We work on the subgraph induced by P (candidate vertices)
    # Using a recursive approach with a list P of candidate vertices
    
    def dfs(P):
        nonlocal best_set, best_size
        
        if not P:
            if len(best_set) < best_size:
                pass  # nothing to do
            return
        if len(P) + best_size <= best_size:
            return
        
        # Greedy color the vertices in P
        # We need to color the subgraph induced by P
        p_set = set(P)
        # Build adjacency restricted to P
        p_adj = {v: [u for u in adj_list[v] if u in p_set] for v in P}
        
        # Order P by degree in the induced subgraph
        p_order = sorted(P, key=lambda v: len(p_adj[v]))
        
        # Greedy coloring in this order
        pos_map = {v: i for i, v in enumerate(p_order)}
        colors = [0] * len(p_order)
        for i, v in enumerate(p_order):
            used = set()
            for u in p_adj[v]:
                if pos_map[u] < i:
                    used.add(colors[pos_map[u]])
            c = 1
            while c in used:
                c += 1
            colors[pos_map[v]] = c
        
        # Group vertices by color
        max_color = max(colors) if colors else 0
        color_classes = [[] for _ in range(max_color + 1)]
        for i, v in enumerate(p_order):
            color_classes[colors[i]].append(v)
        
        # Upper bound: best_size + max_color
        if best_size + max_color <= best_size:
            return
        
        # Branch: for each color class from highest to lowest
        for c in range(max_color, 0, -1):
            if best_size + c <= best_size:
                break
            for v in color_classes[c]:
                # Include v in the independent set
                new_P = [u for u in P if u != v and u not in p_adj[v]]
                # Upper bound check
                if len(new_P) + best_size + 1 <= best_size:
                    continue
                dfs(new_P)
                # Remove v from P for next iteration
                P = [u for u in P if u != v]
                if len(P) + best_size <= best_size:
                    break
            if len(P) + best_size <= best_size:
                break
    
    # Initialize with a greedy solution
    greedy_set = set()
    remaining = list(range(n))
    while remaining:
        # Pick vertex with minimum degree in remaining
        v = min(remaining, key=lambda x: sum(1 for u in remaining if u in adj[x]))
        greedy_set.add(v)
        remaining = [u for u in remaining if u != v and u not in adj[v]]
    
    best_set = set(greedy_set)
    best_size = len(best_set)
    
    # Run branch and bound
    dfs(list(range(n)))
    
    return best_set

def construct():
    N = 512
    limit = N
    
    # Build conflict graph: edge (i, j) if lcm(i, j) > limit
    # Nodes are 1..N
    # We'll index by 0..N-1 corresponding to values 1..N
    n = N
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if _lcm(a, b) > limit:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find maximum independent set
    mis = _max_independent_set(adj)
    
    # Convert back to values
    result = [v + 1 for v in mis]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
