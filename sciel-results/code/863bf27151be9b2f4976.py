import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build conflict graph: C[i][j] = True if i and j CANNOT coexist
    C = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                C[i][j] = True
                C[j][i] = True
    
    # Build adjacency lists (conflicts)
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and C[i][j]:
                adj[i].append(j)
    
    # Greedy initial solution: all odd numbers (no conflicts among them)
    S = set(range(1, n + 1, 2))
    best_size = len(S)
    best_set = set(S)
    
    # Try greedy improvement: add vertices that don't conflict
    for v in range(1, n + 1):
        if v not in S:
            if all(not C[v][u] for u in S):
                S.add(v)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Tomita-style Maximum Independent Set via branch-and-bound
    # We find maximum independent set in the complement of conflict graph
    # i.e., maximum clique in the "compatibility" graph
    
    # Compatibility: two vertices are compatible if they do NOT conflict
    # We want max set of mutually compatible vertices = max independent set in conflict graph
    
    # Use recursive branch and bound with coloring upper bound
    # State: candidates (set of vertices that can still be added), current solution size
    
    sys.setrecursionlimit(10000)
    
    # Order vertices by degree (descending) for better pruning
    # Higher degree vertices first in the ordering
    vertex_order = sorted(range(1, n + 1), key=lambda v: -len(adj[v]))
    
    # Map from original vertex to position in ordering
    pos_in_order = {v: i for i, v in enumerate(vertex_order)}
    
    # For each vertex, precompute which other vertices it conflicts with (as a set for fast lookup)
    conflict_set = [set() for _ in range(n + 1)]
    for v in range(1, n + 1):
        conflict_set[v] = set(adj[v])
    
    def color_bound(candidates):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns list of color classes (each a set of vertices with same color).
        The number of colors is an upper bound on the maximum independent set size.
        """
        if not candidates:
            return []
        
        remaining = set(candidates)
        color_classes = []
        
        while remaining:
            # Greedy: pick a vertex, assign it a color, remove it and its neighbors from remaining
            # For coloring: we want to color the graph so that adjacent vertices have different colors
            # A color class is an independent set in the conflict graph
            # So we want to partition candidates into as few independent sets as possible
            
            # Pick vertex with highest degree in remaining subgraph
            best_v = max(remaining, key=lambda v: sum(1 for u in remaining if C[v][u]))
            color_class = {best_v}
            remaining.remove(best_v)
            
            # Add other vertices that are compatible with all in color_class
            for u in list(remaining):
                if all(not C[u][w] for w in color_class):
                    color_class.add(u)
                    remaining.remove(u)
            
            color_classes.append(color_class)
        
        return color_classes
    
    def branch_and_bound(candidates, current_size, current_set):
        """
        Recursive branch and bound for maximum independent set.
        candidates: set of vertices that can still be added
        current_size: size of current independent set
        current_set: the current independent set (as a set)
        Returns (best_size, best_set)
        """
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return best_size, best_set
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return best_size, best_set
        
        # Upper bound: current_size + number of colors needed
        color_classes = color_bound(candidates)
        upper_bound = current_size + len(color_classes)
        
        if upper_bound <= best_size:
            return best_size, best_set
        
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the candidate subgraph
        v = max(candidates, key=lambda x: sum(1 for y in candidates if C[x][y]))
        
        # Branch 1: include v
        new_candidates = set()
        for u in candidates:
            if u != v and not C[v][u]:
                new_candidates.add(u)
        
        current_set.add(v)
        branch_and_bound(new_candidates, current_size + 1, current_set)
        current_set.remove(v)
        
        # Branch 2: exclude v
        candidates.discard(v)
        branch_and_bound(candidates, current_size, current_set)
        
        return best_size, best_set
    
    # Start B&B with all vertices as candidates
    # But first, let's use a smarter initial candidate set
    # Start with vertices not in our greedy solution, plus try to improve
    
    # Actually, let's do a proper B&B from scratch with a good initial bound
    initial_candidates = set(range(1, n + 1))
    initial_set = set()
    
    branch_and_bound(initial_candidates, 0, initial_set)
    
    # After B&B, also try local search polish
    # Try to add any vertex that doesn't conflict
    for v in range(1, n + 1):
        if v not in best_set:
            if all(not C[v][u] for u in best_set):
                best_set.add(v)
    
    return list(best_set)
