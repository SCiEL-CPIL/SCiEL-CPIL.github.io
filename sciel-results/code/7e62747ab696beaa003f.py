"""Strong edge colouring via Backtracking with Branch and Bound."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import defaultdict
    import time

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try many times to build a bounded-degree graph
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, n, max_deg):
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict sets
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # conflict[i] is a list of indices j such that edge i and edge j conflict
        conflict = [[] for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                if x == u or x == v or y == u or y == v:
                    conflict[i].append(j)
                    conflict[j].append(i)
                else:
                    if x in adj[u] or x in adj[v] or y in adj[u] or y in adj[v]:
                        conflict[i].append(j)
                        conflict[j].append(i)
        
        # Get initial upper bound using greedy
        degs = [len(conflict[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: -degs[i])
        
        color_greedy = [-1] * m
        num_colors_greedy = 0
        for i in order:
            used = set()
            for j in conflict[i]:
                if color_greedy[j] != -1:
                    used.add(color_greedy[j])
            c = 0
            while c in used:
                c += 1
            color_greedy[i] = c
            num_colors_greedy = max(num_colors_greedy, c + 1)
            
        best_color = color_greedy[:]
        best_k = num_colors_greedy
        
        # Backtracking to find minimum k
        # We try k from best_k - 1 down to 1
        
        import sys
        sys.setrecursionlimit(10000)
        
        node_count = 0
        NODE_LIMIT = 200000 # Limit per k attempt
        
        def backtrack(k, color, assigned_count):
            nonlocal node_count
            node_count += 1
            if node_count > NODE_LIMIT:
                return False
            
            if assigned_count == m:
                return True
            
            # Select next unassigned vertex from static order
            for i in order:
                if color[i] == -1:
                    # Check available colors
                    used = set()
                    for j in conflict[i]:
                        if color[j] != -1:
                            used.add(color[j])
                            
                    # Try colors
                    for c in range(k):
                        if c not in used:
                            color[i] = c
                            if backtrack(k, color, assigned_count + 1):
                                return True
                            color[i] = -1
                    return False
            return False
        
        # Try to improve on greedy
        for k in range(best_k - 1, 0, -1):
            node_count = 0
            color = [-1] * m
            if backtrack(k, color, 0):
                # Found a solution with k colors
                best_color = color[:]
                best_k = k
                # Continue to see if we can go lower
            else:
                # No solution with k, so best_k is the current best
                break
                
        # Return the best coloring found
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        return [best_color[edge_to_idx[e]] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        if not edges:
            out.append([])
            continue
        colors = strong_edge_coloring(edges, n, d)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
