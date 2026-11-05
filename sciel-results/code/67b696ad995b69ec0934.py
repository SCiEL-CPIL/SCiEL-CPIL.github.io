import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    A = list(range(10))
    B = list(range(10, 20))
    
    best_edges = []
    best_score = 0
    
    # Fast check: given a bipartite graph with colors, count non-rainbow 4-cycles
    def count_bad_cycles(adj_colors):
        """adj_colors[u][v] = color or None if no edge. u in A, v in B."""
        bad = 0
        for i, u1 in enumerate(A):
            for u2 in A[i+1:]:
                # Common neighbors in B
                common = []
                for v in B:
                    if adj_colors[u1][v] is not None and adj_colors[u2][v] is not None:
                        common.append(v)
                for j, v1 in enumerate(common):
                    for v2 in common[j+1:]:
                        c1 = adj_colors[u1][v1]
                        c2 = adj_colors[u2][v1]
                        c3 = adj_colors[u2][v2]
                        c4 = adj_colors[u1][v2]
                        if len({c1, c2, c3, c4}) < 4:
                            bad += 1
        return bad
    
    # Try random coloring strategies with greedy edge addition
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random color assignment function
        perm_a = list(range(10))
        perm_b = list(range(10))
        random.shuffle(perm_a)
        random.shuffle(perm_b)
        
        def color_func(a, b):
            return (perm_a[a] + perm_b[b-10] + attempt * 7) % 20
        
        # Build adjacency with colors
        adj_colors = {u: {v: None for v in B} for u in A}
        
        # Greedy: add edges in random order, keep if doesn't create bad cycle
        edge_order = []
        for u in A:
            for v in B:
                edge_order.append((u, v))
        random.shuffle(edge_order)
        
        edge_count = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            c = color_func(u, v)
            
            # Check if adding (u,v) with color c creates a bad 4-cycle
            # A bad cycle would be u-v-u2-v2-u or u-v2-u2-v-u
            # For each u2 in A, v2 in B: cycle u-v-u2-v2-u
            # We need: adj_colors[u2][v] is not None and adj_colors[u2][v2] is not None
            # and {c, adj_colors[u2][v], adj_colors[u2][v2], adj_colors[u][v2]} has < 4 distinct
            
            creates_bad = False
            for u2 in A:
                if u2 == u:
                    continue
                if adj_colors[u2][v] is None:
                    continue
                c2 = adj_colors[u2][v]
                for v2 in B:
                    if v2 == v:
                        continue
                    if adj_colors[u][v2] is None or adj_colors[u2][v2] is None:
                        continue
                    c3 = adj_colors[u2][v2]
                    c4 = adj_colors[u][v2]
                    if len({c, c2, c3, c4}) < 4:
                        creates_bad = True
                        break
                if creates_bad:
                    break
            
            if not creates_bad:
                adj_colors[u][v] = c
                edge_count += 1
        
        if edge_count > best_score:
            best_score = edge_count
            best_edges = []
            for u in A:
                for v in B:
                    if adj_colors[u][v] is not None:
                        best_edges.append([u, v, adj_colors[u][v]])
    
    # Local search: try to add back removed edges or swap colors
    if best_edges and time.time() < deadline - 5:
        # Build current adjacency
        adj_colors = {u: {v: None for v in B} for u in A}
        for e in best_edges:
            u, v, c = e
            adj_colors[u][v] = c
        
        # Try adding edges that were removed
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for u in A:
                for v in B:
                    if adj_colors[u][v] is not None:
                        continue
                    
                    # Try each color
                    for c in range(20):
                        creates_bad = False
                        for u2 in A:
                            if u2 == u:
                                continue
                            if adj_colors[u2][v] is None:
                                continue
                            c2 = adj_colors[u2][v]
                            for v2 in B:
                                if v2 == v:
                                    continue
                                if adj_colors[u][v2] is None or adj_colors[u2][v2] is None:
                                    continue
                                c3 = adj_colors[u2][v2]
                                c4 = adj_colors[u][v2]
                                if len({c, c2, c3, c4}) < 4:
                                    creates_bad = True
                                    break
                            if creates_bad:
                                break
                        
                        if not creates_bad:
                            adj_colors[u][v] = c
                            best_edges.append([u, v, c])
                            added = True
                            break
    
    # Final fallback if nothing found
    if not best_edges:
        best_edges = [[0, 10, 0]]
    
    return best_edges
