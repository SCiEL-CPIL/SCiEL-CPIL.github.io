"""Optimized greedy for rainbow 4-cycle edge coloring on 20 vertices."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    best_edges = None
    best_count = 0
    
    # Precompute all candidate edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    random.seed(42)
    
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Build edge ordering: first few restarts use structured orders
        if restart < 5:
            # Structured orderings: by sum, by difference, etc.
            if restart == 0:
                candidate_edges = sorted(all_edges, key=lambda e: (e[0] + e[1], e[0]))
            elif restart == 1:
                candidate_edges = sorted(all_edges, key=lambda e: (abs(e[0] - e[1]), e[0]))
            elif restart == 2:
                candidate_edges = sorted(all_edges, key=lambda e: (e[1] - e[0], e[0]))
            elif restart == 3:
                candidate_edges = sorted(all_edges, key=lambda e: (e[0] * e[1], e[0]))
            else:
                candidate_edges = sorted(all_edges, key=lambda e: ((e[0] + e[1]) % 7, e[0]))
        else:
            candidate_edges = all_edges[:]
            random.shuffle(candidate_edges)
        
        # Greedy construction
        edge_list = []
        
        # Track adjacency and edge colors using fast integer keys
        # key = min(u,v)*n + max(u,v)
        neighbors = [set() for _ in range(n)]
        edge_color = [-1] * (n * n)  # indexed by min*n+max
        
        for (u, v) in candidate_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            # Check all length-3 paths u-w-x-v
            # w in neighbors[u], x in neighbors[v], (w,x) is an edge
            forbidden = set()
            valid = True
            
            for w in neighbors[u]:
                if w == v:
                    continue
                for x in neighbors[v]:
                    if x == u or x == w:
                        continue
                    
                    # Check if (w, x) is an edge
                    if w < x:
                        key_wx = w * n + x
                    else:
                        key_wx = x * n + w
                    
                    c_wx = edge_color[key_wx]
                    if c_wx == -1:
                        continue
                    
                    # Get colors of the other two edges
                    if u < w:
                        key_uw = u * n + w
                    else:
                        key_uw = w * n + u
                    c_uw = edge_color[key_uw]
                    
                    if x < v:
                        key_xv = x * n + v
                    else:
                        key_xv = v * n + x
                    c_xv = edge_color[key_xv]
                    
                    forbidden.add(c_uw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xv)
            
            if not valid:
                continue
            
            # Pick a color not in forbidden
            available = [c for c in colors if c not in forbidden]
            if not available:
                continue
            
            c = available[0]  # deterministic choice for consistency
            if u < v:
                key = u * n + v
            else:
                key = v * n + u
            edge_color[key] = c
            edge_list.append([u, v, c])
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        count = len(edge_list)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edge_list]
        
        # Early termination if we found a very good solution
        if best_count >= 80:
            break
    
    if best_edges is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]
    
    return best_edges
