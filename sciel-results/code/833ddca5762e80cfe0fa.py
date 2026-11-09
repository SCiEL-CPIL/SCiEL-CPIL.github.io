import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    num_colors = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_result = []
    best_count = 0
    
    def solve(seed):
        random.seed(seed)
        adj = [[] for _ in range(n)]
        color_edges = {c: [] for c in range(num_colors)}
        edge_set = set() # Stores (min(u,v), max(u,v))
        
        edges = all_edges[:]
        random.shuffle(edges)
        
        count = 0
        for u, v in edges:
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for c in colors:
                # Check properness: no neighbor of u or v has color c
                valid = True
                for w, wc in adj[u]:
                    if wc == c:
                        valid = False
                        break
                if valid:
                    for w, wc in adj[v]:
                        if wc == c:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check 4-cycle opposite color conflict
                # We are adding (u,v) with color c.
                # If there is an edge (w,x) with color c, and (v,w) and (x,u) exist, 
                # or (v,x) and (w,u) exist, then we have a 4-cycle with two edges of color c.
                for w, x in color_edges[c]:
                    if w == u or w == v or x == u or x == v:
                        continue
                    
                    # Check if (v, w) and (x, u) are in graph
                    if (min(v, w), max(v, w)) in edge_set and (min(x, u), max(x, u)) in edge_set:
                        valid = False
                        break
                    # Check if (v, x) and (w, u) are in graph
                    if (min(v, x), max(v, x)) in edge_set and (min(w, u), max(w, u)) in edge_set:
                        valid = False
                        break
                
                if valid:
                    # Add edge
                    adj[u].append((v, c))
                    adj[v].append((u, c))
                    color_edges[c].append((u, v))
                    edge_set.add((min(u, v), max(u, v)))
                    count += 1
                    break
        
        # Reconstruct result
        result = []
        for c in range(num_colors):
            for u, v in color_edges[c]:
                result.append([u, v, c])
        return result, count

    # Run multiple seeds
    # Estimate time per solve: ~1ms? 
    # 1000 solves in 1s? 
    # Let's run as many as possible.
    seed = 0
    while time.time() < deadline:
        res, cnt = solve(seed)
        if cnt > best_count:
            best_count = cnt
            best_result = res
        seed += 1
    
    return best_result
