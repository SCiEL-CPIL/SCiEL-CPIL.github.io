import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute edge endpoints
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    # Map edge (u,v) to index
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        
    # Precompute for each edge, the list of other 3 edges in each 4-cycle containing it
    # cycle_info[i] = list of tuples (e2, e3, e4)
    cycle_info = [[] for _ in range(NUM_EDGES)]
    
    # For each edge (u, v), find all pairs (w, x) with w, x not in {u, v}
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for w_idx in range(len(others)):
            for x_idx in range(w_idx + 1, len(others)):
                w = others[w_idx]
                x = others[x_idx]
                # Cycle 1: u-v-w-x-u -> Edges (u,v), (v,w), (w,x), (x,u)
                e1 = i
                e2 = edge_idx[(v, w)]
                e3 = edge_idx[(w, x)]
                e4 = edge_idx[(x, u)]
                cycle_info[i].append((e2, e3, e4))
                
                # Cycle 2: u-v-x-w-u -> Edges (u,v), (v,x), (x,w), (w,u)
                e1 = i
                e2 = edge_idx[(v, x)]
                e3 = edge_idx[(x, w)]
                e4 = edge_idx[(w, u)]
                cycle_info[i].append((e2, e3, e4))
                
    best_score = 0
    best_colors = None
    
    # Run multiple restarts
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        colors = [-1] * NUM_EDGES
        used_colors = set()
        next_color = 0
        
        # Randomize edge order for diversity
        order = list(range(NUM_EDGES))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline:
                break
                
            # Try to assign next_color
            valid_new = True
            blocking_colors = set()
            
            for e2, e3, e4 in cycle_info[i]:
                if colors[e2] != -1 and colors[e3] != -1 and colors[e4] != -1:
                    c2, c3, c4 = colors[e2], colors[e3], colors[e4]
                    if c2 != c3 and c3 != c4 and c2 != c4:
                        # If we use next_color, and next_color not in {c2,c3,c4}, it's a rainbow
                        if next_color not in (c2, c3, c4):
                            valid_new = False
                            blocking_colors.add(c2)
                            blocking_colors.add(c3)
                            blocking_colors.add(c4)
                            
            if valid_new:
                colors[i] = next_color
                used_colors.add(next_color)
                next_color += 1
            else:
                # Must pick from blocking_colors or other valid colors
                # Try blocking colors first
                found = False
                for c in blocking_colors:
                    # Check if c is valid
                    is_valid = True
                    for e2, e3, e4 in cycle_info[i]:
                        if colors[e2] != -1 and colors[e3] != -1 and colors[e4] != -1:
                            c2, c3, c4 = colors[e2], colors[e3], colors[e4]
                            # Check if {c, c2, c3, c4} has 4 distinct elements
                            if len({c, c2, c3, c4}) == 4:
                                is_valid = False
                                break
                    if is_valid:
                        colors[i] = c
                        used_colors.add(c)
                        found = True
                        break
                        
                if not found:
                    # Fallback: try any used color
                    for c in used_colors:
                        is_valid = True
                        for e2, e3, e4 in cycle_info[i]:
                            if colors[e2] != -1 and colors[e3] != -1 and colors[e4] != -1:
                                c2, c3, c4 = colors[e2], colors[e3], colors[e4]
                                if len({c, c2, c3, c4}) == 4:
                                    is_valid = False
                                    break
                        if is_valid:
                            colors[i] = c
                            used_colors.add(c)
                            found = True
                            break
                            
                if not found:
                    # Last resort: 0
                    colors[i] = 0
                    used_colors.add(0)
                    
        # Verify result
        # Count distinct colors
        score = len(used_colors)
        if score > best_score:
            best_score = score
            best_colors = colors[:]
            
    if best_colors is None:
        return [0] * NUM_EDGES
        
    return best_colors
