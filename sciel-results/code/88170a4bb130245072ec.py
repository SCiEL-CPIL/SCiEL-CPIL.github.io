import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        # adj_mask[i] is an integer bitmask of neighbors of i
        adj_mask = [0] * n
        # color_mat[i][j] is the color of edge (i,j), -1 if no edge
        color_mat = [[-1]*n for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if edge (u,v) can be added with some color
            # Forbidden colors are those that would create a non-rainbow 4-cycle
            forbidden = [False] * num_colors
            has_conflict = False # If any 4-cycle already has duplicate colors among its 3 existing edges, we can't fix it by choosing c(u,v), but we also can't add (u,v) if it creates a cycle with non-distinct 3 edges? 
            # Actually, if the 3 existing edges of a 4-cycle are not distinct, the graph is already invalid? 
            # No, the constraint is on the final graph. If we add (u,v), the 4-cycle must be rainbow.
            # If the other 3 edges already have a duplicate color, then no color for (u,v) can make it rainbow.
            # So we just mark all colors as forbidden for that cycle.
            
            valid = True
            # Iterate neighbors of v
            mask_v = adj_mask[v]
            mask_u = adj_mask[u]
            
            # Get list of neighbors for iteration
            # Since n=20, we can just loop 0..19
            # But using bit operations is faster
            # neighbors_v = [w for w in range(n) if (mask_v >> w) & 1]
            
            for w in range(n):
                if w == u: continue
                if not ((mask_v >> w) & 1): continue
                
                c_vw = color_mat[v][w]
                
                for x in range(n):
                    if x == v: continue
                    if x == w: continue
                    if not ((mask_u >> x) & 1): continue
                    if not ((adj_mask[w] >> x) & 1): continue # w and x must be connected
                    
                    # Cycle u-v-w-x-u
                    c_wx = color_mat[w][x]
                    c_xu = color_mat[x][u]
                    
                    # Check if the 3 existing edges are distinct
                    if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                        # This 4-cycle cannot be made rainbow by adding (u,v)
                        # So (u,v) cannot be added in this context?
                        # Wait, if we add (u,v), the cycle exists. If it's not rainbow, it's invalid.
                        # So we cannot add (u,v) if it completes a 4-cycle that is already "broken" (has duplicate colors among the other 3).
                        valid = False
                        break
                    else:
                        # The 3 edges are distinct.
                        # The new color c(u,v) must be different from all 3.
                        forbidden[c_vw] = True
                        forbidden[c_wx] = True
                        forbidden[c_xu] = True
            
            if not valid:
                continue
                
            # Find a valid color
            valid_colors = [c for c in range(num_colors) if not forbidden[c]]
            if valid_colors:
                c = random.choice(valid_colors)
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                color_mat[u][v] = c
                color_mat[v][u] = c
                edges.append([u, v, c])
                
        return edges

    restarts = 0
    while time.time() < deadline:
        edges = run_greedy(seed=restarts)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        if best_count == 190:
            break
            
        restarts += 1

    return best_edges
