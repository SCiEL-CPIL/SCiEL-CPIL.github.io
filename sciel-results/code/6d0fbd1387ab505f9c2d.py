import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    
    def is_valid_addition(u, v, c, adj, color_mat):
        # Neighbors of u and v
        # Using lists for faster iteration if graph is sparse, but for dense, lists are fine.
        # Precomputing neighbors might be faster, but we update adj dynamically.
        # We can just iterate 0..n-1 and check adj.
        # Or maintain neighbor lists.
        # Let's stick to checking adj matrix directly for simplicity, n=20 is small.
        
        for w in range(n):
            if w == v: continue
            if not adj[v][w]: continue
            # w is a neighbor of v
            
            for x in range(n):
                if x == u or x == v or x == w: continue
                if not adj[u][x]: continue
                # x is a neighbor of u
                
                if adj[w][x]:
                    # Cycle u-v-w-x-u exists
                    c_vw = color_mat[v][w]
                    c_wx = color_mat[w][x]
                    c_xu = color_mat[x][u]
                    
                    # Check distinctness
                    if c == c_vw or c == c_wx or c == c_xu:
                        return False
                    if c_vw == c_wx or c_vw == c_xu:
                        # This would mean the existing graph was invalid, 
                        # but we assume it's valid. 
                        # However, if the existing graph was valid, these must be distinct.
                        # We only need to check the new color c against the others.
                        # Wait, if the existing graph was valid, then for any 4-cycle already present, colors are distinct.
                        # The cycle u-v-w-x-u is NEW because it includes (u,v).
                        # So (v,w), (w,x), (x,u) are existing edges.
                        # Do they form a path v-w-x-u? Yes.
                        # Does a 4-cycle v-w-x-u-... exist without (u,v)? No, it's a path.
                        # So the colors of (v,w), (w,x), (x,u) are not constrained to be distinct by previous 4-cycles involving them?
                        # Actually, if there was another 4-cycle using (v,w) and (w,x), it would have constrained them.
                        # But specifically, are c_vw, c_wx, c_xu distinct?
                        # Not necessarily. They are just edges in a path.
                        # However, for the NEW 4-cycle to be rainbow, ALL 4 must be distinct.
                        # So we must check:
                        # c != c_vw
                        # c != c_wx
                        # c != c_xu
                        # c_vw != c_wx
                        # c_vw != c_xu
                        # c_wx != c_xu
                        
                        # If the existing graph is valid, does it guarantee c_vw != c_wx?
                        # No. They are adjacent edges.
                        # But wait, if there is a 4-cycle containing both (v,w) and (w,x), say v-w-x-y-v, then c_vw, c_wx, c_xy, c_yv are distinct.
                        # This implies c_vw != c_wx.
                        # Is it possible that no 4-cycle contains both (v,w) and (w,x)?
                        # Yes.
                        # So we MUST check all pairwise distinctness among the 4 colors.
                        
                        pass # The logic below handles it

                    if len({c, c_vw, c_wx, c_xu}) < 4:
                        return False
                        
        return True

    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            added = False
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for c in colors:
                if is_valid_addition(u, v, c, adj, color_mat):
                    adj[u][v] = 1
                    adj[v][u] = 1
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    edges.append([u, v, c])
                    added = True
                    break
            
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
