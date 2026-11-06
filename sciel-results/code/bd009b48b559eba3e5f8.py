import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns a set of valid colors for edge (u, v).
        If the set is empty, the edge cannot be added.
        """
        forbidden = set()
        # We need to check all 4-cycles u-v-w-x-u
        # w is neighbor of v, x is neighbor of u
        # w != u, x != v, w != x
        # w ~ x must hold
        
        for w in neighbors[v]:
            if w == u: continue
            cw_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                # Cycle u-v-w-x-u
                # Edges: (u,v) [new], (v,w), (w,x), (x,u)
                c_vw = cw_vw
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # The 3 existing edges must have distinct colors among themselves?
                # If not, the 4-cycle cannot be rainbow regardless of c(u,v).
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        valid_colors = set(range(num_colors)) - forbidden
        return valid_colors

    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
            
            if valid_colors:
                # Pick a random valid color
                c = random.choice(list(valid_colors))
                
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
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
