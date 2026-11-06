import random
import time
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # Helper to check if a set of edges is valid
    def is_valid(edges):
        if not edges:
            return True
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        # Check all 4-cycles
        # u < v < w < x is not enough, we need all combinations
        # Iterate over all pairs of edges that could form opposite sides?
        # Simpler: Iterate over all u, v, w, x distinct
        # u-v-w-x-u
        for u in range(n):
            for v in range(n):
                if u == v or not adj[u][v]: continue
                for w in range(n):
                    if w == u or w == v or not adj[v][w]: continue
                    for x in range(n):
                        if x == u or x == v or x == w: continue
                        if not adj[w][x] or not adj[x][u]: continue
                        # Check colors
                        c1 = color_mat[u][v]
                        c2 = color_mat[v][w]
                        c3 = color_mat[w][x]
                        c4 = color_mat[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    def get_valid_colors(u, v, adj, color_mat, neighbors):
        forbidden = set()
        
        # Check all 4-cycles u-v-w-x-u
        for w in neighbors[v]:
            if w == u: continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        return set(range(num_colors)) - forbidden

    def run_greedy_bipartite():
        """
        Attempts to construct a valid complete bipartite graph K_10,10
        and then adds intra-partition edges.
        """
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        U = list(range(10))
        V = list(range(10, 20))
        
        # 1. Try to color K_10,10
        # We use a backtracking-like greedy with random restarts for the bipartite part
        success = False
        for attempt in range(50):
            random.seed(1000 + attempt)
            adj_temp = [[0]*n for _ in range(n)]
            color_temp = [[-1]*n for _ in range(n)]
            edges_temp = []
            
            # Order bipartite edges
            bip_edges = [(u, v) for u in U for v in V]
            random.shuffle(bip_edges)
            
            valid = True
            for u, v in bip_edges:
                # Get valid colors considering only previously added edges
                # Since it's bipartite, 4-cycles only form when both rows and cols are filled
                forbidden = set()
                # Check 4-cycles u-v-w-x-u
                # w in V, x in U
                # w != v, x != u, w ~ x (edge w-x doesn't exist in bipartite unless we added it? No, bipartite edges are U-V)
                # Wait, in K_10,10, edges are only between U and V.
                # A 4-cycle is u1-v1-u2-v2-u1.
                # Edges: (u1,v1), (v1,u2), (u2,v2), (v2,u1).
                # All are bipartite edges.
                
                for w in V:
                    if w == v: continue
                    if not adj_temp[v][w]: continue # v-w edge must exist
                    # v is in V, w is in V? No, v in V, w in V. Edge (v,w) doesn't exist in bipartite.
                    # My adjacency check `adj_temp[v][w]` will be 0 for V-V edges.
                    # So this loop does nothing for bipartite?
                    # Correct. In a bipartite graph, neighbors of v (in V) are in U.
                    # So `for w in neighbors[v]` would iterate over U.
                    
                # Let's use the general get_valid_colors logic but with temp structures
                # We need to pass adj_temp, color_temp, neighbors_temp
                # But neighbors_temp is not built.
                
                # Let's just use the general function with temp structures
                # We need to build neighbors_temp
                # To save time, we'll just use the general greedy on the full graph but with a bias?
                
                pass
            
            # The bipartite specific logic is tricky to isolate quickly.
            # Let's stick to a robust full-graph greedy with better heuristics.
            break # Skip this complex path for now, use the improved general greedy below

    def run_greedy_improved(seed=None):
        if seed is not None:
            random.seed(seed)
        
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Heuristic: Try to add edges that are "central" first? Or random?
        # Random is safer.
        random.shuffle(all_edges)
        
        # To improve density, we can try multiple orders if the first one fails early
        # But for speed, we just do one pass.
        
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

    # Strategy:
    # 1. Run many random restarts of the improved greedy.
    # 2. Keep the best result.
    
    restarts = 0
    while time.time() < deadline:
        edges = run_greedy_improved(seed=restarts)
        
        # Verify validity (cheap for small graphs, but crucial)
        if is_valid(edges):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                # If we reach a very high number, we can stop
                if best_count > 100:
                    break
        else:
            # If invalid, try to repair or just discard
            # For now, discard
            pass
        restarts += 1
        
        # If we've run many restarts and haven't improved, maybe increase randomization?
        # The random seed handles this.

    return best_edges
