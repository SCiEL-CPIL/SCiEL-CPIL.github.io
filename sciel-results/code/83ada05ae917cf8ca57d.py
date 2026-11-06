import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible_edges = len(all_edges)
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns a set of valid colors for edge (u, v).
        If the set is empty, the edge cannot be added.
        """
        forbidden = set()
        
        # Check all 4-cycles u-v-w-x-u
        # w is neighbor of v, x is neighbor of u
        # w != u, x != v, w != x
        # w ~ x must hold
        
        for w in neighbors[v]:
            if w == u: 
                continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: 
                    continue
                if x == w: 
                    continue
                if not adj[w][x]: 
                    continue
                
                # Cycle u-v-w-x-u
                # Edges: (u,v) [new], (v,w), (w,x), (x,u)
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # The 3 existing edges must have distinct colors among themselves
                # If not, the 4-cycle cannot be rainbow regardless of c(u,v)
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                # The new color must be distinct from the existing 3
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
        blocked_edges = []
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            valid_colors = get_valid_colors(u, v, adj, color_mat, neighbors)
            
            if valid_colors:
                c = random.choice(list(valid_colors))
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            else:
                blocked_edges.append((u, v))
                
        return edges, blocked_edges, adj, color_mat, neighbors

    def local_search(edges, blocked_edges, adj, color_mat, neighbors, seed=None):
        """
        Tries to improve the solution by recoloring existing edges to allow adding blocked edges.
        """
        if seed is not None:
            random.seed(seed)
            
        current_edges = list(edges)
        current_blocked = list(blocked_edges)
        current_adj = [row[:] for row in adj]
        current_color_mat = [row[:] for row in color_mat]
        current_neighbors = [set(s) for s in neighbors]
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any blocked edge
            # We check all blocked edges to see if they are now unblocked
            new_blocked = []
            for u, v in current_blocked:
                valid_colors = get_valid_colors(u, v, current_adj, current_color_mat, current_neighbors)
                if valid_colors:
                    c = random.choice(list(valid_colors))
                    current_adj[u][v] = 1
                    current_adj[v][u] = 1
                    current_color_mat[u][v] = c
                    current_color_mat[v][u] = c
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                    current_edges.append([u, v, c])
                    improved = True
                    # Restart the blocked list scan or just break? 
                    # Breaking and restarting the while loop is safer to re-evaluate all blocked edges.
                    break
                else:
                    new_blocked.append((u, v))
            
            if improved:
                current_blocked = new_blocked
                no_improve_count = 0
                continue
            
            current_blocked = new_blocked
            
            # 2. If no edge could be added, try recoloring an existing edge
            if current_edges:
                # Pick a random edge
                edge_idx = random.randint(0, len(current_edges) - 1)
                u, v, c_old = current_edges[edge_idx]
                
                # Get all valid colors for this edge, EXCLUDING the current one?
                # No, we want to change it. So we look for valid colors != c_old.
                # But wait, changing the color might invalidate other 4-cycles?
                # No, the condition is symmetric. If a color is valid for (u,v), it's valid.
                # We need to find a new color c_new such that (u,v) with c_new is valid.
                
                valid_colors = get_valid_colors(u, v, current_adj, current_color_mat, current_neighbors)
                valid_colors.discard(c_old)
                
                if valid_colors:
                    c_new = random.choice(list(valid_colors))
                    
                    # Apply change
                    current_color_mat[u][v] = c_new
                    current_color_mat[v][u] = c_new
                    current_edges[edge_idx] = [u, v, c_new]
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If we've tried many times without improvement, we might want to restart or give up on this local search
                    if no_improve_count > 100:
                        break
            else:
                break
        
        # Rebuild edge list from adj/color_mat to be sure
        final_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if current_adj[u][v]:
                    final_edges.append([u, v, current_color_mat[u][v]])
        
        return final_edges

    restarts = 0
    while time.time() < deadline:
        edges, blocked_edges, adj, color_mat, neighbors = run_greedy(seed=restarts)
        
        # Try local search to improve
        if time.time() < deadline - 10:
            improved_edges = local_search(edges, blocked_edges, adj, color_mat, neighbors, seed=restarts)
            if improved_edges and len(improved_edges) > len(edges):
                edges = improved_edges
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        if best_count == total_possible_edges:
            break
            
        restarts += 1

    return best_edges
