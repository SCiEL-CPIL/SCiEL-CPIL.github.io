import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # We aim for n=17. If we can't find it, we return the best we found.
    # The seed program found ~12. We want to beat that.
    
    def has_simple_path_len_5(u, v, adj_masks):
        """
        Checks if there is a simple path of length 5 from u to v in the graph defined by adj_masks.
        adj_masks is a list of integers (bitmasks).
        """
        # DFS with bitmask for visited nodes
        # We want a path u -> x1 -> x2 -> x3 -> x4 -> v
        # This is a path of 5 edges.
        
        def dfs(node, depth, visited_mask):
            if depth == 5:
                return node == v
            # Get neighbors of node
            # adj_masks[node] contains bits for neighbors
            # We need to iterate through set bits
            neighbors = adj_masks[node]
            # Exclude visited nodes
            neighbors &= ~visited_mask
            
            while neighbors:
                # Extract lowest set bit
                lsb = neighbors & -neighbors
                nei = lsb.bit_length() - 1
                neighbors ^= lsb
                
                if dfs(nei, depth + 1, visited_mask | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def creates_c6(adj_masks, c, u, v):
        """
        Checks if adding edge (u, v) with color c creates a C6.
        This is equivalent to checking if there is a path of length 5 between u and v
        in the graph formed by edges of color c (excluding u-v which is not there yet).
        """
        if u == v:
            return False
        return has_simple_path_len_5(u, v, adj_masks[c])

    def try_find_solution(n_target):
        E = n_target * (n_target - 1) // 2
        edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        
        # We will try a few random permutations of edges
        for attempt in range(50):
            if time.time() > deadline:
                return None
                
            random.shuffle(edges)
            
            # adj_masks[c][i] is a bitmask of neighbors of i in color c
            adj_masks = [[0] * n_target for _ in range(4)]
            color_matrix = [[-1] * n_target for _ in range(n_target)]
            
            def backtrack(idx):
                if time.time() > deadline:
                    raise TimeoutError()
                
                if idx == len(edges):
                    return True
                    
                u, v = edges[idx]
                
                # Try colors in random order
                col_order = [0, 1, 2, 3]
                random.shuffle(col_order)
                
                for c in col_order:
                    # Check if adding (u,v) to color c creates a C6
                    if creates_c6(adj_masks, c, u, v):
                        continue
                    
                    # Assign color c
                    adj_masks[c][u] |= (1 << v)
                    adj_masks[c][v] |= (1 << u)
                    color_matrix[u][v] = c
                    color_matrix[v][u] = c
                    
                    if backtrack(idx + 1):
                        return True
                        
                    # Unassign
                    adj_masks[c][u] &= ~(1 << v)
                    adj_masks[c][v] &= ~(1 << u)
                    color_matrix[u][v] = -1
                    color_matrix[v][u] = -1
                    
                return False
            
            try:
                if backtrack(0):
                    res_colors = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            res_colors.append(color_matrix[i][j])
                    return res_colors
            except TimeoutError:
                return None
            except RecursionError:
                return None
                
        return None

    best_n = 9
    best_colors = None
    
    # Seed solution for n=9 (from current program)
    n = 9
    best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    # Try to find larger n
    # Start from 17 and go down? Or 10 and up?
    # Since 17 is the known max, trying 17 first is good if we can.
    # But 17 is very hard. 16 is also hard.
    # Let's try 17, then 16, then 15...
    # If we find 17, we are done.
    
    for n_target in range(17, 9, -1):
        if time.time() > deadline - 5: # Leave 5s for returning
            break
            
        sol = try_find_solution(n_target)
        if sol is not None:
            best_n = n_target
            best_colors = sol
            break # Found the largest possible (or at least a valid one for this n)
            
    return {'n': best_n, 'colors': best_colors}
