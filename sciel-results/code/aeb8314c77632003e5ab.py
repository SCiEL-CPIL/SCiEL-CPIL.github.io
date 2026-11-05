import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    num_colors = 5
    
    def count_c4s(col, n):
        """Count total monochromatic C4s using bitmasks."""
        total = 0
        for c in range(num_colors):
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if col[i][j] == c:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            for u in range(n):
                for v in range(u+1, n):
                    common = adj_mask[u] & adj_mask[v]
                    k = bin(common).count('1')
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def is_valid(col, n):
        return count_c4s(col, n) == 0

    def try_backtracking(n):
        """
        Backtracking with forward-checking: assign colors to edges in lexicographic order.
        Prune if any assignment creates a monochromatic C4.
        """
        col = [[-1]*n for _ in range(n)]
        
        # Precompute edge list in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # For forward-checking: when assigning color c to edge (u,v),
        # check if this creates a C4 in color c.
        # A C4 in color c exists if there are vertices a,b such that
        # both (a,u) and (a,v) have color c, OR (b,u) and (b,v) have color c.
        # More precisely: after assigning (u,v) to color c, check all pairs (a,b)
        # where a < b, a != u,v, b != u,v, and see if both a and b are common
        # neighbors of u and v in color c.
        
        def would_create_c4(col, n, u, v, c):
            """Check if assigning color c to (u,v) creates a monochromatic C4 in color c."""
            # Find all vertices that are connected to both u and v with color c
            common_neighbors = []
            for w in range(n):
                if w == u or w == v:
                    continue
                # Check if (u,w) and (v,w) both have color c
                if col[u][w] == c and col[v][w] == c:
                    common_neighbors.append(w)
            
            # If there are 2+ common neighbors, a C4 is formed
            if len(common_neighbors) >= 2:
                return True
            
            # Also check: does (u,v) complete a C4 where u and v are opposite vertices?
            # A C4 is u-a-v-b-u. If we assign (u,v) to color c, we need to check
            # if there exist a,b such that (u,a), (a,v), (v,b), (b,u) are all color c.
            # But this is already covered: if a and b are both common neighbors of u and v
            # in color c, then u-a-v-b-u is a C4.
            
            return False

        def backtrack(idx):
            """Recursively assign colors to edges[idx:]."""
            if time.time() > deadline:
                return False
            
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Try each color
            for c in range(num_colors):
                if would_create_c4(col, n, u, v, c):
                    continue
                
                col[u][v] = c
                col[v][u] = c
                
                if backtrack(idx + 1):
                    return True
                
                col[u][v] = -1
                col[v][u] = -1
            
            return False

        if backtrack(0):
            return col
        return None

    def try_sa(n):
        """Simulated annealing fallback."""
        col = [[random.randint(0, num_colors-1) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                col[j][i] = col[i][j]
        
        current_penalty = count_c4s(col, n)
        if current_penalty == 0:
            return col
        
        best_local_penalty = current_penalty
        best_local_col = [row[:] for row in col]
        no_improve = 0
        temp = 1.0
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c:
                continue
            
            col[i][j] = new_c
            col[j][i] = new_c
            
            new_penalty = count_c4s(col, n)
            delta = new_penalty - current_penalty
            
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / temp)):
                current_penalty = new_penalty
                no_improve = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = [row[:] for row in col]
            else:
                col[i][j] = old_c
                col[j][i] = old_c
                no_improve += 1
            
            if no_improve > 500:
                break
            temp *= 0.995
        
        return best_local_col

    # Try backtracking first for smaller n, then SA for larger
    best_sol = None
    best_n = 0
    
    # Try backtracking for n up to 15 (it's exponential, so keep n small)
    for n_try in [12, 13, 14, 15]:
        if time.time() > deadline - 30:
            break
        sol = try_backtracking(n_try)
        if sol is not None and is_valid(sol, n_try):
            best_sol = sol
            best_n = n_try
            print(f"Backtracking found valid solution for n={n_try}")
    
    # If backtracking didn't find a good solution, try SA for larger n
    for n_try in [16, 18, 20, 22, 24, 26]:
        if time.time() > deadline - 20:
            break
        sol = try_sa(n_try)
        if sol is not None and is_valid(sol, n_try):
            best_sol = sol
            best_n = n_try
            print(f"SA found valid solution for n={n_try}")
            break
    
    if best_sol is not None:
        res_colors = []
        for i in range(best_n):
            for j in range(i+1, best_n):
                res_colors.append(best_sol[i][j])
        return {'n': best_n, 'colors': res_colors}
    
    # Fallback to n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
