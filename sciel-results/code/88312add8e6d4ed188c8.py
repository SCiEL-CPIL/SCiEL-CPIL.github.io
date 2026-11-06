import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Seed
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    # Try n=26
    n = 26
    # Init
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 4)
            C[i][j] = c
            C[j][i] = c
            
    # k[a][b][c] for a < b
    k = [[[0 for _ in range(5)] for _ in range(n)] for _ in range(n)]
    
    def get_k_val(a, b, c):
        if a > b: a, b = b, a
        return k[a][b][c]
        
    def set_k_val(a, b, c, val):
        if a > b: a, b = b, a
        k[a][b][c] = val
        
    def calc_cost_pair(a, b, c):
        if a > b: a, b = b, a
        val = k[a][b][c]
        return max(0, val - 1)
        
    # Init k and cost
    total_cost = 0
    for u in range(n):
        for v in range(u+1, n):
            for c in range(5):
                cnt = 0
                for w in range(n):
                    if w == u or w == v: continue
                    if C[u][w] == c and C[v][w] == c:
                        cnt += 1
                k[u][v][c] = cnt
                total_cost += max(0, cnt - 1)
                
    best_cost = total_cost
    best_C = [row[:] for row in C]
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 5.0:
        if T < min_T:
            break
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        c_old = C[u][v]
        c_new = random.randint(0, 4)
        if c_new == c_old: continue
        
        # Calculate delta cost
        delta_cost = 0
        
        # Affected pairs: (w, u) and (w, v) for all w != u, v
        # We need to store old costs to revert if rejected
        # Or just calculate delta and apply, then revert if needed.
        
        # List of (a, b, c) to update
        # And their old k values?
        # We can just calculate the change in cost for each affected pair.
        
        # For each w:
        #   Check C[w][u] and C[w][v]
        
        # To allow reverting, we can store the state of k and C.
        # But k is large.
        # Better: Calculate delta_cost. If accepted, apply changes to k and C.
        # If rejected, do nothing.
        
        # Wait, if we reject, we didn't change anything, so no revert needed.
        # If we accept, we must update k and C.
        
        # So:
        # 1. Calculate delta_cost.
        # 2. If delta_cost < 0 or random < exp(-delta/T):
        #    Accept: Update C and k.
        
        # Calculation:
        # For each w in range(n):
        #   if w == u or w == v: continue
        #   
        #   # Pair (w, u)
        #   a, b = (w, u) if w < u else (u, w)
        #   c_w_u = C[w][u]
        #   if c_w_u == c_old:
        #       # k[a][b][c_old] decreases by 1
        #       old_k = k[a][b][c_old]
        #       new_k = old_k - 1
        #       old_c = max(0, old_k - 1)
        #       new_c = max(0, new_k - 1)
        #       delta_cost += new_c - old_c
        #   if c_w_u == c_new:
        #       # k[a][b][c_new] increases by 1
        #       old_k = k[a][b][c_new]
        #       new_k = old_k + 1
        #       old_c = max(0, old_k - 1)
        #       new_c = max(0, new_k - 1)
        #       delta_cost += new_c - old_c
        #       
        #   # Pair (w, v)
        #   a, b = (w, v) if w < v else (v, w)
        #   c_w_v = C[w][v]
        #   if c_w_v == c_old:
        #       old_k = k[a][b][c_old]
        #       new_k = old_k - 1
        #       old_c = max(0, old_k - 1)
        #       new_c = max(0, new_k - 1)
        #       delta_cost += new_c - old_c
        #   if c_w_v == c_new:
        #       old_k = k[a][b][c_new]
        #       new_k = old_k + 1
        #       old_c = max(0, old_k - 1)
        #       new_c = max(0, new_k - 1)
        #       delta_cost += new_c - old_c
        #
        #   # Note: If c_w_u == c_old and c_w_v == c_old, both are processed.
        #   # If c_w_u == c_new and c_w_v == c_new, both are processed.
        #   # Can c_w_u == c_old and c_w_u == c_new? No.
        #   # Can c_w_u == c_old and c_w_v == c_new? Yes.
        
        # Optimization: Precompute lists of w for each color?
        # No, just loop.
        
        # Let's implement this loop.
        
        # To speed up, we can break early if delta_cost is already very large?
        # No, we need exact delta for SA.
        
        # Acceptance
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            # Apply changes
            C[u][v] = c_new
            C[v][u] = c_new
            total_cost += delta_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_C = [row[:] for row in C]
                if best_cost == 0:
                    # Found valid coloring
                    break
            
            # Update k
            for w in range(n):
                if w == u or w == v: continue
                # Pair (w, u)
                a, b = (w, u) if w < u else (u, w)
                c_w_u = C[w][u] # Note: C[w][u] is unchanged
                if c_w_u == c_old:
                    k[a][b][c_old] -= 1
                if c_w_u == c_new:
                    k[a][b][c_new] += 1
                    
                # Pair (w, v)
                a, b = (w, v) if w < v else (v, w)
                c_w_v = C[w][v]
                if c_w_v == c_old:
                    k[a][b][c_old] -= 1
                if c_w_v == c_new:
                    k[a][b][c_new] += 1
        # Else reject
        
        T *= cooling
        
    if best_cost == 0:
        # Extract colors
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(best_C[i][j])
        return {'n': n, 'colors': colors}
        
    # If not found, return seed
    return {'n': n_seed, 'colors': colors_seed}
