"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Build initial n=32 coloring (Greenwood-Gleason)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n = 32
    # color_matrix[i][j] = color
    color_matrix = [[0] * n for _ in range(n)]
    # adj[c][i] = bitmask of neighbors j
    adj = [[0] * n for _ in range(4)]
    
    for i in range(n):
        for j in range(i + 1, n):
            c = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
            color_matrix[i][j] = c
            color_matrix[j][i] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    current_n = n
    best_n = n
    
    def get_cost(cm, ad, n_val):
        cost = 0
        for c in range(4):
            for i in range(n_val):
                mask_i = ad[c][i]
                # Iterate over set bits in mask_i
                # m = mask_i
                # while m:
                #   lsb = m & -m
                #   j = lsb.bit_length() - 1
                #   if j > i:
                #       common = mask_i & ad[c][j]
                #       cost += common.bit_count()
                #   m ^= lsb
                # This is O(N^2) roughly.
                m = mask_i
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if j > i:
                        common = mask_i & ad[c][j]
                        cost += common.bit_count()
                    m ^= lsb
        return cost // 3
        
    def run_sa(cm, ad, n_val, time_limit):
        T = 1.0
        current_cost = get_cost(cm, ad, n_val)
        best_cost = current_cost
        best_cm = [row[:] for row in cm]
        best_ad = [row[:] for row in ad]
        
        start = time.time()
        while time.time() - start < time_limit:
            i = random.randint(0, n_val - 1)
            j = random.randint(0, n_val - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            c_old = cm[i][j]
            # Pick random new color
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            lost = (ad[c_old][i] & ad[c_old][j]).bit_count()
            gained = (ad[c_new][i] & ad[c_new][j]).bit_count()
            delta = gained - lost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                ad[c_old][i] &= ~(1 << j)
                ad[c_old][j] &= ~(1 << i)
                ad[c_new][i] |= (1 << j)
                ad[c_new][j] |= (1 << i)
                
                cm[i][j] = c_new
                cm[j][i] = c_new
                
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_cm = [row[:] for row in cm]
                    best_ad = [row[:] for row in ad]
                    
        return best_cm, best_ad, best_cost

    while time.time() < deadline and current_n < 50:
        # Try to extend
        new_n = current_n + 1
        # Copy state
        cm_new = [row[:] for row in color_matrix]
        # Expand cm_new
        for i in range(current_n):
            cm_new[i].append(0)
        cm_new.append([0] * new_n)
        
        ad_new = [row[:] for row in adj]
        for c in range(4):
            ad_new[c].append(0)
            
        # Randomize new edges
        for i in range(current_n):
            c = random.randint(0, 3)
            cm_new[i][new_n - 1] = c
            cm_new[new_n - 1][i] = c
            ad_new[c][i] |= (1 << (new_n - 1))
            ad_new[c][new_n - 1] |= (1 << i)
            
        # Run SA
        cm_res, ad_res, cost_res = run_sa(cm_new, ad_new, new_n, 5.0)
        
        if cost_res == 0:
            color_matrix = cm_res
            adj = ad_res
            current_n = new_n
            best_n = current_n
        else:
            # Perturbation: run SA on current state to change basin
            # Only if we have time
            if time.time() < deadline - 10:
                cm_res, ad_res, cost_res = run_sa(color_matrix, adj, current_n, 2.0)
                if cost_res == 0:
                    color_matrix = cm_res
                    adj = ad_res
                # If cost_res > 0, we should revert to last valid state
                # But we started with a valid state, so if SA didn't find 0, 
                # it might have degraded. We should keep the best valid state.
                # run_sa returns best found. If best found cost > 0, we discard it.
                # So we do nothing if cost_res > 0.
                
    # Generate output
    colors = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            colors.append(color_matrix[i][j])
            
    return {'n': current_n, 'colors': colors}

# EVOLVE-BLOCK-END
