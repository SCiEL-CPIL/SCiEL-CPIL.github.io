"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    def compute_cost(p):
        return int(np.sum(af * bf[np.ix_(p, p)]))
        
    def delta_swap(p, i, j):
        x = p[i]
        y = p[j]
        if x == y:
            return 0.0
            
        # Vectorized delta calculation
        # We need sum over k != i, j
        # Terms:
        # 1. a[i, k] * (b[y, p[k]] - b[x, p[k]])
        # 2. a[k, i] * (b[p[k], y] - b[p[k], x])
        # 3. a[j, k] * (b[x, p[k]] - b[y, p[k]])
        # 4. a[k, j] * (b[p[k], x] - b[p[k], y])
        
        # Create a mask or just use slices?
        # For n=150, creating a mask is fine.
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # To avoid creating a new array for indices, we can compute the full sum and subtract i and j terms?
        # Let's try full sum minus i,j terms.
        
        # S1 = sum_k a[i, k] * b[y, p[k]] - sum_k a[i, k] * b[x, p[k]]
        # This is dot product of a[i, :] with b[y, p] and b[x, p].
        
        # Let's just do the masked version as it's safe and n is small.
        ks = np.where(mask)[0]
        pk = p[ks]
        
        # Pre-extract rows/cols
        ai = af[i, ks]
        ki = af[ks, i]
        aj = af[j, ks]
        kj = af[ks, j]
        
        by_pk = bf[y, pk]
        bx_pk = bf[x, pk]
        bpk_y = bf[pk, y]
        bpk_x = bf[pk, x]
        
        term1 = np.dot(ai, by_pk) - np.dot(ai, bx_pk)
        term2 = np.dot(ki, bpk_y) - np.dot(ki, bpk_x)
        term3 = np.dot(aj, bx_pk) - np.dot(aj, by_pk)
        term4 = np.dot(kj, bpk_x) - np.dot(kj, bpk_y)
        
        sum_k = term1 + term2 + term3 + term4
        
        diag_old = af[i, j] * bf[x, y] + af[j, i] * bf[y, x]
        diag_new = af[i, j] * bf[y, x] + af[j, i] * bf[x, y]
        
        return (diag_new - diag_old) + sum_k

    # Initialization: Random
    p = np.random.permutation(n)
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115
    
    # Simulated Annealing parameters
    T = 1000.0 # Initial temperature
    T_min = 1.0
    cooling_rate = 0.99
    
    # Estimate a good T based on typical delta magnitude?
    # Let's sample a few deltas to set T.
    if n > 2:
        deltas = []
        for _ in range(10):
            i, j = np.random.choice(n, 2, replace=False)
            deltas.append(abs(delta_swap(p, i, j)))
        if deltas:
            T = max(1.0, np.mean(deltas) * 10)
            
    random.seed(42)
    np.random.seed(42)
    
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        # Check for restart
        if no_improve_steps > max_no_improve:
            # Restart from best
            p = best_p.copy()
            current_cost = best_cost
            T = T * 0.5 # Cool down or restart temp? Let's restart temp to explore
            T = max(100.0, T)
            no_improve_steps = 0
            # Perturb best solution
            for _ in range(5):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            current_cost = compute_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        # Perform one swap
        i, j = np.random.choice(n, 2, replace=False)
        d = delta_swap(p, i, j)
        
        if d < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Metropolis acceptance
            if T > T_min:
                # Avoid overflow in exp
                if d / T < 50: 
                    if random.random() < np.exp(-d / T):
                        p[i], p[j] = p[j], p[i]
                        current_cost += d
                        no_improve_steps += 1
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
