"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (150s kill)
    
    # Helper: Calculate exact cost
    def calc_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    # Precompute row sums for a faster delta approximation if needed, 
    # but we stick to the exact O(n) delta for correctness.
    
    # Helper: Calculate delta for swapping facilities i and j in permutation p
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        
        # Vectorized computation for the sum part
        diff_a = a[i] - a[j]
        diff_b = b[lj, p] - b[li, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract terms for k=i and k=j to avoid double counting or incorrect inclusion
        term_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
        term_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
        
        off_diag_sum = total_dot - term_i - term_j
        
        diag_delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
        
        return int(diag_delta + 2 * off_diag_sum)

    # Initial Solution: Linear Assignment on Row Sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # Strategy: Iterated Local Search with Adaptive Perturbation and "Error Repair"
    # We maintain a single current solution and a best solution.
    # The "error repair" mechanism is a specific type of perturbation where we
    # intentionally degrade the solution (make bad moves) but then immediately
    # run a short, aggressive local search to "repair" the damage, hoping to
    # escape the local optimum basin.
    
    current_p = p_init.copy()
    current_cost = best_cost
    
    num_iter = 0
    max_iter = 10000  # Just a cap, time is the real limit
    
    # Adaptive perturbation strength
    perturb_strength = 5
    
    while time.time() < deadline - 5.0 and num_iter < max_iter:
        num_iter += 1
        
        # 1. Local Search (First Improvement 2-Opt)
        # We run until a local optimum is reached or a time slice expires.
        # To allow for more iterations, we limit the number of passes in each LS phase.
        ls_start = time.time()
        ls_time_limit = min(10.0, max(1.0, (deadline - time.time()) / 20.0)) # Dynamic time limit
        
        improved = True
        pass_count = 0
        max_passes = 50
        
        while improved and pass_count < max_passes:
            if time.time() - ls_start > ls_time_limit:
                break
                
            improved = False
            pass_count += 1
            
            # Randomize starting point to avoid cyclic behavior
            start_i = random.randint(0, n - 1)
            
            for k in range(n):
                i = (start_i + k) % n
                if time.time() - ls_start > ls_time_limit:
                    break
                    
                u = current_p[i]
                
                start_j = random.randint(0, n - 1)
                for m in range(n):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() - ls_start > ls_time_limit:
                        break
                    
                    delta = swap_delta(current_p, i, j)
                    
                    if delta < 0:
                        # Perform swap
                        v = current_p[j]
                        current_p[i], current_p[j] = v, u
                        current_cost += delta
                        
                        # Update best
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                            
                        improved = True
                        break # First improvement found, restart inner loop
                    
                if improved:
                    break
            
        # 2. Perturbation (Error Repair Strategy)
        # Instead of random swaps, we do "destructive" moves followed by "constructive" repair.
        # Here, we simulate "error" by making K random swaps (regardless of cost),
        # then immediately run a short local search to repair.
        
        if time.time() > deadline - 10.0:
            break
            
        # Adaptive perturbation: if we haven't improved best in a while, perturb more
        # For simplicity, we keep it fixed but could adapt based on stagnation counter.
        
        num_swaps = perturb_strength
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                current_p[i], current_p[j] = current_p[j], current_p[i]
        
        current_cost = calc_cost(current_p)
        
        # "Repair" Phase: A very short local search to recover from the perturbation
        # This is the "error repair" part: we don't just jump, we try to land near a good spot.
        repair_start = time.time()
        repair_time_limit = min(2.0, max(0.5, (deadline - time.time()) / 50.0))
        
        improved = True
        while improved and time.time() - repair_start < repair_time_limit:
            improved = False
            start_i = random.randint(0, n - 1)
            
            for k in range(n):
                i = (start_i + k) % n
                if time.time() - repair_start > repair_time_limit:
                    break
                    
                u = current_p[i]
                start_j = random.randint(0, n - 1)
                for m in range(n):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() - repair_start > repair_time_limit:
                        break
                    
                    delta = swap_delta(current_p, i, j)
                    
                    if delta < 0:
                        v = current_p[j]
                        current_p[i], current_p[j] = v, u
                        current_cost += delta
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                            
                        improved = True
                        break
                if improved:
                    break

    # Final check
    if sorted(best_p.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
