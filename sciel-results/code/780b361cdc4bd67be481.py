# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def calc_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        if u == v:
            return 0.0
            
        d1 = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        A_diff_row = a[i, :] - a[j, :]
        A_diff_col = a[:, i] - a[:, j]
        
        B_diff_row = b[v, p] - b[u, p]
        B_diff_col = b[p, v] - b[p, u]
        
        Sum1_all = np.dot(A_diff_row, B_diff_row)
        Sum2_all = np.dot(A_diff_col, B_diff_col)
        
        sub1_i = A_diff_row[i] * B_diff_row[i]
        sub2_i = A_diff_col[i] * B_diff_col[i]
        sub1_j = A_diff_row[j] * B_diff_row[j]
        sub2_j = A_diff_col[j] * B_diff_col[j]
        
        Sum1 = Sum1_all - sub1_i - sub1_j
        Sum2 = Sum2_all - sub2_i - sub2_j
        
        return d1 + Sum1 + Sum2

    def local_search(p, current_cost, max_time=10.0):
        """Steepest descent local search with time limit"""
        improved = True
        local_start = time.time()
        while improved and (time.time() - local_start) < max_time:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = calc_delta(i, j, p)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        return p, current_cost

    def simulated_annealing(p, current_cost, T_init, cooling_rate, min_T, max_time=30.0):
        """Simulated annealing with time limit"""
        T = T_init
        sa_start = time.time()
        rng = random.Random(42)
        while (time.time() - sa_start) < max_time and T > min_T:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            delta = calc_delta(i, j, p)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
            T *= cooling_rate
        return p, current_cost

    # Phase 1: Local search to polish initial solution
    p, current_cost = local_search(p.copy(), current_cost, max_time=15.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Phase 2: Multiple SA restarts with perturbation
    random.seed(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 25))
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        # Perturb the best solution
        p_restart = best_p.copy()
        current_cost_restart = best_cost
        
        # Apply random swaps (perturbation)
        num_perturb = random.randint(10, 30)
        for _ in range(num_perturb):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                delta = calc_delta(i, j, p_restart)
                p_restart[i], p_restart[j] = p_restart[j], p_restart[i]
                current_cost_restart += delta
        
        # Local search on perturbed solution
        p_restart, current_cost_restart = local_search(p_restart, current_cost_restart, max_time=8.0)
        
        if current_cost_restart < best_cost:
            best_cost = current_cost_restart
            best_p = p_restart.copy()
        
        # SA from the locally optimal perturbed solution
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            T_init = max(100.0, abs(current_cost_restart - best_cost) * 0.1 + 100.0)
            p_restart, current_cost_restart = simulated_annealing(
                p_restart, current_cost_restart, 
                T_init=T_init, cooling_rate=0.995, min_T=1.0,
                max_time=remaining
            )
            if current_cost_restart < best_cost:
                best_cost = current_cost_restart
                best_p = p_restart.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
