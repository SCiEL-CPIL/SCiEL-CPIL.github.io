"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        sum_all = np.dot(diff_a, diff_b)
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        return int(2 * (sum_all - term_i - term_j))
    
    def first_improvement_ls(p, current_cost, max_passes=50):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() - start_time > timeout:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, current_cost
    
    def best_improvement_ls(p, current_cost):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        return p, current_cost
    
    def lap_guided_perturb(p, current_cost, k):
        """Use LAP to propose a globally consistent reassignment for k facilities."""
        # Select k facilities with highest flow sums to reassign
        flow_sums = a.sum(axis=1)
        # Choose k random facilities, biased toward high-flow ones
        weights = flow_sums + 1
        weights = weights / weights.sum()
        selected = np.random.choice(n, size=k, replace=False, p=weights)
        
        # For these facilities, solve a small LAP against remaining locations
        # Actually, let's do a 2-opt style: pick k pairs and swap based on LAP guidance
        # Simpler: do k random swaps but guided by gradient
        for _ in range(k):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        return p, current_cost
    
    def simulated_annealing_step(p, current_cost, temperature):
        """SA move: accept worse moves with probability exp(-delta/T)."""
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            return p, current_cost, False
        delta = get_delta(p, i, j)
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            return p, current_cost, True
        return p, current_cost, False
    
    # Initial solution: rank matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local search on initial solution
    p, current_cost = first_improvement_ls(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # ILS with simulated annealing perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve = 0
    np.random.seed(42)
    
    # SA parameters
    T_init = 500.0
    T_min = 1.0
    cooling_rate = 0.995
    
    temperature = T_init
    
    while time.time() - start_time < timeout - 3.0:
        # Adaptive perturbation
        if no_improve < 5:
            k = 2
        elif no_improve < 12:
            k = 4
        elif no_improve < 25:
            k = 8
        else:
            k = 15
        
        # Perturb with SA-style moves
        for _ in range(k):
            if time.time() - start_time > timeout - 3.0:
                break
            p, current_cost, _ = simulated_annealing_step(p, current_cost, temperature)
        
        # Local search
        p, current_cost = first_improvement_ls(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
            # Reset temperature on improvement
            temperature = max(T_init * 0.5, T_min)
        else:
            no_improve += 1
            p = best_p.copy()
            current_cost = best_cost
        
        # Cool down
        temperature = max(T_min, temperature * cooling_rate)
        
        if no_improve > 40:
            # Restart with higher temperature
            temperature = T_init
            no_improve = 0
            p = best_p.copy()
            current_cost = best_cost
    
    # Final polish with best-improvement
    p = best_p.copy()
    current_cost = best_cost
    if time.time() - start_time < timeout - 0.5:
        p, current_cost = best_improvement_ls(p, current_cost)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
