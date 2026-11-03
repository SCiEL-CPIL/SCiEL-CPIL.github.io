"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _fast_delta(a, a_sum, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    a_sum is a + a.T
    D is the current distance matrix b[p, p].
    """
    # S = (a_sum[i] - a_sum[j]) dot (D[j] - D[i])
    diff_a = a_sum[i] - a_sum[j]
    diff_D = D[j] - D[i]
    S = np.dot(diff_a, diff_D)
    
    # Subtract k=i and k=j terms from S
    # T_i = (2*a[i,i] - a[j,i] - a[i,j]) * (D[j,i] - D[i,i])
    # T_j = (a[i,j] + a[j,i] - 2*a[j,j]) * (D[j,j] - D[i,j])
    
    # Note: a_sum[i,i] = 2*a[i,i]
    # a_sum[j,i] = a[j,i] + a[i,j]
    # So 2*a[i,i] - (a[j,i] + a[i,j]) is (a_sum[i,i] - a_sum[j,i])? 
    # a_sum[j,i] is element (j,i). a_sum[i,j] is element (i,j).
    # My derivation: T_i coeff was (2a[i,i] - a[j,i] - a[i,j]).
    # This is (a_sum[i,i] - a_sum[j,i])? No. a_sum[j,i] = a[j,i] + a[i,j].
    # So yes, coeff_i = a_sum[i,i] - a_sum[j,i].
    
    coeff_i = a_sum[i, i] - a_sum[j, i]
    term_i = coeff_i * (D[j, i] - D[i, i])
    
    coeff_j = a_sum[i, j] - a_sum[j, j]
    # Wait, my derivation for T_j coeff: (a[i,j] + a[j,i] - 2a[j,j])
    # a_sum[i,j] = a[i,j] + a[j,i].
    # a_sum[j,j] = 2a[j,j].
    # So coeff_j = a_sum[i,j] - a_sum[j,j].
    term_j = coeff_j * (D[j, j] - D[i, j])
    
    # Add back the specific diagonal and cross terms for Delta
    # Delta = S - T_i - T_j + (a[i,i] - a[j,j])*(D[j,j] - D[i,i]) + (a[i,j] - a[j,i])*(D[j,i] - D[i,j])
    
    diag_term = (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
    cross_term = (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
    
    return S - term_i - term_j + diag_term + cross_term

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    a_sum = a + a.T
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _fast_delta(a, a_sum, D, i, j)
                if delta < 0:
                    # Perform swap in p
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    # Swap rows i and j
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    # Swap cols i and j
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    current_cost += delta
                    improved = True
                    
    return p, current_cost

def _perturb(p, k=5):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial random permutation
    rng = np.random.default_rng(42)
    p_init = rng.permutation(n)
    
    # First local search on random init
    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, time_budget - 5, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Perturb
        p_pert = _perturb(best_p, k=5)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
