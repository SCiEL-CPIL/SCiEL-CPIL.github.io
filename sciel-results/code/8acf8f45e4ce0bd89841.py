"""Stronger QAP solver for wil100: incremental C/D updates, greedy + KL + perturbation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost."""
    return np.sum(a * b[np.ix_(p, p)])

def _compute_c(a, b, p):
    """
    Compute C[i, pi] = sum_k a[k,i] * b[p[k], pi] for each position i and location pi.
    Shape: (n, n)
    """
    n = len(p)
    B_perm = b[np.ix_(p, np.arange(n))]
    C = a.T @ B_perm
    return C

def _compute_d(a, b, p):
    """
    Compute D[i, pi] = sum_k a[i,k] * b[pi, p[k]] for each position i and location pi.
    Shape: (n, n)
    """
    n = len(p)
    B_perm = b[np.ix_(p, np.arange(n))]
    D = a @ B_perm.T
    return D

def _update_cd_after_swap(a, b, p, C, D, i, j):
    """
    Incrementally update C and D after swapping positions i and j.
    pi = p[i] (old location at i), pj = p[j] (old location at j).
    After swap: p[i] = pj, p[j] = pi.
    
    C[i, col] = sum_k a[k,i] * b[p[k], col]
    D[i, col] = sum_k a[i,k] * b[col, p[k]]
    
    Only C[i,*] and D[i,*] change (and C[j,*] and D[j,*]).
    
    For C:
    Old C[i, col] = sum_k a[k,i] * b[p[k], col]
    New C[i, col] = sum_k a[k,i] * b[p'[k], col]
    where p'[i] = pj (was pi), p'[j] = pi (was pj), others same.
    
    New C[i, col] = Old C[i, col] - a[i,i]*b[pi,col] - a[j,i]*b[pj,col] + a[i,i]*b[pj,col] + a[j,i]*b[pi,col]
                  = Old C[i, col] + a[i,i]*(b[pj,col] - b[pi,col]) + a[j,i]*(b[pi,col] - b[pj,col])
                  = Old C[i, col] + (a[i,i] - a[j,i]) * (b[pj,col] - b[pi,col])
    
    Similarly:
    New C[j, col] = Old C[j, col] + (a[j,j] - a[i,j]) * (b[pi,col] - b[pj,col])
                  = Old C[j, col] - (a[j,j] - a[i,j]) * (b[pj,col] - b[pi,col])
    
    For D:
    Old D[i, col] = sum_k a[i,k] * b[col, p[k]]
    New D[i, col] = sum_k a[i,k] * b[col, p'[k]]
    
    New D[i, col] = Old D[i, col] - a[i,i]*b[col,pi] - a[i,j]*b[col,pj] + a[i,i]*b[col,pj] + a[i,j]*b[col,pi]
                  = Old D[i, col] + a[i,i]*(b[col,pj] - b[col,pi]) + a[i,j]*(b[col,pi] - b[col,pj])
                  = Old D[i, col] + (a[i,i] - a[i,j]) * (b[col,pj] - b[col,pi])
    
    New D[j, col] = Old D[j, col] + (a[j,j] - a[j,i]) * (b[col,pi] - b[col,pj])
                  = Old D[j, col] - (a[j,j] - a[j,i]) * (b[col,pj] - b[col,pi])
    """
    pi = p[i]
    pj = p[j]
    
    # Update C rows
    diff_b = b[pj] - b[pi]  # shape (n,)
    C[i] += (a[i, i] - a[j, i]) * diff_b
    C[j] -= (a[j, j] - a[i, j]) * diff_b
    
    # Update D rows
    # D[i, col] += (a[i,i] - a[i,j]) * (b[col,pj] - b[col,pi])
    # b[col, pj] - b[col, pi] for all col = b[:, pj] - b[:, pi]
    diff_b_T = b[:, pj] - b[:, pi]  # shape (n,)
    D[i] += (a[i, i] - a[i, j]) * diff_b_T
    D[j] -= (a[j, j] - a[j, i]) * diff_b_T
    
    # Now update p
    p[i], p[j] = pj, pi
    
    return C, D, p

def _greedy_initial(a, b):
    """Rank-matching heuristic: high flow -> low distance."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _hungarian_initial(a, b):
    """
    Use Hungarian algorithm on a reduced cost matrix.
    Cost[i, j] = sum_k a[k,i] * b[j, k] + sum_k a[i,k] * b[k, j]
    This approximates the QAP cost as a linear assignment.
    """
    from scipy.optimize import linear_sum_assignment
    n = len(a)
    # C[i, j] = sum_k a[k,i] * b[j, k] = (a.T @ b)[i, j]
    # D[i, j] = sum_k a[i,k] * b[k, j] = (a @ b)[i, j]
    C = a.T @ b
    D = a @ b
    cost_mat = C + D
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _kl_search(a, b, p, deadline, max_rounds=10000):
    """
    Kernighan-Lin style local search with incremental C/D updates.
    """
    n = len(p)
    C = _compute_c(a, b, p)
    D = _compute_d(a, b, p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    idx = np.arange(n)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        C_pi = C[idx, p]  # C[i, p[i]]
        D_pi = D[idx, p]  # D[i, p[i]]
        
        C_ij = C[:, p]  # C_ij[i,j] = C[i, p[j]]
        D_ij = D[:, p]  # D_ij[i,j] = D[i, p[j]]
        
        delta_mat = (C_ij - C_pi[:, None] + C_pi[None, :] - C_ij.T 
                   + D_ij - D_pi[:, None] + D_pi[None, :] - D_ij.T)
        
        np.fill_diagonal(delta_mat, np.inf)
        
        min_idx = np.unravel_index(np.argmin(delta_mat), delta_mat.shape)
        i, j = min_idx
        min_delta = delta_mat[i, j]
        
        if min_delta >= 0:
            break
        
        # Apply swap with incremental update
        C, D, p = _update_cd_after_swap(a, b, p, C, D, i, j)
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 145.0
    start = time.time()
    
    np.random.seed(42)
    
    # Try multiple initializations
    best_p = None
    best_cost = np.inf
    
    # Greedy initialization
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Hungarian initialization
    p_hung = _hungarian_initial(a, b)
    cost_hung = _qap_cost(a, b, p_hung)
    
    # Pick better initial
    if cost_greedy <= cost_hung:
        p_init = p_greedy
        cost_init = cost_greedy
    else:
        p_init = p_hung
        cost_init = cost_hung
    
    # KL search from best initial
    p_ls, cost_ls = _kl_search(a, b, p_init.copy(), deadline - 5)
    if cost_ls < best_cost:
        best_p = p_ls
        best_cost = cost_ls
    
    # Try perturbation + KL search
    while time.time() < deadline - 10:
        p_pert = best_p.copy()
        num_swaps = np.random.randint(3, 10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        remaining = deadline - 10 - (time.time() - start)
        if remaining < 3:
            break
        p_ls, cost_ls = _kl_search(a, b, p_pert.copy(), time.time() + remaining)
        if cost_ls < best_cost:
            best_p = p_ls
            best_cost = cost_ls
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
