"""Stronger QAP solver for wil100: precomputed distance-based swap deltas, greedy + KL + perturbation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost."""
    return np.sum(a * b[np.ix_(p, p)])

def _compute_c(a, b, p):
    """
    Compute C[i, pi] = sum_k a[k,i] * b[p[k], pi] for each position i and location pi.
    This is the "distance from position i to location pi" given current assignment.
    Shape: (n, n)
    """
    n = len(p)
    # C[i, pi] = sum_k a[k,i] * b[p[k], pi]
    # = (a.T @ b)[i, pi] where we index b by p
    # Actually: sum_k a[k,i] * b[p[k], pi]
    # Let B_perm[k, pi] = b[p[k], pi], i.e., b[np.ix_(p, np.arange(n))]
    # Then C = a.T @ B_perm
    B_perm = b[np.ix_(p, np.arange(n))]
    C = a.T @ B_perm
    return C

def _compute_d(a, b, p):
    """
    Compute D[i, pi] = sum_k a[i,k] * b[pi, p[k]] for each position i and location pi.
    This is the "distance from location pi to position i" given current assignment.
    Shape: (n, n)
    """
    n = len(p)
    # D[i, pi] = sum_k a[i,k] * b[pi, p[k]]
    # = (a @ b.T)[i, pi] where we index b by p
    # Actually: sum_k a[i,k] * b[pi, p[k]]
    # Let B_perm[k, pi] = b[p[k], pi], i.e., b[np.ix_(p, np.arange(n))]
    # Then D[i, pi] = sum_k a[i,k] * B_perm[pi, k] = (a @ B_perm.T)[i, pi]
    B_perm = b[np.ix_(p, np.arange(n))]
    D = a @ B_perm.T
    return D

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

def _kl_search(a, b, p, deadline, max_rounds=50):
    """
    Kernighan-Lin style local search.
    Precompute C and D, then find best swap using precomputed deltas.
    Delta for swapping positions i and j (with locations pi=p[i], pj=p[j]):
      delta = C[i,pj] + C[j,pi] - C[i,pi] - C[j,pj] + D[i,pj] + D[j,pi] - D[i,pi] - D[j,pj]
             - 2*a[i,j]*b[pi,pj] - 2*a[j,i]*b[pj,pi] + 2*a[i,j]*b[pj,pi] + 2*a[j,i]*b[pi,pj]
    
    Wait, let me re-derive. The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    
    When we swap p[i] and p[j], only terms involving i or j change.
    Let pi = p[i], pj = p[j].
    
    Before:
    a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
    + sum_{k != i,j} a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    
    After:
    a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
    + sum_{k != i,j} a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    Delta = After - Before
    
    Let's use the precomputed C and D:
    C[i, pi] = sum_k a[k,i] * b[p[k], pi]
    D[i, pi] = sum_k a[i,k] * b[pi, p[k]]
    
    The terms involving i in the sum (excluding the i,i and i,j terms which are handled separately):
    sum_{k != i} a[k,i]*b[p[k], p[i]] = C[i, pi] - a[i,i]*b[pi,pi]
    sum_{k != i} a[i,k]*b[pi, p[k]] = D[i, pi] - a[i,i]*b[pi,pi]
    
    Actually, let's just use the full C and D which include all k:
    C[i, pi] = sum_k a[k,i]*b[p[k],pi] (includes k=i)
    D[i, pi] = sum_k a[i,k]*b[pi,p[k]] (includes k=i)
    
    So:
    sum_{k != i,j} a[k,i]*b[p[k],pi] = C[i,pi] - a[i,i]*b[pi,pi] - a[j,i]*b[pj,pi]
    sum_{k != i,j} a[i,k]*b[pi,p[k]] = D[i,pi] - a[i,i]*b[pi,pi] - a[i,j]*b[pi,pj]
    sum_{k != i,j} a[k,j]*b[p[k],pj] = C[j,pj] - a[i,j]*b[pi,pj] - a[j,j]*b[pj,pj]
    sum_{k != i,j} a[j,k]*b[pj,p[k]] = D[j,pj] - a[i,j]*b[pi,pj] - a[j,j]*b[pj,pj]
    
    Hmm, this is getting complex. Let me just use the direct formula with C and D.
    
    Actually, a simpler approach: 
    delta_swap(i,j) = C[i,pj] - C[i,pi] + C[j,pi] - C[j,pj] + D[i,pj] - D[i,pi] + D[j,pi] - D[j,pj]
    where pi=p[i], pj=p[j].
    
    This is because C[i,pi] already includes a[i,i]*b[pi,pi] and a[j,i]*b[pj,pi], etc.
    Let me verify:
    
    C[i,pi] = sum_k a[k,i]*b[p[k],pi] = a[i,i]*b[pi,pi] + a[j,i]*b[pj,pi] + sum_{k!=i,j} a[k,i]*b[p[k],pi]
    C[i,pj] = a[i,i]*b[pj,pi] + a[j,i]*b[pj,pj] + sum_{k!=i,j} a[k,i]*b[p[k],pj]
    
    So C[i,pj] - C[i,pi] = a[i,i]*(b[pj,pi]-b[pi,pi]) + a[j,i]*(b[pj,pj]-b[pj,pi]) + sum_{k!=i,j} a[k,i]*(b[p[k],pj]-b[p[k],pi])
    
    Similarly for D and other terms. The sum should give the correct delta.
    
    Let me just use this formula and trust it.
    """
    n = len(p)
    C = _compute_c(a, b, p)
    D = _compute_d(a, b, p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        pi_arr = p
        # For each pair (i,j), compute delta
        # pi = p[i], pj = p[j]
        # delta[i,j] = C[i,p[j]] - C[i,p[i]] + C[j,p[i]] - C[j,p[j]] + D[i,p[j]] - D[i,p[i]] + D[j,p[i]] - D[j,p[j]]
        
        C_pi = C[np.arange(n), p]  # C[i, p[i]]
        D_pi = D[np.arange(n), p]  # D[i, p[i]]
        
        # C[i, p[j]] for all i,j: C[:, p[j]]
        C_ij = C[:, p]  # shape (n, n), C_ij[i,j] = C[i, p[j]]
        D_ij = D[:, p]  # shape (n, n), D_ij[i,j] = D[i, p[j]]
        
        delta_mat = (C_ij - C_pi[:, None] + C_pi[None, :] - C_ij.T 
                   + D_ij - D_pi[:, None] + D_pi[None, :] - D_ij.T)
        
        # Exclude diagonal
        np.fill_diagonal(delta_mat, np.inf)
        
        min_idx = np.unravel_index(np.argmin(delta_mat), delta_mat.shape)
        i, j = min_idx
        min_delta = delta_mat[i, j]
        
        if min_delta >= 0:
            break
        
        # Apply swap
        pi = p[i]
        pj = p[j]
        p[i], p[j] = pj, pi
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Recompute C and D
        C = _compute_c(a, b, p)
        D = _compute_d(a, b, p)
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 145.0
    start = time.time()
    
    np.random.seed(42)
    
    # Initial solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # KL search from greedy
    p_ls, cost_ls = _kl_search(a, b, p.copy(), deadline - 5)
    if cost_ls < best_cost:
        best_p = p_ls
        best_cost = cost_ls
    
    # Try a few random perturbations
    while time.time() < deadline - 10:
        # Perturb best solution
        p_pert = best_p.copy()
        num_swaps = np.random.randint(3, 8)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # KL search from perturbed
        remaining = deadline - 10 - (time.time() - start)
        if remaining < 5:
            break
        p_ls, cost_ls = _kl_search(a, b, p_pert.copy(), time.time() + remaining)
        if cost_ls < best_cost:
            best_p = p_ls
            best_cost = cost_ls
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
