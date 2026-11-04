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
    coeff_i = a_sum[i, i] - a_sum[j, i]
    term_i = coeff_i * (D[j, i] - D[i, i])
    
    coeff_j = a_sum[i, j] - a_sum[j, j]
    term_j = coeff_j * (D[j, j] - D[i, j])
    
    diag_term = (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
    cross_term = (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
    
    return S - term_i - term_j + diag_term + cross_term

def _vectorized_deltas(a, a_sum, D, i, n):
    """
    Compute delta cost for swapping position i with all other positions j.
    Returns array of deltas for j in range(n), excluding i.
    """
    # For each j, delta = S - T_i - T_j + diag_term + cross_term
    # S_j = (a_sum[i] - a_sum[j]) dot (D[j] - D[i])
    # = a_sum[i] dot D[j] - a_sum[i] dot D[i] - a_sum[j] dot D[j] + a_sum[j] dot D[i]
    
    # Precompute for fixed i:
    # a_sum[i] dot D[i] is constant
    ai_dot_Di = np.dot(a_sum[i], D[i])
    
    # a_sum[i] dot D[j] for all j: a_sum[i] @ D
    ai_dot_D_all = a_sum[i] @ D  # shape (n,)
    
    # a_sum[j] dot D[j] for all j: sum of a_sum * D along axis 1
    aj_dot_Dj = np.sum(a_sum * D, axis=1)  # shape (n,)
    
    # a_sum[j] dot D[i] for all j: a_sum @ D[i]
    aj_dot_Di = a_sum @ D[i]  # shape (n,)
    
    # S_j = ai_dot_D_all[j] - ai_dot_Di - aj_dot_Dj[j] + aj_dot_Di[j]
    S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
    
    # T_i_j = (a_sum[i,i] - a_sum[j,i]) * (D[j,i] - D[i,i])
    # = (a_sum[i,i] - a_sum[:,i]) * (D[:,i] - D[i,i])
    coeff_i_all = a_sum[i, i] - a_sum[:, i]
    term_i_all = coeff_i_all * (D[:, i] - D[i, i])
    
    # T_j_i = (a_sum[i,j] - a_sum[j,j]) * (D[j,j] - D[i,j])
    # = (a_sum[i,:] - np.diag(a_sum)) * (np.diag(D) - D[i,:])
    coeff_j_all = a_sum[i, :] - np.diag(a_sum)
    term_j_all = coeff_j_all * (np.diag(D) - D[i, :])
    
    # diag_term_j = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
    diag_term_all = (a[i, i] - np.diag(a)) * (np.diag(D) - D[i, i])
    
    # cross_term_j = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
    cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
    
    deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
    deltas[i] = 0  # Exclude self-swap
    return deltas

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Local search with vectorized delta computation for each position.
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
            # Compute all deltas for swapping position i with any j
            deltas = _vectorized_deltas(a, a_sum, D, i, n)
            # Find best improving move
            j = np.argmin(deltas)
            if deltas[j] < 0:
                # Perform swap
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += deltas[j]
                improved = True
    
    return p, current_cost

def _rank_match_init(a, b):
    """Initialize by matching facility ranks to location ranks based on row sums."""
    n = len(a)
    # Sort facilities by their total flow (row sum of a)
    fac_order = np.argsort(a.sum(axis=1))
    # Sort locations by their total distance (row sum of b)
    loc_order = np.argsort(b.sum(axis=1))
    # Assign facility at rank k to location at rank k
    p = np.empty(n, dtype=int)
    p[fac_order] = loc_order
    return p

def _perturb(p, k=5, rng=None):
    """Randomly swap k pairs of positions."""
    if rng is None:
        rng = np.random.default_rng()
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    
    # Try multiple initialization strategies
    inits = []
    
    # 1. Rank-matching initialization
    p_rank = _rank_match_init(a, b)
    inits.append(p_rank)
    
    # 2. Random initialization
    p_rand = rng.permutation(n)
    inits.append(p_rand)
    
    # 3. Another random initialization with different seed
    p_rand2 = rng.permutation(n)
    inits.append(p_rand2)
    
    # Run local search on each initialization
    for p_init in inits:
        if time.time() - start_time > time_budget - 5:
            break
        p_ls, cost_ls = _local_search_vectorized(a, b, p_init, time_budget - 5, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Perturb
        p_pert = _perturb(best_p, k=5, rng=rng)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
