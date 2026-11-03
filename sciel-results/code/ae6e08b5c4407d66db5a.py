"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _eigenvalue_init(a, b):
    """
    Initialize permutation using eigenvalue decomposition.
    The idea: find eigenvectors of a and b, then match them to minimize
    the bilinear form. This captures structural correlations beyond row sums.
    """
    n = a.shape[0]
    
    # Use symmetric part for eigenvalue stability
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Get top few eigenvectors
        k = min(16, n)
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b_sym)
        
        # Use the most negative eigenvectors (lowest energy modes)
        # Reorder by eigenvalue (ascending)
        idx_a = np.argsort(w_a)[:k]
        idx_b = np.argsort(w_b)[:k]
        
        eig_a = v_a[:, idx_a]
        eig_b = v_b[:, idx_b]
        
        # Compute assignment matrix: for each facility i, location j,
        # score = sum over eigenvector pairs of (eig_a[i,m] * eig_b[j,m])
        # We want to minimize this, so we assign facility i to the location j
        # that minimizes the sum.
        
        # This is an assignment problem. Use Hungarian algorithm.
        from scipy.optimize import linear_sum_assignment
        
        # Build cost matrix
        cost_mat = eig_a @ eig_b.T  # shape (n, n)
        
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        p = np.zeros(n, dtype=np.int64)
        p[col_ind] = row_ind
        
        return p
    
    except Exception:
        # Fallback to random
        p = np.arange(n, dtype=np.int64)
        np.random.shuffle(p)
        return p

def _pairwise_interaction_init(a, b):
    """
    Initialize based on pairwise interaction strengths.
    Facilities with high mutual flow should be placed at locations with low mutual distance.
    """
    n = a.shape[0]
    
    # Compute flow interaction strength for each pair
    flow_strength = np.abs(a) + np.abs(a.T)
    flow_strength = np.sum(flow_strength, axis=1)  # per-facility total interaction
    
    # Compute distance interaction strength for each location
    dist_strength = np.abs(b) + np.abs(b.T)
    dist_strength = np.sum(dist_strength, axis=1)  # per-location total distance
    
    # Sort facilities by flow strength (descending) and locations by distance strength (ascending)
    # This is a heuristic: high-interaction facilities go to low-distance locations
    fac_order = np.argsort(flow_strength)[::-1]
    loc_order = np.argsort(dist_strength)
    
    p = np.zeros(n, dtype=np.int64)
    for i, fac in enumerate(fac_order):
        p[fac] = loc_order[i]
    
    return p

def _hybrid_init(a, b, n):
    """
    Combine multiple initialization strategies and pick the best.
    """
    inits = []
    
    # Eigenvalue-based
    p_eig = _eigenvalue_init(a, b)
    inits.append(p_eig)
    
    # Pairwise interaction-based
    p_pair = _pairwise_interaction_init(a, b)
    inits.append(p_pair)
    
    # Random
    p_rand = np.arange(n, dtype=np.int64)
    np.random.shuffle(p_rand)
    inits.append(p_rand)
    
    # Pick the one with lowest initial cost
    best_p = None
    best_cost = np.inf
    for p in inits:
        cost = _compute_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p

def _fast_delta(a, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    D is the current distance matrix b[p, p].
    """
    s1 = np.dot(a[i], D[j]) - np.dot(a[i], D[i])
    s2 = np.dot(a[:, i], D[:, j]) - np.dot(a[:, i], D[:, i])
    s3 = np.dot(a[j], D[i]) - np.dot(a[j], D[j])
    s4 = np.dot(a[:, j], D[:, i]) - np.dot(a[:, j], D[:, j])
    
    total = s1 + s2 + s3 + s4
    
    ki = a[i,i]*(D[j,i]-D[i,i]) + a[i,i]*(D[i,j]-D[i,i]) + a[j,i]*(D[i,i]-D[j,i]) + a[i,j]*(D[i,i]-D[i,j])
    
    kj = a[i,j]*(D[j,j]-D[i,j]) + a[j,i]*(D[j,j]-D[j,i]) + a[j,j]*(D[i,j]-D[j,j]) + a[j,j]*(D[j,i]-D[j,j])
    
    total -= ki + kj
    
    direct = a[i,j]*D[j,i] + a[j,i]*D[i,j] - a[i,j]*D[i,j] - a[j,i]*D[j,i]
    
    total += direct
    
    return total

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _fast_delta(a, D, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    
                    D[i], D[j] = D[j].copy(), D[i].copy()
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
    
    # Use hybrid initialization
    p_init = _hybrid_init(a, b, n)
    
    # First local search on hybrid init
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
