"""Combined QAP solver: vectorized steepest-descent local search with Hungarian re-optimization and adaptive ILS."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _fast_swap_delta(a, D, i, j):
    """
    Compute delta cost for swapping positions i and j in permutation p.
    D is the current distance matrix b[p, p].
    Uses vectorized dot products for speed.
    """
    # S1 = a[i] dot D[j] - a[i] dot D[i]
    # S2 = a[:, i] dot D[:, j] - a[:, i] dot D[:, i]
    # S3 = a[j] dot D[i] - a[j] dot D[j]
    # S4 = a[:, j] dot D[:, i] - a[:, j] dot D[:, j]
    
    s1 = np.dot(a[i], D[j]) - np.dot(a[i], D[i])
    s2 = np.dot(a[:, i], D[:, j]) - np.dot(a[:, i], D[:, i])
    s3 = np.dot(a[j], D[i]) - np.dot(a[j], D[j])
    s4 = np.dot(a[:, j], D[:, i]) - np.dot(a[:, j], D[:, j])
    
    total = s1 + s2 + s3 + s4
    
    # Subtract k=i contribution (self-interaction terms)
    ki = (a[i, i] * (D[j, i] - D[i, i]) + 
          a[i, i] * (D[i, j] - D[i, i]) + 
          a[j, i] * (D[i, i] - D[j, i]) + 
          a[i, j] * (D[i, i] - D[i, j]))
    
    # Subtract k=j contribution
    kj = (a[j, j] * (D[i, j] - D[j, j]) + 
          a[j, j] * (D[j, i] - D[j, j]) + 
          a[i, j] * (D[j, j] - D[i, j]) + 
          a[j, i] * (D[j, j] - D[j, i]))
    
    return total - ki - kj


def _local_search(a, b, p, cost, deadline):
    """Steepest Descent Local Search with fast swap delta calculation."""
    n = len(p)
    p = p.copy()
    
    # Precompute symmetric A for potential optimizations
    D = b[np.ix_(p, p)].copy()
    
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        for i in range(n - 1):
            if time.time() > deadline:
                return p, cost
                
            # Find best swap partner for position i
            best_j = -1
            best_delta = 0
            
            for j in range(i + 1, n):
                delta = _fast_swap_delta(a, D, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0:
                # Perform the swap
                i, j = i, best_j
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                cost += best_delta
                improved = True
    
    return p, cost


def _hungarian_reopt(a, b, p):
    """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
    n = len(p)
    
    # B_p[i, j] = b[i, p[j]]
    B_p = b[:, p]
    
    # M1[i, j] = sum_k a[i, k] * b[j, p[k]] = (a @ B_p.T)[i, j]
    M1 = a @ B_p.T
    
    # Bp2[i, j] = b[p[i], j]
    Bp2 = b[p, :]
    
    # M2[i, j] = sum_k a[k, i] * b[p[k], j] = (a.T @ Bp2)[i, j]
    M2 = a.T @ Bp2
    
    C = M1 + M2
    
    row_ind, col_ind = linear_sum_assignment(C)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def _rank_matching_init(a, b):
    """Initial solution: rank matching based on row sums."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _perturb(p, k=4, rng=None):
    """Randomly swap k pairs of positions."""
    if rng is None:
        rng = random.Random()
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer under 120s limit
    
    rng = random.Random(42)
    
    # Initial solution: rank matching
    p = _rank_matching_init(a, b)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, best_p)
    
    # First local search
    current_p, current_cost = _local_search(a, b, p, best_cost, deadline)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost
    
    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() < deadline - 10:
        iteration += 1
        
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
        
        # Perturb from best solution
        p = _perturb(best_p, k=perturb_k, rng=rng)
        cost = _compute_cost(a, b, p)
        
        # Local search from perturbed
        p, cost = _local_search(a, b, p, cost, deadline)
        
        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = _hungarian_reopt(a, b, p)
            h_cost = _compute_cost(a, b, h_p)
            h_p, h_cost = _local_search(a, b, h_p, h_cost, deadline)
            if h_cost < cost:
                p, cost = h_p, h_cost
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
    
    # Final polish: one more local search pass on best solution
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = _local_search(a, b, final_p, final_cost, deadline)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()
