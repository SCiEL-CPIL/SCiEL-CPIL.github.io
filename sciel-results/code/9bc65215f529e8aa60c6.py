"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

DEADLINE_FRACTION = 0.92  # safety margin


def _compute_cost(a, b, p):
    """Exact integer cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_swap_delta(a, b, p, i, j):
    """
    Delta in cost if we swap facilities i and j (i.e., p[i] and p[j] are exchanged).
    We want to minimize cost, so a negative delta is an improvement.
    """
    pi = p[i]
    pj = p[j]
    
    # Indices not in S
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = np.where(mask)[0]
    
    # S x S terms
    delta = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
             a[i, j] * (b[pj, pi] - b[pi, pj]) +
             a[j, i] * (b[pi, pj] - b[pj, pi]) +
             a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    # S x others
    delta += np.sum(a[i, others] * (b[pj, p[others]] - b[pi, p[others]]))
    delta += np.sum(a[j, others] * (b[pi, p[others]] - b[pj, p[others]]))
    
    # others x S
    delta += np.sum(a[others, i] * (b[p[others], pi] - b[p[others], pj]))
    delta += np.sum(a[others, j] * (b[p[others], pj] - b[p[others], pi]))
    
    return int(delta)


def _local_search(a, b, p, deadline):
    """
    First-improvement swap-based local search.
    p is a numpy array (int64). Returns the improved p (numpy array) and the cost.
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i_idx in range(n):
            if time.time() >= deadline:
                break
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                if time.time() >= deadline:
                    break
                j = indices[j_idx]
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break
    
    return p, current_cost


def _hungarian_init(a, b):
    """
    Construct cost matrix C where C[i,j] = -sum_k a[i,k] * b[j,k].
    This is the linear relaxation: if facility i is at location j, the contribution
    to the objective from terms where i is the 'first' index is sum_k a[i,k] * b[j, p[k]].
    But since p is unknown, we approximate by assuming the 'second' location is also j
    in a decoupled way? No, the standard linear relaxation for QAP Q = sum a[i,j] b[p[i],p[j]]
    is not simply separable like this unless we fix one part.
    
    However, the directive suggests: C_ij = -sum_k a[i,k] b[j,k].
    This corresponds to minimizing sum_i,j a[i,j] b[p[i], p[j]].
    If we approximate p[j] ~ j (identity) for the second index in the sum for the relaxation,
    it's not quite right.
    
    Let's look at the structure:
    Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    = sum_i ( sum_j a[i,j] b[p[i], p[j]] )
    
    If we decouple p[i] and p[j], we can't just sum them independently.
    However, a common heuristic initialization is to match facilities to locations based on
    'strength'.
    
    The specific suggestion: C_ij = -sum_k a[i,k] b[j,k].
    Let's check dimensions. a is n x n, b is n x n.
    sum_k a[i,k] b[j,k] is a matrix of size n x n.
    Let's call this M.
    Then we solve min sum_{i,j} M[i, p[i]]? No, linear_sum_assignment solves min sum_{i} C[i, p[i]].
    So if we set C = M, we are minimizing sum_i (sum_k a[i,k] b[p[i], k]).
    This is NOT the QAP cost.
    
    The QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    If we fix p[j] = j, then cost is sum_{i,j} a[i,j] b[p[i], j].
    Then for each i, the cost contribution is sum_j a[i,j] b[p[i], j].
    This is exactly sum_k a[i,k] b[p[i], k] (renaming j to k).
    So if we assume the 'target' locations are fixed to identity (p[j]=j), then the optimal assignment
    for p[i] is given by the Hungarian algorithm on C where C[i, l] = sum_k a[i,k] b[l, k].
    
    This is a valid heuristic: it optimizes the cost assuming the other facilities are at their
    'natural' positions (identity), which is a reasonable first-order approximation if the
    matrices have strong diagonal structure or if we just want a good starting point.
    
    Let's implement this.
    """
    # C[i, l] = sum_k a[i, k] * b[l, k]
    # This is (a @ b.T) ?
    # a[i, k] * b[l, k] -> sum over k.
    # Matrix multiplication: (a @ b.T)[i, l] = sum_k a[i, k] * b.T[k, l] = sum_k a[i, k] * b[l, k].
    # Yes.
    C = a @ b.T
    # We want to minimize sum_i C[i, p[i]].
    # linear_sum_assignment minimizes the sum of C[i, p[i]].
    # So we pass C directly.
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(len(a), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    n = len(a)
    
    # 1. Generate initial solution using Hungarian algorithm on linear relaxation
    p = _hungarian_init(a, b)
    
    # 2. Local Search
    p, cost = _local_search(a, b, p, deadline)
    
    return p.tolist()
# EVOLVE-BLOCK-END
