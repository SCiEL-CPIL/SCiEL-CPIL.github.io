"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # p is a numpy int array of shape (n,)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_swap(a, b, p, i, j):
    """Compute the change in QAP cost if we swap positions i and j in permutation p.
    
    p: current permutation (facility -> location mapping is p[facility] = location)
    Swapping means facility i goes to location p[j] and facility j goes to location p[i].
    We return delta = new_cost - old_cost.
    """
    li = p[i]
    lj = p[j]
    
    # The cost contribution involving facilities i and j with all other facilities k,
    # plus the i-j and j-i terms, changes when we swap their locations.
    
    # Old contributions:
    # For all k != i, j: a[i,k]*b[li, p[k]] + a[k,i]*b[p[k], li] + a[j,k]*b[lj, p[k]] + a[k,j]*b[p[k], lj]
    # Plus: a[i,j]*b[li, lj] + a[j,i]*b[lj, li]
    
    # New contributions (after swap, i->lj, j->li):
    # For all k != i, j: a[i,k]*b[lj, p[k]] + a[k,i]*b[p[k], lj] + a[j,k]*b[li, p[k]] + a[k,j]*b[p[k], li]
    # Plus: a[i,j]*b[lj, li] + a[j,i]*b[li, lj]
    
    # Delta = new - old
    
    # Let's compute efficiently using vectorized ops.
    # a_i = a[i, :], a_j = a[j, :]
    # b_row_li = b[li, :], b_row_lj = b[lj, :]
    # b_col_li = b[:, li], b_col_lj = b[:, lj]
    
    # For each k (k != i, j), the change in terms involving i and k:
    #   a[i,k] * (b[lj, p[k]] - b[li, p[k]]) + a[k,i] * (b[p[k], lj] - b[p[k], li])
    # And terms involving j and k:
    #   a[j,k] * (b[li, p[k]] - b[lj, p[k]]) + a[k,j] * (b[p[k], li] - b[p[k], lj])
    
    # Combined for facility i terms: a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
    # For facility j terms: a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
    
    # Note: b[lj,p[k]]-b[li,p[k]] = -(b[li,p[k]]-b[lj,p[k]])
    # So combined: (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    # But we must exclude k=i and k=j from the sum over k.
    
    ai = a[i, :]
    aj = a[j, :]
    aiT = a[:, i]
    ajT = a[:, j]
    
    b_li = b[li, :]  # row li
    b_lj = b[lj, :]  # row lj
    b_liT = b[:, li]  # col li
    b_ljT = b[:, lj]  # col lj
    
    # For all k:
    # term1[k] = (ai[k] - aj[k]) * (b_lj[p[k]] - b_li[p[k]])
    # term2[k] = (aiT[k] - ajT[k]) * (b_ljT[p[k]] - b_liT[p[k]])
    
    # But we need to be careful: when k=i or k=j, the p[k] values are li or lj themselves,
    # and those terms are handled separately.
    
    # Let's compute for all k first, then subtract k=i and k=j contributions.
    
    pk = p  # array of locations
    
    diff_b_row = b_lj[pk] - b_li[pk]  # shape (n,)
    diff_b_col = b_ljT[pk] - b_liT[pk]  # shape (n,)
    
    term1 = (ai - aj) * diff_b_row
    term2 = (aiT - ajT) * diff_b_col
    
    total_delta = np.sum(term1) + np.sum(term2)
    
    # Now subtract the k=i and k=j contributions from the sum above,
    # because those were included but we handle them separately.
    
    # When k=i: p[k] = p[i] = li
    # term1[i] = (ai[i] - aj[i]) * (b_lj[li] - b_li[li])
    # term2[i] = (aiT[i] - ajT[i]) * (b_ljT[li] - b_liT[li])
    # ai[i] = a[i,i], aj[i] = a[j,i]
    # aiT[i] = a[i,i], ajT[i] = a[j,i]
    
    # When k=j: p[k] = p[j] = lj
    # term1[j] = (ai[j] - aj[j]) * (b_lj[lj] - b_li[lj])
    # term2[j] = (aiT[j] - ajT[j]) * (b_ljT[lj] - b_liT[lj])
    # ai[j] = a[i,j], aj[j] = a[j,j]
    # aiT[j] = a[j,i], ajT[j] = a[j,j]
    
    # Remove k=i and k=j from total:
    total_delta -= term1[i] + term2[i] + term1[j] + term2[j]
    
    # Now add the correct contributions for k=i and k=j:
    # When k=i (facility i itself), after swap it's at lj:
    # The self-term for i: a[i,i]*b[lj, lj] (new) vs a[i,i]*b[li, li] (old)
    # When k=j (facility j itself), after swap it's at li:
    # a[j,j]*b[li, li] (new) vs a[j,j]*b[lj, lj] (old)
    
    # The i-j cross terms:
    # Old: a[i,j]*b[li, lj] + a[j,i]*b[lj, li]
    # New: a[i,j]*b[lj, li] + a[j,i]*b[li, lj]
    
    # So the self and cross terms delta:
    self_delta = a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
    cross_delta = a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
    
    total_delta += self_delta + cross_delta
    
    return int(total_delta)


def _initial_permutation(a, b):
    """Generate an initial permutation using a combination of rank-matching and a few random restarts."""
    n = len(a)
    best_p = None
    best_cost = np.inf
    
    for trial in range(5):
        if trial == 0:
            # Rank-matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            # Random with some structure
            p = np.random.permutation(n).astype(np.int64)
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin
    
    # Get initial permutation
    p, best_cost = _initial_permutation(a, b)
    current_cost = best_cost
    best_p = p.copy()
    
    # Simulated Annealing
    T = 1000.0
    T_min = 1.0
    alpha = 0.995
    
    while T > T_min and time.time() < deadline:
        # Choose a random swap (non-adjacent preferred for large steps)
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        # Bias towards larger jumps
        if abs(i - j) < n // 4 and np.random.random() < 0.5:
            j = np.random.randint(n)
            if i == j:
                continue
        
        delta = _delta_swap(a, b, p, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
