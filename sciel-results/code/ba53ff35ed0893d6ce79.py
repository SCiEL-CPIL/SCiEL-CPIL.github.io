"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact integer QAP cost for permutation p (facility i at location p[i])."""
    n = len(p)
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Use matrix multiplication for speed:
    # b_p = b[np.ix_(p, p)]
    # cost = np.sum(a * b_p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute change in cost if we swap facilities i and j in permutation p.
    p is a numpy array. Returns delta = new_cost - old_cost.
    """
    li = p[i]
    lj = p[j]
    if li == lj:
        return 0
    
    # Cost contribution involving i and j before swap:
    # For pair (i,j): a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
    # For pair (i,k) k!=i,j: a[i,k]*b[li,p[k]] + a[k,i]*b[p[k],li]
    # For pair (j,k) k!=i,j: a[j,k]*b[lj,p[k]] + a[k,j]*b[p[k],lj]
    
    # After swap:
    # For pair (i,j): a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
    # For pair (i,k) k!=i,j: a[i,k]*b[lj,p[k]] + a[k,i]*b[p[k],lj]
    # For pair (j,k) k!=i,j: a[j,k]*b[li,p[k]] + a[k,j]*b[p[k],li]
    
    # Delta for pair (i,j):
    # a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
    # = (a[i,j]-a[j,i])*(b[lj,li]-b[li,lj])
    
    # For pairs with k:
    # Sum over k != i,j of:
    # a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[k,i]*(b[p[k],lj]-b[p[k],li])
    # + a[j,k]*(b[li,p[k]]-b[lj,p[k]]) + a[k,j]*(b[p[k],li]-b[p[k],lj])
    # = Sum over k != i,j of:
    # (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    # + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    n = len(p)
    # Efficient computation using vectorized operations
    # We need sum over all k != i,j
    
    # Let's compute for all k first, then subtract k=i and k=j
    # Term 1: (a[i,k]-a[j,k]) * (b[lj,p[k]]-b[li,p[k]])
    # Term 2: (a[k,i]-a[k,j]) * (b[p[k],lj]-b[p[k],li])
    
    # Vectorized over k:
    diff_a_ij = a[i, :] - a[j, :]  # shape (n,)
    diff_b_lj_li = b[lj, p] - b[li, p]  # shape (n,)
    term1_all = diff_a_ij * diff_b_lj_li  # shape (n,)
    
    diff_a_ki_kj = a[:, i] - a[:, j]  # shape (n,)
    diff_b_pk_lj_li = b[p, lj] - b[p, li]  # shape (n,)
    term2_all = diff_a_ki_kj * diff_b_pk_lj_li  # shape (n,)
    
    # Sum over all k
    total = np.sum(term1_all) + np.sum(term2_all)
    
    # Subtract k=i and k=j
    # For k=i:
    term1_i = diff_a_ij[i] * diff_b_lj_li[i]
    term2_i = diff_a_ki_kj[i] * diff_b_pk_lj_li[i]
    
    # For k=j:
    term1_j = diff_a_ij[j] * diff_b_lj_li[j]
    term2_j = diff_a_ki_kj[j] * diff_b_pk_lj_li[j]
    
    sum_k = total - term1_i - term2_i - term1_j - term2_j
    
    # Delta for pair (i,j) itself:
    delta_ij = (a[i, j] - a[j, i]) * (b[lj, li] - b[li, lj])
    
    return int(delta_ij + sum_k)


def _simulated_annealing(a, b, p, t_start, t_end, iters_per_temp, n_temps, deadline):
    """Simulated annealing with swap moves."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    p = p.copy()
    
    for t_idx in range(n_temps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        if n_temps > 1:
            temp = t_start * (t_end / t_start) ** (t_idx / max(n_temps - 1, 1))
        else:
            temp = t_start
        
        for _ in range(iters_per_temp):
            if time.time() > deadline:
                return best_p, best_cost
            
            # Random swap
            i, j = np.random.choice(n, size=2, replace=False)
            if i == j:
                continue
            
            delta = _delta_swap(a, b, p, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                # Accept
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost


def _tabu_search(a, b, p, t_start, t_end, iters_per_temp, n_temps, tabu_len, deadline):
    """Tabu search with swap moves."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    p = p.copy()
    
    # Tabu list: store (i, j, iteration)
    tabu_list = []
    tabu_iter = 0
    
    for t_idx in range(n_temps):
        if time.time() > deadline:
            break
        
        for _ in range(iters_per_temp):
            if time.time() > deadline:
                return best_p, best_cost
            
            # Find best non-tabu swap
            best_delta = float('inf')
            best_i, best_j = -1, -1
            best_tabu = False
            
            # Sample a subset of swaps for efficiency
            n_samples = min(n * 5, 500)
            for _ in range(n_samples):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                # Check if tabu
                is_tabu = False
                for ti, tj, titer in tabu_list:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        if tabu_iter - titer < tabu_len:
                            is_tabu = True
                            break
                
                delta = _delta_swap(a, b, p, i, j)
                
                # Aspiration: accept if it improves best even if tabu
                if delta < best_cost:
                    best_tabu = False
                else:
                    best_tabu = is_tabu
                
                if not best_tabu and delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
                    best_tabu = False
            
            if best_i == -1:
                # No good move found, try a random one
                i, j = np.random.choice(n, size=2, replace=False)
                best_i, best_j = i, j
                best_delta = _delta_swap(a, b, p, i, j)
            
            # Perform swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            # Add to tabu list
            tabu_list.append((best_i, best_j, tabu_iter))
            tabu_iter += 1
            
            # Clean old tabu entries
            while tabu_list and tabu_iter - tabu_list[0][2] >= tabu_len:
                tabu_list.pop(0)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with Hungarian algorithm on a heuristic cost matrix
    # We want to match facilities to locations to minimize sum a[i,j]*b[p[i],p[j]]
    # This is not directly a linear assignment, so we use a surrogate.
    # One approach: use the row sums of a and b as a proxy.
    
    # Better: use a greedy construction followed by local search
    # Or use the rank-matching idea from the seed but improve it
    
    # Let's use a combination: start with rank-matching, then improve with SA/TS
    
    # Rank-matching initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # Also try a random initialization
    p_rand = np.random.permutation(n)
    
    # Evaluate both
    cost_init = _qap_cost(a, b, p_init)
    cost_rand = _qap_cost(a, b, p_rand)
    
    if cost_rand < cost_init:
        p_best = p_rand.copy()
        cost_best = cost_rand
    else:
        p_best = p_init.copy()
        cost_best = cost_init
    
    # Run simulated annealing
    if time.time() < deadline - 30:
        p_sa, cost_sa = _simulated_annealing(
            a, b, p_best,
            t_start=1000.0,
            t_end=0.01,
            iters_per_temp=100,
            n_temps=50,
            deadline=deadline - 10
        )
        if cost_sa < cost_best:
            p_best = p_sa
            cost_best = cost_sa
    
    # Run tabu search on the best result
    if time.time() < deadline - 20:
        p_ts, cost_ts = _tabu_search(
            a, b, p_best,
            t_start=1000.0,
            t_end=0.01,
            iters_per_temp=100,
            n_temps=30,
            tabu_len=10,
            deadline=deadline - 5
        )
        if cost_ts < cost_best:
            p_best = p_ts
            cost_best = cost_ts
    
    # Final polish: greedy local search with 2-opt style swaps
    if time.time() < deadline - 5:
        p_polish = p_best.copy()
        polish_cost = cost_best
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 3:
                        break
                    delta = _delta_swap(a, b, p_polish, i, j)
                    if delta < 0:
                        p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
                        polish_cost += delta
                        improved = True
        
        if polish_cost < cost_best:
            p_best = p_polish
            cost_best = polish_cost
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
