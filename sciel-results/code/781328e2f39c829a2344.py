import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Calculate the QAP cost for a given permutation."""
    # Cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)

def _two_opt(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Perform 2-opt local search.
    Uses vectorized delta calculation for speed.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute row sums of A to help? 
    # Delta for swapping i and j:
    # Old terms involving i or j:
    # sum_{k != i, j} a[i,k] b[p[i], p[k]] + a[k,i] b[p[k], p[i]]
    # sum_{k != i, j} a[j,k] b[p[j], p[k]] + a[k,j] b[p[k], p[j]]
    # a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]]
    
    # New terms:
    # sum_{k != i, j} a[i,k] b[p[j], p[k]] + a[k,i] b[p[k], p[j]]
    # sum_{k != i, j} a[j,k] b[p[i], p[k]] + a[k,j] b[p[k], p[i]]
    # a[i,j] b[p[j], p[i]] + a[j,i] b[p[i], p[j]]
    
    # Delta = New - Old
    
    # We can compute this efficiently.
    # Let's just use a simple loop with some vectorization if possible, 
    # but for n=256, a pure Python loop over n^2 pairs might be too slow 
    # if delta calculation is not O(1).
    # However, 256^2 = 65536 pairs. If each takes ~10us, that's 0.6s per pass.
    # We can do ~100 passes in 60s.
    
    # To speed up, we can precompute:
    # S_i(k) = a[i,k] * b[p[i], p[k]] + a[k,i] * b[p[k], p[i]]
    # This is hard to update incrementally for all k when p[i] changes.
    
    # Let's try a simpler 2-opt that recalculates cost for candidates?
    # No, that's O(n^2) per swap.
    
    # Let's use the standard delta formula with precomputed vectors.
    # For a fixed i, we can compute delta for all j > i in a vectorized way?
    # Delta(i, j) depends on p[i] and p[j].
    
    # Given the complexity, let's implement a randomized 2-opt 
    # that checks a subset of swaps per iteration to keep it fast.
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time_limit and start_time and time.time() > time_limit:
            break
        improved = False
        iters += 1
        
        # Randomly shuffle order of checking to avoid bias
        indices = np.random.permutation(n)
        
        for i_idx in range(n):
            i = indices[i_idx]
            if time_limit and start_time and time.time() > time_limit:
                return p, current_cost
            
            # Compute delta for swapping i with all j > i (in original index space)
            # To keep it simple and fast, we'll just check a random subset of j's
            # or all j's but with a fast approximate check?
            
            # Let's do all j > i
            # This is O(n^2) per full pass.
            
            # Precompute terms for i
            # pi = p[i], pj = p[j]
            # Term_i_old_k = a[i,k] * b[pi, p[k]] + a[k,i] * b[p[k], pi]
            # Term_j_old_k = a[j,k] * b[pj, p[k]] + a[k,j] * b[p[k], pj]
            
            # This is getting complicated to vectorize fully without large memory.
            # For n=256, let's just do a simple loop but optimize the inner calculation.
            
            pi = p[i]
            # Vectorized calculation of cost contribution of i to all k
            # cost_i_k = a[i, :] * b[pi, p] + a[:, i] * b[p, pi]
            # Note: a[i, i] and a[:, i] include k=i term which we handle separately or exclude.
            
            cost_i_old = a[i, :] * b[pi, p] + a[:, i] * b[p, pi]
            
            best_j = -1
            best_delta = 0
            
            for j in range(n):
                if j == i:
                    continue
                pj = p[j]
                
                # Cost contribution of j to all k
                # We can reuse some calculations if we were looping j first, but here i is fixed.
                # Let's just compute delta directly.
                
                # Old cost terms involving i or j:
                # C_old = sum_{k!=i,j} a[i,k]b[pi,pk] + a[k,i]b[pk,pi] + a[j,k]b[pj,pk] + a[k,j]b[pk,pj] + a[i,j]b[pi,pj] + a[j,i]b[pj,pi]
                
                # New cost terms:
                # C_new = sum_{k!=i,j} a[i,k]b[pj,pk] + a[k,i]b[pk,pj] + a[j,k]b[pi,pk] + a[k,j]b[pk,pi] + a[i,j]b[pj,pi] + a[j,i]b[pi,pj]
                
                # Delta = C_new - C_old
                
                # Vectorized for all k:
                # diff_k = (a[i,k] * (b[pj,pk] - b[pi,pk]) + a[k,i] * (b[pk,pj] - b[pk,pi]) +
                #           a[j,k] * (b[pi,pk] - b[pj,pk]) + a[k,j] * (b[pk,pi] - b[pk,pj]))
                
                # We can compute this for all k at once.
                # b_pj_pk = b[pj, p]
                # b_pi_pk = b[pi, p]
                # b_pk_pj = b[p, pj]
                # b_pk_pi = b[p, pi]
                
                # delta_k = a[i, :] * (b[pj, p] - b[pi, p]) + a[:, i] * (b[p, pj] - b[p, pi]) + \
                #           a[j, :] * (b[pi, p] - b[pj, p]) + a[:, j] * (b[p, pi] - b[p, pj])
                
                # Sum over k != i, j
                # delta_sum = np.sum(delta_k) - delta_k[i] - delta_k[j]
                
                # Plus the direct i-j terms:
                # old_ij = a[i,j] * b[pi, pj] + a[j,i] * b[pj, pi]
                # new_ij = a[i,j] * b[pj, pi] + a[j,i] * b[pi, pj]
                # delta_ij = new_ij - old_ij
                
                # Total delta = delta_sum + delta_ij
                
                # This is O(n) per j. Total O(n^2) per i. O(n^3) per pass.
                # 256^3 = 16.7M ops. In NumPy, this is fast.
                
                # Let's implement this vectorized version.
                
                b_pj_p = b[pj, :]
                b_pi_p = b[pi, :]
                b_p_pj = b[:, pj]
                b_p_pi = b[:, pi]
                
                # Note: b is symmetric? QAPLIB instances usually have symmetric A and B.
                # If B is symmetric, b[pj, p] == b[p, pj].
                # Let's assume symmetry for speed if possible, but keep general.
                
                delta_k = a[i, :] * (b_pj_p - b_pi_p) + a[:, i] * (b_p_pj - b_p_pi) + \
                          a[j, :] * (b_pi_p - b_pj_p) + a[:, j] * (b_p_pi - b_p_pj)
                
                # Exclude i and j from sum
                delta_sum = np.sum(delta_k)
                delta_sum -= delta_k[i]
                delta_sum -= delta_k[j]
                
                old_ij = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
                new_ij = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
                delta_ij = new_ij - old_ij
                
                total_delta = delta_sum + delta_ij
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_j = j
            
            if best_j != -1:
                # Perform swap
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_delta
                improved = True
    
    return p, current_cost

def solve(a, b):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = a.shape[0]
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # 1. Initialization: Random or Greedy
    # Let's use a simple greedy or random start
    p = np.random.permutation(n)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Initial 2-opt to get a good starting point
    best_p, best_cost = _two_opt(a, b, best_p, max_iters=20, time_limit=deadline, start_time=start_time)
    
    # Main Loop: Iterated Local Search with Hungarian-based Perturbation
    noise_scale = 1000.0  # Tuning parameter
    base_noise = 1000.0
    
    while time.time() < deadline - 10.0:  # Keep 10s for final polish
        # Compute linearized cost matrix (Gradient)
        # G = A P B^T + A^T P B
        # Create permutation matrix P (sparse or dense?)
        # Dense P is 256x256, which is fine.
        P = np.zeros((n, n), dtype=np.float64)
        P[np.arange(n), best_p] = 1.0
        
        # Compute Gradient
        G = a @ P @ b.T + a.T @ P @ b
        
        # Add noise to escape local optima
        # The noise should be random but structured? 
        # Standard Gaussian noise works well for diversity.
        noise = np.random.randn(n, n) * noise_scale
        G_noisy = G + noise
        
        # Solve Assignment Problem
        # We want to minimize cost, so we minimize G_noisy.
        # linear_sum_assignment minimizes sum(cost[i, j])
        row_ind, col_ind = linear_sum_assignment(G_noisy)
        p_new = col_ind
        
        # Evaluate new solution
        cost_new = _qap_cost(a, b, p_new)
        
        # Polish with 2-opt
        p_new, cost_new = _two_opt(a, b, p_new, max_iters=10, time_limit=deadline, start_time=start_time)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new
            # Reduce noise scale slightly as we find better solutions?
            # Or keep it constant to maintain diversity.
            # Let's keep it constant for now.
        else:
            # If no improvement, we might want to increase noise to explore more
            # But if we are close to deadline, we should stop exploring and just polish.
            pass
    
    # Final Polish: Run 2-opt on best solution with more iterations
    if time.time() < deadline - 2.0:
        best_p, best_cost = _two_opt(a, b, best_p, max_iters=50, time_limit=deadline, start_time=start_time)
        
    return best_p.tolist()
