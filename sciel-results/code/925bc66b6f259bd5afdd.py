"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    We want to swap p[i] and p[j].
    
    Let x = p[i], y = p[j].
    Old contribution involving i or j:
      sum_k a[i,k] * b[x, p[k]] + sum_k a[k,i] * b[p[k], x] + sum_k a[j,k] * b[y, p[k]] + sum_k a[k,j] * b[p[k], y]
      - 2 * a[i,j] * b[x,y]  (because the term a[i,j]*b[x,y] and a[j,i]*b[y,x] are counted in the sums above,
        but in the total cost they appear as a[i,j]*b[x,y] + a[j,i]*b[y,x]. Since a and b are symmetric,
        a[i,j]=a[j,i], b[x,y]=b[y,x], so total is 2*a[i,j]*b[x,y].)
      Wait, let's be more careful.
    
    Total cost = sum_{m,n} a[m,n] * b[p[m], p[n]]
    
    When we swap p[i] and p[j], let x = p[i], y = p[j].
    The terms that change are those where m=i or n=i or m=j or n=j.
    
    Let's define:
    S_i = sum_{k != i, j} a[i,k] * b[x, p[k]] + sum_{k != i, j} a[k,i] * b[p[k], x]
    S_j = sum_{k != i, j} a[j,k] * b[y, p[k]] + sum_{k != i, j} a[k,j] * b[p[k], y]
    Cross = a[i,j] * b[x,y] + a[j,i] * b[y,x]
    
    Old contribution = S_i + S_j + Cross
    Note: S_i includes terms with b[x, p[k]] and b[p[k], x]. Since b is symmetric, b[p[k],x] = b[x,p[k]].
    So S_i = 2 * sum_{k != i,j} a[i,k] * b[x, p[k]] if a is symmetric? No, a[i,k] != a[k,i] necessarily.
    Actually, S_i = sum_{k != i,j} a[i,k] * b[x, p[k]] + sum_{k != i,j} a[k,i] * b[p[k], x]
    = sum_{k != i,j} (a[i,k] + a[k,i]) * b[x, p[k]]  (since b is symmetric)
    
    After swap, p[i]=y, p[j]=x.
    New S_i' = sum_{k != i,j} a[i,k] * b[y, p[k]] + sum_{k != i,j} a[k,i] * b[p[k], y]
             = sum_{k != i,j} (a[i,k] + a[k,i]) * b[y, p[k]]
    New S_j' = sum_{k != i,j} a[j,k] * b[x, p[k]] + sum_{k != i,j} a[k,j] * b[p[k], x]
             = sum_{k != i,j} (a[j,k] + a[k,j]) * b[x, p[k]]
    New Cross' = a[i,j] * b[y,x] + a[j,i] * b[x,y] = 2 * a[i,j] * b[x,y] (same as Cross if a,b symmetric)
    
    So delta = (S_i' + S_j' + Cross') - (S_i + S_j + Cross)
             = S_i' - S_i + S_j' - S_j
             = sum_{k != i,j} (a[i,k] + a[k,i]) * (b[y, p[k]] - b[x, p[k]])
             + sum_{k != i,j} (a[j,k] + a[k,j]) * (b[x, p[k]] - b[y, p[k]])
             = sum_{k != i,j} [ (a[i,k] + a[k,i]) - (a[j,k] + a[k,j]) ] * (b[y, p[k]] - b[x, p[k]])
    
    This is O(n) per swap evaluation.
    """
    x = p[i]
    y = p[j]
    # Precomputed: a_sym[k] = a[i,k] + a[k,i] for all k, similarly for j
    # But we can just compute on the fly for simplicity, or precompute row sums.
    
    # Let's precompute a_row_sum_sym[k] = a[k,:] + a[:,k] ... no.
    # Actually, let's precompute for each facility m, the vector v_m[k] = a[m,k] + a[k,m]
    # Then delta = sum_{k != i,j} (v_i[k] - v_j[k]) * (b[y, p[k]] - b[x, p[k]])
    
    # For efficiency, we should precompute v = a + a.T
    # And then use that.
    
    # But in this function, we don't have access to precomputed data unless we pass it.
    # Let's just do the O(n) computation directly for now.
    
    # To make it faster, we can precompute:
    # diff_a = (a[i,:] + a[:,i]) - (a[j,:] + a[:,j])
    # diff_b = b[y,:] - b[x,:]
    # Then delta = sum_{k != i,j} diff_a[k] * diff_b[p[k]]
    
    diff_a = (a[i, :] + a[:, i]) - (a[j, :] + a[:, j])
    diff_b = b[y, :] - b[x, :]
    
    # Now compute sum_{k != i,j} diff_a[k] * diff_b[p[k]]
    # We can do this with numpy:
    # temp = diff_a * diff_b[p]
    # Then subtract the terms for k=i and k=j
    temp = diff_a * diff_b[p]
    # Subtract k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[x]
    # Subtract k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[y]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta


def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute for faster swap evaluation
    a_sym = a + a.T  # a_sym[i,k] = a[i,k] + a[k,i]
    
    # Generate initial candidate using rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Local search: 2-opt swaps
    # We'll do multiple passes until no improvement or time limit
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    best_p = p.copy()
                    improved = True
    
    # If we have time, try a few random restarts with local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        # Try a few random perturbations
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            p2 = best_p.copy()
            # Randomly swap a few pairs
            for _ in range(10):
                i, j = random.sample(range(n), 2)
                p2[i], p2[j] = p2[j], p2[i]
            
            # Local search on p2
            improved2 = True
            while improved2:
                improved2 = False
                for i in range(n):
                    if time.time() - start_time > time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() - start_time > time_limit:
                            break
                        delta = _swap_delta(a, b, p2, i, j)
                        if delta < 0:
                            p2[i], p2[j] = p2[j], p2[i]
                            best_cost2 = _qap_cost(a, b, p2)
                            improved2 = True
                
                if time.time() - start_time > time_limit:
                    break
            
            cost2 = _qap_cost(a, b, p2)
            if cost2 < best_cost:
                best_cost = cost2
                best_p = p2.copy()
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
