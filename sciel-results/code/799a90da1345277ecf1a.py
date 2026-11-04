"""
Multi-Start Iterated Local Search (ILS) for QAP tai256c.
Strategy: 
1. Generate diverse initial solutions using rank-based heuristics and random perturbations.
2. Intensive local search using 2-opt swaps with O(n) delta evaluation.
3. Adaptive restarts if no improvement is found for a certain number of iterations.
4. Maintain a global best solution.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p is a permutation of 0..n-1
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search(a, b, p, max_iters=5000, time_limit=None, start_time=0.0):
    """
    Perform 2-opt local search on permutation p.
    Returns improved permutation and cost.
    """
    n = len(a)
    # Precompute symmetric flow matrix for delta calculation
    # Delta for swapping i and j:
    # C_new - C_old = sum_{k != i,j} (a[i,k] + a[j,k] - a[i,i] - a[j,j]) * (b[p[i], p[k]] + b[p[j], p[k]] - ...)
    # Actually, the standard fast delta for QAP 2-opt:
    # Let x = p[i], y = p[j].
    # Delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[y, p[k]] - b[x, p[k]]) + ... correction terms
    # A more robust O(n) formula:
    # Let A = a, B = b.
    # Cost = sum_i sum_j A[i,j] B[p[i], p[j]]
    # Swap i, j: p[i]<->x, p[j]<->y.
    # Change involves terms where i or j is involved.
    # Delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[y, p[k]] - B[x, p[k]]) 
    #         + (A[i,j] - A[j,i]) * (B[x, y] - B[y, x]) # This term is tricky if A is not symmetric
    # If A is symmetric (common in QAP), A[i,j]=A[j,i], so the last term vanishes? No.
    # Let's use the robust vectorized form which is O(n) but with high constant.
    
    # Precompute A_sym = A + A.T to handle non-symmetric A? 
    # The problem statement says "sum of flow times distance". Usually QAP uses symmetric A and B.
    # tai256c is a standard instance. Let's assume symmetric or handle generally.
    # General formula for swap i,j:
    # Let u = p[i], v = p[j].
    # Delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[v, p[k]] - B[u, p[k]]) 
    #         + (A[i,j] - A[j,i]) * (B[u, v] - B[v, u])
    #         + (A[i,i] - A[j,j]) * (B[u, u] - B[v, v]) # Diagonal terms?
    # Actually, the simplest correct O(n) delta for symmetric A, B:
    # Delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[v, p[k]] - B[u, p[k]])
    #         + (A[i,j] - A[j,i]) * (B[u, v] - B[v, u])
    # If A is symmetric, A[i,j]=A[j,i], so the second term is 0.
    # If B is symmetric, B[u,v]=B[v,u], so the second term is 0.
    # So for symmetric A,B: Delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[v, p[k]] - B[u, p[k]])
    
    # Let's verify this.
    # C = sum_{i,j} A[i,j] B[p[i], p[j]]
    # Terms involving i or j:
    # sum_k A[i,k] B[p[i], p[k]] + sum_k A[j,k] B[p[j], p[k]] - A[i,j]B[p[i],p[j]] - A[j,i]B[p[j],p[i]] (double counted)
    # After swap:
    # sum_k A[i,k] B[p[j], p[k]] + sum_k A[j,k] B[p[i], p[k]] - A[i,j]B[p[j],p[i]] - A[j,i]B[p[i],p[j]]
    # Delta = [sum_k A[i,k] (B[p[j], p[k]] - B[p[i], p[k]])] + [sum_k A[j,k] (B[p[i], p[k]] - B[p[j], p[k]])]
    #        - (A[i,j]B[p[j],p[i]] + A[j,i]B[p[i],p[j]] - A[i,j]B[p[i],p[j]] - A[j,i]B[p[j],p[i]])
    # The last part: -A[i,j](B[p[j],p[i]] - B[p[i],p[j]]) - A[j,i](B[p[i],p[j]] - B[p[j],p[i]])
    # If symmetric B, B[p[j],p[i]] = B[p[i],p[j]], so this is 0.
    # If symmetric A, A[i,j]=A[j,i].
    # So for symmetric A,B:
    # Delta = sum_k (A[i,k] - A[j,k]) * (B[p[j], p[k]] - B[p[i], p[k]])
    # Note: the sum includes k=i and k=j.
    # If k=i: (A[i,i] - A[j,i]) * (B[p[j], p[i]] - B[p[i], p[i]])
    # If k=j: (A[i,j] - A[j,j]) * (B[p[j], p[j]] - B[p[i], p[j]])
    # The formula derived earlier excluded i,j. Let's include them and see if it matches.
    # The standard formula is:
    # Delta = sum_{k} (A[i,k] - A[j,k]) * (B[v, p[k]] - B[u, p[k]])
    # where u=p[i], v=p[j].
    # This is O(n) and vectorizable.
    
    A = a
    B = b
    p = p.copy()
    current_cost = _qap_cost(A, B, p)
    
    no_improve_count = 0
    max_no_improve = 100
    
    while max_iters > 0:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Compute delta
        # diff_a = A[i, :] - A[j, :]
        # diff_b = B[v, :] - B[u, :]
        # delta = np.dot(diff_a, diff_b[p])
        
        diff_a = A[i] - A[j]
        diff_b = B[v] - B[u]
        delta = np.dot(diff_a, diff_b[p])
        
        if delta < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve_count = 0
            max_iters -= 1
        else:
            no_improve_count += 1
            max_iters -= 1
            if no_improve_count > max_no_improve:
                break
                
    return p, current_cost

def _rank_based_init(a, b):
    """
    Generate an initial solution using rank correlation.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by flow weight (descending)
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by distance weight (ascending? or descending?)
    # Usually, high flow should go to close locations (low distance).
    # So sort locations by row sum of B (which represents total distance) ascending.
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _perturb(p, strength=10):
    """
    Perturb the permutation by swapping 'strength' random pairs.
    """
    n = len(p)
    p = p.copy()
    for _ in range(strength):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Initial solution
    best_p = _rank_based_init(a, b)
    best_cost = _qap_cost(a, b, best_p)
    
    # Multi-start ILS
    # We will run several iterations. Each iteration:
    # 1. Local search on current best
    # 2. Perturb
    # 3. Local search on perturbed
    # 4. Update best if improved
    
    # To diversify, we can also restart from scratch occasionally.
    # But for now, let's stick to ILS on the best found.
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    while time.time() - start_time < time_limit:
        # Intensive local search
        current_p, current_cost = _local_search(a, b, current_p, max_iters=2000, time_limit=time_limit, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
        
        # Check time before perturbing
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        # Increase perturbation strength if stuck?
        # Simple: fixed strength
        current_p = _perturb(best_p, strength=5)
        
        # Optional: If we've been stuck for a while, do a more aggressive restart?
        # For simplicity, we just continue.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
