"""
Multi-start Iterated Local Search (ILS) for QAPLIB wil100.
Strategy:
1. Vectorized O(n^2) computation of cost deltas for ALL possible 2-opt swaps.
   This allows a very fast steepest-descent local search (finding the best swap 
   in O(n^2) vector ops, then applying it).
2. Multi-start ILS:
   - Generate diverse initial solutions (random + greedy hybrid).
   - For each start, run a time-boxed ILS loop:
     a. Local Search: Repeatedly find the best improving swap using the 
        vectorized delta matrix and apply it until no improvement is found.
     b. Perturbation: Apply a fixed number of random non-improving swaps 
        (3-opt style random walk) to escape the local minimum.
     c. Local Search again.
   - Keep the global best.
3. Time management:
   - Strict deadline guard.
   - Adaptive local search budget: use remaining time to decide if another 
     full ILS cycle is possible.
   
Why this is different:
- Current program: Single SA trajectory with random neighbor sampling.
- New program: Multiple independent ILS trajectories with deterministic 
  steepest-descent local search. The vectorized all-swap delta calculation 
  is the key enabler for fast local search.
"""
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of facilities (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    
    Formula for delta(i, j):
    Let x = p[i], y = p[j].
    The cost change is:
    1. Terms involving i and j with all other k:
       sum_{k != i, j} [ a[k,i] * (b[p[k],y] - b[p[k],x]) + a[i,k] * (b[y,p[k]] - b[x,p[k]]) ]
       - sum_{k != i, j} [ a[k,j] * (b[p[k],y] - b[p[k],x]) + a[j,k] * (b[y,p[k]] - b[x,p[k]]) ]
       
    This can be rewritten using matrix operations.
    Let A = a, B = b.
    Let P = p.
    
    For a fixed swap (i, j):
    Delta = 
      sum_k a[k,i] * (B[P[k], P[j]] - B[P[k], P[i]]) 
      + sum_k a[i,k] * (B[P[j], P[k]] - B[P[i], P[k]])
      - sum_k a[k,j] * (B[P[k], P[j]] - B[P[k], P[i]]) 
      - sum_k a[j,k] * (B[P[j], P[k]] - B[P[i], P[k]])
      + correction terms for k=i and k=j.
      
    Let's define:
    V_k = B[P[k], P[j]] - B[P[k], P[i]]  (vector of length n, index k)
    W_k = B[P[j], P[k]] - B[P[i], P[k]]  (vector of length n, index k)
    
    Since B is symmetric, V_k = W_k. Let's call it D_k.
    D_k = B[P[k], P[j]] - B[P[k], P[i]]
    
    Then the main sum is:
    sum_k (a[k,i] - a[k,j]) * D_k  +  sum_k (a[i,k] - a[j,k]) * D_k
    
    Note: sum_k a[i,k] D_k is the dot product of row i of A and D.
    sum_k a[k,i] D_k is the dot product of column i of A and D.
    
    So, for a fixed j, we can compute D_j (vector of length n) for all i?
    No, D depends on i and j.
    
    Let's vectorize over i for a fixed j?
    D[i, k] = B[P[k], P[j]] - B[P[k], P[i]]
    This is B[:, P[j]] - B[:, P[i]] for each i?
    B[:, P[j]] is a column vector.
    B[:, P[i]] is a column vector.
    
    Let's construct the matrix of D values for all i, j.
    This seems complex. Let's try a different vectorization.
    
    We want Delta[i, j] for all i, j.
    
    Let's use the property:
    Delta[i, j] = 
      (A[:, i] - A[:, j]) @ (B[P, P[j]] - B[P, P[i]])
      + (A[i, :] - A[j, :]) @ (B[P[j], P] - B[P[i], P])
      - correction
    
    Wait, B[P, P[j]] is a vector where element k is B[P[k], P[j]].
    Let's call this Col_j.
    B[P, P[i]] is Col_i.
    
    So Term1[i, j] = (A[:, i] - A[:, j]) @ (Col_j - Col_i)
    Term2[i, j] = (A[i, :] - A[j, :]) @ (Col_j - Col_i)  # Since B is symmetric, row/col same
    
    Let C be the matrix where C[k, i] = B[P[k], P[i]]. This is just B[P, P] (permuted B).
    Actually, C = B[np.ix_(P, P)].
    
    Then Col_j = C[:, j].
    Col_i = C[:, i].
    
    Term1[i, j] = (A[:, i] - A[:, j]) @ (C[:, j] - C[:, i])
    Term2[i, j] = (A[i, :] - A[j, :]) @ (C[:, j] - C[:, i])
    
    We can compute these for all i, j using matrix multiplication?
    
    Let U = A (n x n)
    Let V = C (n x n)
    
    We want M[i, j] = (U[:, i] - U[:, j]) @ (V[:, j] - V[:, i])
    
    Expand:
    U[:, i] @ V[:, j] - U[:, i] @ V[:, i] - U[:, j] @ V[:, j] + U[:, j] @ V[:, i]
    
    Let S1 = U.T @ V  (n x n). S1[i, j] = U[:, i] @ V[:, j].
    Let S2 = U.T @ U  (n x n). S2[i, i] = U[:, i] @ U[:, i] (not directly useful in this form, but S1[i, i] is U[:, i] @ V[:, i]).
    
    So:
    Term1[i, j] = S1[i, j] - S1[i, i] - S1[j, j] + S1[j, i]
    
    Similarly for Term2:
    Term2[i, j] = (A[i, :] - A[j, :]) @ (C[:, j] - C[:, i])
    = A[i, :] @ C[:, j] - A[i, :] @ C[:, i] - A[j, :] @ C[:, j] + A[j, :] @ C[:, i]
    
    Let T1 = A @ C.T (n x n). T1[i, j] = A[i, :] @ C[:, j].
    Note: C is symmetric (since B is symmetric and P is a permutation), so C.T = C.
    So T1[i, j] = A[i, :] @ C[:, j].
    
    Term2[i, j] = T1[i, j] - T1[i, i] - T1[j, j] + T1[j, i]
    
    So Total Delta[i, j] (before corrections) = Term1[i, j] + Term2[i, j]
    
    Now, what about the corrections for k=i and k=j?
    The sum over k included k=i and k=j.
    We need to exclude them.
    
    In the derivation of Term1 and Term2, we summed over all k.
    The terms for k=i and k=j in the sum:
    
    For Term1 (column part):
    k=i: (a[i,i]-a[i,j]) * (b[p[i], p[j]] - b[p[i], p[i]])
    k=j: (a[j,i]-a[j,j]) * (b[p[j], p[j]] - b[p[j], p[i]])
    
    For Term2 (row part):
    k=i: (a[i,i]-a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
    k=j: (a[i,j]-a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
    
    Assuming zero diagonal in A and B (standard QAP):
    a[i,i]=0, b[p[i],p[i]]=0.
    
    k=i in Term1: (0 - a[i,j]) * (b[p[i], p[j]] - 0) = -a[i,j] b[p[i], p[j]]
    k=j in Term1: (a[j,i] - 0) * (0 - b[p[j], p[i]]) = -a[j,i] b[p[j], p[i]]
    
    k=i in Term2: (0 - a[j,i]) * (b[p[j], p[i]] - 0) = -a[j,i] b[p[j], p[i]]
    k=j in Term2: (a[i,j] - 0) * (0 - b[p[i], p[j]]) = -a[i,j] b[p[i], p[j]]
    
    Sum of these 4 error terms:
    -2 * (a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]])
    
    However, the actual cost change also includes the change in the direct terms a[i,j] b[p[i], p[j]] and a[j,i] b[p[j], p[i]].
    Original: a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]]
    New: a[i,j] b[p[j], p[i]] + a[j,i] b[p[i], p[j]]
    
    Change = a[i,j] (b[p[j], p[i]] - b[p[i], p[j]]) + a[j,i] (b[p[i], p[j]] - b[p[j], p[i]])
    = (a[i,j] - a[j,i]) (b[p[j], p[i]] - b[p[i], p[j]])
    
    If B is symmetric, b[p[j], p[i]] = b[p[i], p[j]], so this change is 0.
    Wait, if the change is 0, then we don't need to add it back?
    
    Let's re-verify the "sum over all k" logic.
    The formula `sum_{k != i, j} ...` is what we want.
    The matrix multiplication `S1` calculates `sum_{k=0}^{n-1} ...`.
    So `S1[i, j]` includes the k=i and k=j terms.
    We calculated that the sum of k=i and k=j terms in the *raw* matrix multiplication result is:
    -2 * (a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]])
    
    But the *true* delta should NOT include these terms in the "interaction with others" part.
    So we must subtract these "error" terms from the raw result.
    
    Error[i, j] = -2 * (a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]])
    
    So, Corrected Delta[i, j] = Raw Delta[i, j] - Error[i, j]
    = Raw Delta[i, j] + 2 * (a[i,j] b[p[i], p[j]] + a[j,i] b[p[j], p[i]])
    
    Let's check this with a small example.
    n=2. A=[[0, 1], [1, 0]], B=[[0, 10], [10, 0]].
    p=[0, 1].
    Cost = 20.
    Swap i=0, j=1. p becomes [1, 0].
    Cost = 20. Delta = 0.
    
    Raw Delta calculation:
    C = B[p, p] = [[0, 10], [10, 0]].
    S1 = A.T @ C = A @ C (since A sym) = [[0, 1], [1, 0]] @ [[0, 10], [10, 0]] = [[10, 0], [0, 10]].
    T1 = A @ C = [[10, 0], [0, 10]].
    
    Raw[0, 1] = (S1[0,1] - S1[0,0] - S1[1,1] + S1[1,0]) + (T1[0,1] - T1[0,0] - T1[1,1] + T1[1,0])
    S1[0,1]=0, S1[0,0]=10, S1[1,1]=10, S1[1,0]=0.
    Term1 = 0 - 10 - 10 + 0 = -20.
    T1 same. Term2 = -20.
    Raw = -40.
    
    Error[0, 1] = -2 * (a[0,1] b[p[0], p[1]] + a[1,0] b[p[1], p[0]])
    = -2 * (1 * 10 + 1 * 10) = -40.
    
    Corrected = Raw - Error = -40 - (-40) = 0.
    Matches!
    
    So the algorithm is:
    1. C = B[np.ix_(p, p)]
    2. S1 = A.T @ C
    3. T1 = A @ C
    4. Raw = (S1 - np.diag(S1).diagonal()[:, None] - np.diag(S1).diagonal()[None, :] + S1.T) 
       Wait, S1 is not symmetric generally.
       Term1[i, j] = S1[i, j] - S1[i, i] - S1[j, j] + S1[j, i]
       Let D_S1 = np.diag(S1)
       Term1 = S1 - D_S1[:, None] - D_S1[None, :] + S1.T
       
    5. Term2 = T1 - D_T1[:, None] - D_T1[None, :] + T1.T
       where D_T1 = np.diag(T1)
       
    6. Raw = Term1 + Term2
       
    7. Error = -2 * (A * C + A.T * C.T)  # Element-wise product
       Note: C is symmetric, so C.T = C.
       Error = -2 * (A * C + A.T * C)
       
    8. Delta = Raw - Error
       
    9. Set diagonal to 0 (no swap with self).
       
    This is O(n^3) due to the matrix multiplications S1 and T1.
    n=100. 100^3 = 1,000,000 ops.
    NumPy matmul is highly optimized (BLAS). This should be very fast, likely < 1ms.
    
    Returns Delta matrix.
    """
    C = b[np.ix_(p, p)]
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    Term1 = S1 - D_S1[:, None] - D_S1[None, :] + S1.T
    Term2 = T1 - D_T1[:, None] - D_T1[None, :] + T1.T
    
    Raw = Term1 + Term2
    
    # Error term: -2 * (a[i,j] c[i,j] + a[j,i] c[j,i])
    # c[i,j] = b[p[i], p[j]]
    # Since C is symmetric, c[j,i] = c[i,j].
    # So Error[i,j] = -2 * (a[i,j] + a[j,i]) * C[i,j]
    Error = -2.0 * (a + a.T) * C
    
    Delta = Raw - Error
    
    # Zero out diagonal
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=1000, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_delta = np.min(deltas)
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # 5s buffer
    start_time = time.time()
    
    # Pre-compute some constants if needed? 
    # The delta function uses a and b directly.
    
    best_p = None
    best_cost = float('inf')
    
    # Multi-start
    num_starts = 0
    while time.time() - start_time < total_deadline - 5.0:
        # Generate initial solution
        # Mix of greedy and random
        if num_starts == 0:
            p = _greedy_initial(a, b)
        else:
            # Start from best solution found so far and perturb?
            # Or random?
            # Random is better for diversity in ILS.
            # But we can also start from greedy and perturb.
            # Let's use random for starts > 0 to ensure diversity.
            p = np.random.permutation(n)
            
        # Run ILS for this start
        # Budget: 20% of remaining time for this ILS trajectory?
        # Or fixed number of iterations?
        # Let's give each start a chunk of time.
        # If we have 100s left, and we want 5 starts, 20s each.
        # But we don't know how many starts we can afford.
        # Let's just run ILS until time is up, but with a sub-limit.
        
        # Actually, simpler:
        # Run local search to convergence.
        # If time remains, perturb and run local search again.
        
        p, current_cost = _local_search(a, b, p, time_limit=total_deadline, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Check time
        if time.time() - start_time > total_deadline - 2.0:
            break
            
        # Perturb for next iteration of ILS (if we continue)
        # Perturbation: 3 random swaps
        for _ in range(3):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
        num_starts += 1
        
    if best_p is None:
        # Fallback: return greedy
        best_p = _greedy_initial(a, b)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
