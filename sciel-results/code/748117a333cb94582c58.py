"""
Vector Alignment Initialization + Vectorized First-Improvement Local Search for QAP.

Strategy:
1. Initialization: Use a spectral/soft-assignment approach. 
   - Compute row sums of A and B.
   - Sort facilities by A-row-sum and locations by B-row-sum.
   - This "Rank Matching" is a strong baseline. 
   - To improve, we can use a few iterations of a linear assignment problem (LAP) 
     approximation or simply stick to the rank match which is fast and robust for tai256c.
   - Actually, a better init is to use the eigenvectors of A and B if they were 
     structured, but for general QAP, rank matching on row sums is standard and fast.
   
2. Local Search Engine: 
   - Instead of Tabu Search or Simulated Annealing, use a **First-Improvement** 
     local search with a **Vectorized Delta Evaluation**.
   - The key is speed. We evaluate the delta for swapping facility at position i 
     with facility at position j.
   - Delta(i, j) = sum_k (A[i,k] - A[j,k]) * (B[p[i], p[k]] - B[p[j], p[k]]) ... wait, 
     the standard formula is:
     Let x = p[i], y = p[j].
     Delta = sum_{k != i, j} (A[i,k] - A[j,k]) * (B[x, p[k]] - B[y, p[k]]) 
             + (A[i,j] - A[j,i]) * (B[x, y] - B[y, x]) ... 
     Actually, the O(n) delta for swapping positions i and j in permutation p:
     Let u = p[i], v = p[j].
     The change in cost is:
     C_new - C_old = 
       - 2 * sum_{k != i, j} A[i, k] B[u, p[k]] + 2 * sum_{k != i, j} A[j, k] B[v, p[k]]
       + 2 * A[i, j] B[u, v] - 2 * A[i, j] B[v, u]  (cross terms between i and j)
       - 2 * A[j, i] B[v, u] + 2 * A[j, i] B[u, v] ... wait, A is symmetric usually.
       
     Let's use the precomputed interaction matrix approach for O(1) or O(n) updates.
     A faster approach for "First Improvement" is to scan i from 0 to n-1, 
     and for each i, find the best j > i to swap with.
     To make it vectorized:
     For a fixed i, we can compute the delta for swapping i with all j in O(n) 
     using vector operations.
     Delta_i_j = (A[i, :] - A[j, :]) dot (B[p[i], p] - B[p[j], p]) ... this is not quite right.
     
     Let's derive the O(n) delta for swapping positions i and j (facilities u=p[i], v=p[j]):
     Let S = {k | k != i, j}.
     Cost terms involving i or j:
     Old: 
       sum_{k in S} A[i,k]B[u,p[k]] + A[k,i]B[p[k],u] 
       + sum_{k in S} A[j,k]B[v,p[k]] + A[k,j]B[p[k],v]
       + A[i,j]B[u,v] + A[j,i]B[v,u]
     New:
       sum_{k in S} A[i,k]B[v,p[k]] + A[k,i]B[p[k],v]
       + sum_{k in S} A[j,k]B[u,p[k]] + A[k,j]B[p[k],u]
       + A[i,j]B[v,u] + A[j,i]B[u,v]
       
     Assuming A and B are symmetric (tai256c is symmetric):
     Old_i_part = 2 * sum_{k in S} A[i,k]B[u,p[k]] + A[i,j]B[u,v]
     New_i_part = 2 * sum_{k in S} A[i,k]B[v,p[k]] + A[i,j]B[v,u]
     Old_j_part = 2 * sum_{k in S} A[j,k]B[v,p[k]] + A[j,i]B[v,u]
     New_j_part = 2 * sum_{k in S} A[j,k]B[u,p[k]] + A[j,i]B[u,v]
     
     Delta = New - Old
     = 2 * sum_{k in S} [ A[i,k](B[v,p[k]] - B[u,p[k]]) + A[j,k](B[u,p[k]] - B[v,p[k]]) ]
       + A[i,j](B[v,u] - B[u,v]) + A[j,i](B[u,v] - B[v,u])
     Since B is symmetric, B[u,v] = B[v,u], so the last two terms cancel.
     Delta = 2 * sum_{k in S} (A[i,k] - A[j,k]) * (B[v,p[k]] - B[u,p[k]])
     
     This is O(n) to compute for a pair (i,j).
     To do "First Improvement" efficiently:
     We can iterate i from 0 to n-1.
     For each i, we want to find j > i such that Delta(i,j) < 0.
     We can compute the vector D_j for all j > i in O(n^2) total per pass? 
     No, computing D_j for all j is O(n^2) per i, which is O(n^3) per pass. Too slow.
     
     Alternative: Random Sampling of Moves with Vectorized Evaluation.
     Sample M random pairs (i,j). Compute their deltas in a vectorized batch.
     Apply the best improving move (or first improving move).
     This is effectively a stochastic local search, but with a much larger batch size 
     allowed by vectorization, leading to faster convergence.
     
     Let's implement:
     1. Init: Rank Matching.
     2. Loop:
        a. Generate M random pairs (i,j).
        b. Compute deltas for all M pairs in one vectorized operation (if possible) 
           or loop in Python but with minimal overhead. 
           Actually, looping M=1000 in Python is fast (~1ms). 
           M=10000 is ~10ms. 
           We can do ~10000 evaluations per ms.
        c. Find the best improving delta.
        d. If found, apply it. Reset no_improve counter.
        e. If not found, increase M or restart.
        
     To answer the "error_repair" idea: 
     Instead of random restarts, we will implement a "Repair" phase.
     If stuck (no improvement in last X steps), we identify the facility with the 
     highest "penalty" (sum of A[i,k]*B[p[i],p[k]] for k!=i) and force it to move 
     to a location that minimizes this penalty, ignoring other effects for a moment.
     This is a greedy repair step.
"""

import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _solve_tai256c(a, b, time_limit=115.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute symmetric versions if needed, but assume symmetric for delta formula
    # tai256c is symmetric.
    a_sym = a
    b_sym = b
    
    # 1. Initialization: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute B rows for fast access? b[p] is a view.
    
    # Parameters
    M = 2000  # Number of random moves to sample per iteration
    max_no_improve = 50
    no_improve_count = 0
    rng = np.random.default_rng(42)
    
    # Main Loop
    while time.time() - start_time < time_limit:
        # Sample M random pairs (i, j)
        # Ensure i != j
        i_idx = rng.integers(0, n, size=M)
        j_idx = rng.integers(0, n, size=M)
        # Mask where i == j
        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        if len(i_idx) == 0:
            continue
            
        # Compute Deltas
        # u = p[i_idx], v = p[j_idx]
        u = p[i_idx]
        v = p[j_idx]
        
        # Delta = 2 * sum_{k != i, j} (A[i,k] - A[j,k]) * (B[v,p[k]] - B[u,p[k]])
        # We can compute this for all M pairs in a loop, but vectorizing over M is tricky 
        # because i and j vary.
        # However, we can compute the term T[i, k] = A[i, k] * B[p[i], p[k]]? No.
        
        # Let's loop over the M pairs but use numpy for the inner sum.
        # This is M * O(n) operations.
        # M=2000, n=256. 2000 * 256 = 512,000 ops. Very fast.
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Optimization: Precompute B_u and B_v for the sampled u and v?
        # B[u] is a row of B.
        
        for idx in range(len(i_idx)):
            i = i_idx[idx]
            j = j_idx[idx]
            u_val = p[i]
            v_val = p[j]
            
            # Vectorized sum for this pair
            # diff_a = a[i, :] - a[j, :]
            # diff_b = b[v_val, :] - b[u_val, :]
            # We need sum over k != i, j of diff_a[k] * diff_b[p[k]]
            
            # To avoid O(n) Python loop, use numpy:
            # term = (a[i, :] - a[j, :]) * (b[v_val, p] - b[u_val, p])
            # Sum, then subtract terms for k=i and k=j
            
            diff_a = a[i, :] - a[j, :]
            diff_b = b[v_val, :] - b[u_val, :]
            
            # p is the permutation. p[k] is the location of facility k.
            # We need diff_b[p[k]] for all k.
            # This is b[v_val, p] - b[u_val, p]
            
            product = diff_a * (b[v_val, p] - b[u_val, p])
            
            # Subtract k=i and k=j
            # At k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[u_val]
            # At k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[v_val]
            
            total_sum = np.sum(product)
            correction_i = diff_a[i] * diff_b[u_val]
            correction_j = diff_a[j] * diff_b[v_val]
            
            delta = 2.0 * (total_sum - correction_i - correction_j)
            
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
        
        if best_i != -1:
            # Apply move
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            no_improve_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve_count += 1
            
            # Error Repair Strategy
            if no_improve_count > max_no_improve:
                # Identify the facility with the highest local penalty
                # Penalty for facility i at location p[i]:
                # sum_{k != i} A[i, k] * B[p[i], p[k]]
                
                # Compute penalties for all i
                # This is O(n^2)
                # We can approximate or do it exactly.
                # Let's do it exactly for the repair step.
                
                # B_p = b[p, p] is the distance matrix under current assignment.
                # Penalty[i] = sum_k A[i, k] * B_p[i, k] - A[i, i]*B_p[i, i]
                # Since A[i,i]=0 usually, Penalty[i] = sum_k A[i, k] * B_p[i, k]
                
                B_p = b[np.ix_(p, p)]
                # A is n x n. B_p is n x n.
                # Penalties = sum(A * B_p, axis=1)
                penalties = np.sum(a * B_p, axis=1)
                
                # Find the worst facility (highest penalty)
                worst_facility = np.argmax(penalties)
                worst_location = p[worst_facility]
                
                # Find the best location for this facility
                # Cost contribution of facility i at location l:
                # sum_{k != i} A[i, k] * B[l, p[k]]
                # We want to minimize this.
                
                # Vectorized calculation for all locations l:
                # For a given i, cost_l = sum_k A[i, k] * B[l, p[k]]
                # This is (A[i, :]) dot (B[:, p[k]] for all k)?
                # Let's compute matrix M where M[l, k] = B[l, p[k]]
                # M = b[:, p]
                # Then cost_l = A[i, :] dot M[l, :]
                # We want argmin_l cost_l.
                
                M = b[:, p] # Shape (n, n). M[l, k] = b[l, p[k]]
                costs = M @ a[worst_facility, :] # Shape (n,)
                
                # Exclude current location? No, it might be the best.
                best_new_location = np.argmin(costs)
                
                # Swap worst_facility with the facility currently at best_new_location
                if best_new_location != worst_location:
                    # Find index j such that p[j] == best_new_location
                    j = np.where(p == best_new_location)[0][0]
                    i = worst_facility
                    
                    # Apply swap
                    p[i], p[j] = p[j], p[i]
                    current_cost = _qap_cost(a, b, p) # Recompute to be safe
                    no_improve_count = 0
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    # If it's already there, do a random restart
                    p = np.random.permutation(n)
                    current_cost = _qap_cost(a, b, p)
                    no_improve_count = 0

    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
