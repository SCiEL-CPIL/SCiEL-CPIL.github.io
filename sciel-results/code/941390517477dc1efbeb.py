# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Symmetrize matrices for the trace formulation
    # Note: The cost is sum a[i,j] b[p[i], p[j]]. 
    # If we define X as assignment matrix, cost is tr(A X B X^T).
    # If A and B are not symmetric, we can symmetrize them for the relaxation 
    # because tr(A X B X^T) = tr(A_sym X B_sym X^T) is not strictly true unless 
    # we are careful. However, for the relaxation, using symmetrized matrices 
    # often works well. Let's stick to the exact cost function for evaluation.
    
    # 1. Initial Heuristic: Rank Matching
    # Sort facilities by total flow (row sum of A)
    # Sort locations by total distance (row sum of B)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # We want high flow facilities at low distance locations (central)
    # So sort facilities descending by flow, locations ascending by distance
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 2. Continuous Relaxation: FISTA on Birkhoff Polytope
    # Objective: f(X) = tr(A X B X^T)
    # Gradient: G = A X B + B X A (if A,B symmetric)
    # If not symmetric, gradient of tr(A X B X^T) w.r.t X is A X B + B X A? 
    # Let's derive: f(X) = sum_{i,j,k,l} A_{ij} X_{ik} B_{jl} X_{lk}
    # df/dX_{mn} = sum_{j,l} A_{mj} B_{ln} X_{lk} + sum_{i,k} A_{ik} B_{ni} X_{kl}
    # In matrix form: G = A X B + B X A.
    # Note: This assumes the term is tr(A X B X^T).
    # Let's use symmetrized A and B for the gradient to ensure symmetry and stability.
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # FISTA parameters
    L = 1.0 # Lipschitz constant estimate, we will use backtracking or fixed small step
    # For n=150, spectral norm of gradient operator is bounded.
    # Let's use a fixed step size that works well for n=150.
    # Step size alpha = 1 / L. 
    # Empirically, for QAP relaxation, alpha around 0.1 to 0.5 works.
    alpha = 0.2
    t_k = 1.0
    X_prev = X.copy()
    
    # Precompute for gradient
    # G = A_sym @ X @ B_sym + B_sym @ X @ A_sym
    
    for k in range(5000):
        if time.time() > deadline - 60.0: # Reserve 60s for local search
            break
            
        # Gradient
        # G = A_sym X B_sym + B_sym X A_sym
        AX = A_sym @ X
        G = AX @ B_sym + B_sym @ AX
        
        # FISTA Update
        # Y = X + (t_prev - 1) / (t_k + 1) * (X - X_prev)
        t_prev = t_k
        t_k = (1 + np.sqrt(1 + 4 * t_prev * t_prev)) / 2
        Y = X + (t_prev - 1) / t_k * (X - X_prev)
        
        # Gradient Step
        X_new = Y - alpha * G
        
        # Project onto Birkhoff Polytope (Doubly Stochastic)
        # Use Hungarian algorithm on -X_new to maximize sum X_new
        row_ind, col_ind = linear_sum_assignment(-X_new)
        X_proj = np.zeros((n, n), dtype=np.float64)
        X_proj[row_ind, col_ind] = 1.0
        
        # Update best solution from relaxation
        p_relax = np.empty(n, dtype=np.int64)
        p_relax[row_ind] = col_ind
        cost_relax = compute_cost(p_relax)
        
        if cost_relax < best_cost:
            best_cost = cost_relax
            best_p = p_relax.copy()
        
        X_prev = X
        X = X_proj
        
        # Optional: Reduce step size slowly if oscillating? 
        # For now, keep fixed.
        
    # 3. Local Search: Vectorized 2-Swap
    # We start from best_p
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute row sums of B? No, we need specific interactions.
    # For a swap (i, j), let u = p[i], v = p[j].
    # Delta = Cost_new - Cost_old
    # We can vectorize the calculation of delta for all j given a fixed i.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Precompute terms that depend only on i or only on the current assignment
        # Let's try a standard loop but with vectorized inner loop for j.
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
                
            u = p[i]
            
            # We want to swap i with all j > i
            # Delta(i, j) = sum_k a[i,k] (b[u,p[j]] - b[v,p[k]]) + ...
            
            # Let's use the standard incremental update formula but vectorized.
            # Let p be the current permutation.
            # Let u = p[i], v = p[j].
            
            # Terms involving i:
            # sum_k a[i,k] b[u, p[k]] -> a[i, :] @ b[u, p]
            # sum_k a[k,i] b[p[k], u] -> a[:, i] @ b[p, u]
            
            # Terms involving j:
            # sum_k a[j,k] b[v, p[k]] -> a[j, :] @ b[v, p]
            # sum_k a[k,j] b[p[k], v] -> a[:, j] @ b[p, v]
            
            # Cross terms between i and j:
            # a[i,j] b[u,v] + a[j,i] b[v,u]
            
            # Let's precompute vectors for the current p
            # B_p = b[p, :]  # Shape (n, n)
            # B_pT = b[:, p] # Shape (n, n)
            
            # This approach might be too memory intensive or slow to recompute for every i.
            # Instead, we calculate delta for all j at once for a fixed i.
            
            # Let's stick to a slightly optimized Python loop for i, but vectorize j.
            # For fixed i, u is fixed.
            # We need to find j that minimizes delta.
            
            # Precompute for fixed i:
            # term_i_u = a[i, :] @ b[u, p]  # scalar
            # term_i_uT = a[:, i] @ b[p, u] # scalar
            
            # For each j, v = p[j].
            # term_j_v = a[j, :] @ b[v, p]
            # term_j_vT = a[:, j] @ b[p, v]
            
            # This still requires O(n) work per j.
            # Total O(n^2) per pass.
            # n=150, n^2=22500. 22500 * 150 = 3.3M ops per pass.
            # In Python, this is too slow.
            
            # We need to vectorize the j-loop.
            
            # Let's try to compute the delta matrix for all pairs (i, j) at once?
            # Delta(i, j) = ...
            
            # Alternative: Use a library or a very tight C-extension? No.
            # Let's optimize the Python loop with NumPy inside.
            
            # For a fixed i, we can compute the vector of deltas for all j in one go?
            
            # Let's define:
            # S1 = a[i, :]  # Shape (n,)
            # S2 = a[:, i]  # Shape (n,)
            
            # We want to swap i with j.
            # The change in cost comes from:
            # 1. Interactions of i with all k (except j): a[i,k] b[u, p[k]] becomes a[i,k] b[u, p[k]] (no change) 
            #    Wait, p[k] doesn't change for k != i,j.
            #    So a[i,k] b[u, p[k]] stays same.
            #    But a[i,j] b[u, p[j]] becomes a[i,j] b[u, p[i]] = a[i,j] b[u, u]? No, p[i] is u.
            #    Let's be precise.
            
            # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
            
            # Let's use the formula:
            # Delta(i, j) = (a[i,j] b[u,v] + a[j,i] b[v,u]) - (a[i,j] b[u,u] + a[j,i] b[v,v])
            # + sum_{k!=i,j} a[i,k] (b[u, p[k]] - b[v, p[k]])  -- No, p[k] is same.
            # The term is a[i,k] b[u, p[k]]. In new solution, i is still at u? No, i moves to v?
            # No, swap means i goes to v and j goes to u.
            
            # New cost terms involving i:
            # a[i,j] b[v, u]  (i at v, j at u)
            # a[i,k] b[v, p[k]] for k!=i,j
            # a[k,i] b[p[k], v] for k!=i,j
            
            # Old cost terms involving i:
            # a[i,j] b[u, v]
            # a[i,k] b[u, p[k]] for k!=i,j
            # a[k,i] b[p[k], u] for k!=i,j
            
            # Difference for i:
            # a[i,j] (b[v,u] - b[u,v])
            # sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
            # sum_{k!=i,j} a[k,i] (b[p[k], v] - b[p[k], u])
            
            # Similarly for j:
            # a[j,i] (b[u,v] - b[v,u])
            # sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
            # sum_{k!=i,j} a[k,j] (b[p[k], u] - b[p[k], v])
            
            # Note: The term a[i,j] and a[j,i] are handled in the first line.
            
            # Let's combine:
            # Delta = a[i,j] (b[v,u] - b[u,v]) + a[j,i] (b[u,v] - b[v,u])
            # + sum_{k!=i,j} [ a[i,k] + a[j,k] ] (b[v, p[k]] - b[u, p[k]])
            # + sum_{k!=i,j} [ a[k,i] + a[k,j] ] (b[p[k], v] - b[p[k], u])
            
            # This is O(n) per pair.
            
            # Can we vectorize for all j?
            # Let u = p[i].
            # For all j, v = p[j].
            
            # Term 1: a[i,j] (b[v,u] - b[u,v])
            # Term 2: a[j,i] (b[u,v] - b[v,u])
            # Term 3: sum_{k!=i,j} (a[i,k] + a[j,k]) b[v, p[k]] - sum_{k!=i,j} (a[i,k] + a[j,k]) b[u, p[k]]
            # Term 4: sum_{k!=i,j} (a[k,i] + a[k,j]) b[p[k], v] - sum_{k!=i,j} (a[k,i] + a[k,j]) b[p[k], u]
            
            # This is getting complex. Let's stick to the simpler O(n^3) but optimized with NumPy.
            # For n=150, n^3 = 3.3M. If we do 100 passes, 330M ops.
            # NumPy can do 100M+ ops/sec. So 100 passes in ~3-10s.
            # This is feasible!
            
            # Let's implement the O(n^3) loop but with NumPy vectorization for the inner sum.
            
            # Precompute for fixed i:
            # A_i = a[i, :]  # (n,)
            # A_iT = a[:, i] # (n,)
            
            # For all j, we need:
            # v = p[j]
            
            # Let's compute the delta for all j > i in a vectorized way.
            
            # This is hard to do in one vectorized step because of the k!=i,j exclusion.
            # However, we can compute the sum over all k and then subtract the k=i and k=j terms.
            
            # Sum_{k} (a[i,k] + a[j,k]) b[v, p[k]]
            # = a[i, :] @ b[v, p] + a[j, :] @ b[v, p]
            
            # Let's do this.
            
            # For fixed i:
            A_i = a[i, :]
            A_iT = a[:, i]
            
            # We want to find best j.
            # Let's iterate j in Python, but use NumPy for the sums.
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5.0:
                    break
                
                v = p[j]
                A_j = a[j, :]
                A_jT = a[:, j]
                
                # Sum over all k
                # S1 = A_i @ b[v, p] + A_j @ b[v, p]
                # S2 = A_i @ b[u, p] + A_j @ b[u, p]
                # S3 = A_iT @ b[p, v] + A_jT @ b[p, v]
                # S4 = A_iT @ b[p, u] + A_jT @ b[p, u]
                
                # This is 4 dot products of size n.
                # 4 * 150 = 600 ops.
                # 150*150/2 = 11250 pairs.
                # 11250 * 600 = 6.75M ops per pass.
                # This is fast.
                
                S1 = A_i @ b[v, p] + A_j @ b[v, p]
                S2 = A_i @ b[u, p] + A_j @ b[u, p]
                S3 = A_iT @ b[p, v] + A_jT @ b[p, v]
                S4 = A_iT @ b[p, u] + A_jT @ b[p, u]
                
                # Subtract k=i and k=j terms
                # For k=i:
                # Term in S1: a[i,i] b[v, p[i]] + a[j,i] b[v, p[i]] = a[i,i] b[v, u] + a[j,i] b[v, u]
                # Term in S2: a[i,i] b[u, p[i]] + a[j,i] b[u, p[i]] = a[i,i] b[u, u] + a[j,i] b[u, u]
                # Term in S3: a[i,i] b[p[i], v] + a[j,i] b[p[i], v] = a[i,i] b[u, v] + a[j,i] b[u, v]
                # Term in S4: a[i,i] b[p[i], u] + a[j,i] b[p[i], u] = a[i,i] b[u, u] + a[j,i] b[u, u]
                
                # For k=j:
                # Term in S1: a[i,j] b[v, p[j]] + a[j,j] b[v, p[j]] = a[i,j] b[v, v] + a[j,j] b[v, v]
                # Term in S2: a[i,j] b[u, p[j]] + a[j,j] b[u, p[j]] = a[i,j] b[u, v] + a[j,j] b[u, v]
                # Term in S3: a[i,j] b[p[j], v] + a[j,j] b[p[j], v] = a[i,j] b[v, v] + a[j,j] b[v, v]
                # Term in S4: a[i,j] b[p[j], u] + a[j,j] b[p[j], u] = a[i,j] b[v, u] + a[j,j] b[v, u]
                
                # Correction = (S1 - S2 - S3 + S4) - (Terms_i + Terms_j)
                # Wait, the formula was:
                # Delta = a[i,j] (b[v,u] - b[u,v]) + a[j,i] (b[u,v] - b[v,u])
                # + (S1 - S2) - (S3 - S4)  -- No, signs are mixed.
                
                # Let's re-verify the sign.
                # New - Old
                # i terms:
                # a[i,j] b[v,u] - a[i,j] b[u,v]
                # sum_{k!=i,j} a[i,k] b[v, p[k]] - sum_{k!=i,j} a[i,k] b[u, p[k]]
                # sum_{k!=i,j} a[k,i] b[p[k], v] - sum_{k!=i,j} a[k,i] b[p[k], u]
                
                # j terms:
                # a[j,i] b[u,v] - a[j,i] b[v,u]
                # sum_{k!=i,j} a[j,k] b[u, p[k]] - sum_{k!=i,j} a[j,k] b[v, p[k]]
                # sum_{k!=i,j} a[k,j] b[p[k], u] - sum_{k!=i,j} a[k,j] b[p[k], v]
                
                # Combine sums:
                # sum_{k!=i,j} (a[i,k] - a[j,k]) b[v, p[k]] + (a[j,k] - a[i,k]) b[u, p[k])
                # = sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
                
                # sum_{k!=i,j} (a[k,i] - a[k,j]) b[p[k], v] + (a[k,j] - a[k,i]) b[p[k], u]
                # = sum_{k!=i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
                
                # So:
                # Delta = a[i,j] (b[v,u] - b[u,v]) + a[j,i] (b[u,v] - b[v,u])
                # + sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
                # + sum_{k!=i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
                
                # Let D1 = a[i, :] - a[j, :]
                # Let D2 = a[:, i] - a[:, j]
                
                # Sum_all_k D1[k] (b[v, p[k]] - b[u, p[k]])
                # Sum_all_k D2[k] (b[p[k], v] - b[p[k], u])
                
                # Subtract k=i and k=j terms.
                
                D1 = A_i - A_j
                D2 = A_iT - A_jT
                
                # Vectorized sum
                diff_b_vu = b[v, p] - b[u, p]
                diff_b_pv_u = b[p, v] - b[p, u]
                
                Sum1_all = D1 @ diff_b_vu
                Sum2_all = D2 @ diff_b_pv_u
                
                # Subtract k=i
                # D1[i] = a[i,i] - a[j,i]
                # diff_b_vu[i] = b[v, u] - b[u, u]
                # D2[i] = a[i,i] - a[j,i]
                # diff_b_pv_u[i] = b[u, v] - b[u, u]
                
                # Subtract k=j
                # D1[j] = a[i,j] - a[j,j]
                # diff_b_vu[j] = b[v, v] - b[u, v]
                # D2[j] = a[i,j] - a[j,j]
                # diff_b_pv_u[j] = b[v, v] - b[v, u]
                
                Sub_i_1 = D1[i] * (b[v, u] - b[u, u])
                Sub_i_2 = D2[i] * (b[u, v] - b[u, u])
                
                Sub_j_1 = D1[j] * (b[v, v] - b[u, v])
                Sub_j_2 = D2[j] * (b[v, v] - b[v, u])
                
                Sum1 = Sum1_all - Sub_i_1 - Sub_j_1
                Sum2 = Sum2_all - Sub_i_2 - Sub_j_2
                
                Cross = a[i,j] * (b[v,u] - b[u,v]) + a[j,i] * (b[u,v] - b[v,u])
                
                delta = Cross + Sum1 + Sum2
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
