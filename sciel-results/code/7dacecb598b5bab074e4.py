"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix
import sys

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic where needed, 
    # but float64 for LP and spectral operations
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    # 1. Initialization via LP Relaxation
    # The QAP can be relaxed to a Linear Program where the permutation matrix P is 
    # replaced by a doubly stochastic matrix X.
    # Minimize sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # This is quadratic in X. However, a common linear relaxation is to fix one matrix 
    # or use a specific linearization. 
    # A simpler and effective heuristic initialization is to solve the assignment problem 
    # for a "linearized" cost matrix derived from the interaction of A and B.
    
    # Strategy: 
    # 1. Compute a linear cost matrix C where C[i, j] approximates the cost of placing 
    #    facility i at location j.
    #    A common approximation is C[i, j] = sum_k a[i, k] * b[j, k] (or similar).
    #    Actually, the standard linearization for QAP initialization often uses:
    #    C[i, j] = sum_k a[i, k] * (sum_l b[k, l]) + sum_k b[j, k] * (sum_l a[k, l])
    #    But a better one is to use the spectral method or a simple greedy.
    
    # Let's try a robust spectral initialization first as it's fast and often good.
    # Then, if time permits, we can try to refine with LP if we had a linear model, 
    # but QAP is NP-hard. The LP relaxation of QAP is complex.
    # Instead, let's use a **Greedy Construction with Lookahead** or **Spectral + Hungarian**.
    
    # Spectral Initialization (Standard for QAP)
    # Eigenvectors of A and B.
    try:
        # Symmetrize
        A_sym = (a_float + a_float.T) / 2.0
        B_sym = (b_float + b_float.T) / 2.0
        
        # Get top eigenvector
        # For n=150, eigh is fine
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
        
        # Take the eigenvector corresponding to the largest eigenvalue
        # Note: For QAP minimization, we usually align the largest components.
        # If A and B are similar in structure, the top eigenvectors should be aligned.
        idx_A = np.argmax(eigvals_A)
        idx_B = np.argmax(eigvals_B)
        
        v_A = eigvecs_A[:, idx_A]
        v_B = eigvecs_B[:, idx_B]
        
        # Rank matching
        # Sort facilities by v_A and locations by v_B
        fac_order = np.argsort(v_A)
        loc_order = np.argsort(v_B)
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        
    except Exception:
        # Fallback: Random
        p = np.random.permutation(n)

    # 2. Refine with Hungarian Algorithm on a linearized cost?
    # The cost of a permutation p is sum a[i,j] b[p[i], p[j]].
    # This is not linear in p.
    # However, we can iteratively improve using a "linearized" cost matrix based on the current solution.
    # Let p be the current permutation.
    # Define a cost matrix C where C[i, j] = sum_k a[i, k] * b[j, p[k]]
    # If we solve the assignment problem for C, we get a new permutation p'.
    # This is a form of Coordinate Descent / Relaxation.
    
    def compute_linear_cost_matrix(p):
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This can be computed as:
        # Let B_p be the matrix where B_p[j, k] = b[j, p[k]]
        # Then C = a @ B_p ? No.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Note that b[j, p[k]] is the element (j, p[k]) of b.
        # Let's construct the matrix M where M[k, j] = b[j, p[k]].
        # M is essentially b with columns permuted by p.
        # M = b[:, p]
        # Then C[i, j] = sum_k a[i, k] * M[k, j]
        # So C = a @ M = a @ b[:, p]
        
        return a_float @ b_float[:, p]

    # Iterative Linearization (Coordinate Descent on Permutation)
    # This is often faster and effective for QAP.
    current_p = p.copy()
    
    for _ in range(5): # A few iterations of this are cheap
        C = compute_linear_cost_matrix(current_p)
        # Solve Assignment Problem
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        # Check if it's better
        # Cost = sum a[i, j] b[new_p[i], new_p[j]]
        # We can compute this quickly with vectorization
        cost_new = np.sum(a_int * b_int[np.ix_(new_p, new_p)])
        cost_old = np.sum(a_int * b_int[np.ix_(current_p, current_p)])
        
        if cost_new < cost_old:
            current_p = new_p
        else:
            break
            
    p = current_p
    
    # 3. Local Search: 2-Swap
    # We need an efficient delta calculation.
    # Delta for swapping i and j:
    # Let u = p[i], v = p[j].
    # Delta = sum_{k != i, j} (a[k, i] - a[k, j]) * (b[p[k], v] - b[p[k], u]) 
    #       + a[i, i](b[v, v] - b[u, u]) + a[j, j](b[u, u] - b[v, v])
    #       + a[i, j](b[v, u] - b[u, v]) + a[j, i](b[u, v] - b[v, u]) -- these cancel if symmetric
    #       + a[i, j](b[v, u] - b[u, v]) -- wait, let's re-derive carefully.
    
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Terms involving i or j change.
    # 1. a[i,i] b[p[i], p[i]] -> a[i,i] b[v, v] (was b[u, u])
    # 2. a[j,j] b[p[j], p[j]] -> a[j,j] b[u, u] (was b[v, v])
    # 3. a[i,j] b[p[i], p[j]] -> a[i,j] b[v, u] (was b[u, v])
    # 4. a[j,i] b[p[j], p[i]] -> a[j,i] b[u, v] (was b[v, u])
    # 5. Terms where one index is i or j, other is k != i, j.
    #    a[i,k] b[p[i], p[k]] -> a[i,k] b[v, p[k]] (was b[u, p[k]])
    #    a[k,i] b[p[k], p[i]] -> a[k,i] b[p[k], v] (was b[p[k], u])
    #    a[j,k] b[p[j], p[k]] -> a[j,k] b[u, p[k]] (was b[v, p[k]])
    #    a[k,j] b[p[k], p[j]] -> a[k,j] b[p[k], u] (was b[p[k], v])
    
    # Assuming symmetry a[i,j]=a[j,i] and b[u,v]=b[v,u]:
    # Terms 3 & 4: a[i,j] b[v,u] + a[i,j] b[u,v] = 2 a[i,j] b[u,v]
    # Old: 2 a[i,j] b[u,v]. Change is 0.
    
    # So Delta = a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])
    #          + sum_{k != i, j} [ a[i,k](b[v, p[k]] - b[u, p[k]]) + a[k,i](b[p[k], v] - b[p[k], u]) 
    #                            + a[j,k](b[u, p[k]] - b[v, p[k]]) + a[k,j](b[p[k], u] - b[p[k], v]) ]
    
    # Using symmetry: a[i,k]=a[k,i], b[v, p[k]]=b[p[k], v].
    # Term for k: 2 * a[i,k] * (b[v, p[k]] - b[u, p[k]]) - 2 * a[j,k] * (b[v, p[k]] - b[u, p[k]])
    #            = 2 * (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    
    # So Delta = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u]) 
    #          + 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    
    # We can compute the sum efficiently.
    # Let diff_a = a[i, :] - a[j, :]
    # Let diff_b = b[v, :] - b[u, :]
    # We need sum_{k != i, j} diff_a[k] * diff_b[p[k]]
    
    # Vectorized:
    # diff_b_perm = diff_b[p]  # This gives diff_b[p[k]] for all k
    # product = diff_a * diff_b_perm
    # Sum all, then subtract terms for k=i and k=j.
    
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Diagonal part
        delta_diag = (a_int[i, i] - a_int[j, j]) * (b_int[v, v] - b_int[u, u])
        
        # Off-diagonal part
        diff_a = a_int[i, :] - a_int[j, :]
        diff_b = b_int[v, :] - b_int[u, :]
        
        # Map diff_b to current positions
        diff_b_perm = diff_b[p]
        
        total_sum = np.dot(diff_a, diff_b_perm)
        
        # Subtract k=i and k=j terms
        # k=i: diff_a[i] * diff_b_perm[i] = diff_a[i] * diff_b[u]
        # k=j: diff_a[j] * diff_b_perm[j] = diff_a[j] * diff_b[v]
        term_i = diff_a[i] * diff_b[u]
        term_j = diff_a[j] * diff_b[v]
        
        sum_k = total_sum - term_i - term_j
        
        delta = delta_diag + 2 * sum_k
        return delta

    # Local Search Loop
    best_p = p.copy()
    best_cost = int(np.sum(a_int * b_int[np.ix_(best_p, best_p)]))
    current_p = p.copy()
    current_cost = best_cost
    
    # Precompute some things? 
    # We can't precompute much because p changes.
    
    # Use random sampling of swaps to avoid O(n^2) per iteration if n is large.
    # n=150, n^2/2 = 11250. 
    # Each delta is O(n) vectorized. 
    # 11250 * 150 ops ~ 1.7M ops. 
    # In NumPy, this is fast (milliseconds).
    # So we can do full scans!
    
    full_scan = True
    no_improvement_count = 0
    max_no_improvement = 10 # Stop if 10 full scans yield no improvement
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 5:
            break
            
        improved = False
        
        if full_scan:
            # Generate all pairs
            # We can iterate i from 0 to n-1, j from i+1 to n-1
            # To speed up, we can vectorize the delta calculation for all j for a given i?
            # Delta(i, j) depends on p[i] and p[j].
            # diff_a = a[i, :] - a[j, :]
            # diff_b = b[p[i], :] - b[p[j], :]
            # This is hard to vectorize over j because diff_b changes.
            
            # Let's just loop i, and for each i, loop j.
            # 150 * 150 = 22500 iterations of Python loop? No, 11250.
            # Each iteration calls get_delta which does vectorized ops.
            # This might be slow in Python due to loop overhead.
            
            # Optimization: 
            # For a fixed i, u = p[i].
            # diff_b_i = b[u, :]
            # We need sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
            # Let's try to batch it.
            
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                u = current_p[i]
                # diff_a_i = a[i, :]
                # We want to find j that minimizes delta.
                
                # Vectorize over j?
                # j ranges from 0 to n-1, j != i.
                # v_j = current_p[j]
                # diff_b_j = b[v_j, :] - b[u, :]
                
                # This is tricky to fully vectorize without huge memory.
                # Let's stick to a simple loop but optimize the inner part.
                
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit - 1:
                        break
                    d = get_delta(i, j, current_p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
                if time.time() - start_time > time_limit - 1:
                    break
            
            if best_delta < 0:
                # Apply swap
                i, j = best_i, best_j
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += int(best_delta)
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    break
        else:
            # Random sampling
            # ...
            pass

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
