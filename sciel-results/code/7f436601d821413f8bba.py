"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def solve(a, b):
    """
    Solves the QAP instance using Spectral Initialization and Greedy Vector Matching,
    followed by fast local search with optimized delta calculations.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Pre-compute necessary matrix properties for speed
    # a: Flow matrix (n x n)
    # b: Distance matrix (n x n)
    
    # 1. Spectral Initialization
    # The idea: The QAP cost is sum a[i,j] b[p[i], p[j]].
    # We want high flow pairs to be close.
    # Spectral methods align the principal components of the flow and distance matrices.
    
    try:
        # Compute top eigenvectors
        # Since matrices are symmetric, we can use eigh
        # We only need the top few eigenvectors to capture global structure
        k = min(10, n)
        
        # Flow matrix eigenvectors
        w_a, v_a = np.linalg.eigh(a)
        # Sort by eigenvalue descending
        idx_a = np.argsort(w_a)[::-1]
        v_a_top = v_a[:, idx_a[:k]]
        
        # Distance matrix eigenvectors
        w_b, v_b = np.linalg.eigh(b)
        idx_b = np.argsort(w_b)[::-1]
        v_b_top = v_b[:, idx_b[:k]]
        
        # Construct a similarity matrix between facilities and locations
        # We want to map facility i to location j such that their spectral embeddings are similar
        # Specifically, we align the dominant modes.
        
        # A common heuristic: 
        # S = V_a @ V_b.T
        # Then solve the assignment problem on -S (minimize distance in spectral space)
        # Or simply use the rank correlation of the top eigenvector components.
        
        # Let's use a robust spectral assignment:
        # Project facilities and locations onto the top k eigenvectors.
        # Then use a greedy or Hungarian match on the combined spectral distance.
        
        # For speed and robustness, let's use the "Rank-Matching" on the top eigenvector
        # which is often surprisingly strong for QAP instances with clear structure.
        
        # Top eigenvector of A
        top_vec_a = v_a[:, idx_a[0]]
        # Top eigenvector of B
        top_vec_b = v_b[:, idx_b[0]]
        
        # However, simple rank matching on 1D vectors can be fragile.
        # Let's use a 2D embedding for better discrimination.
        # Embedding A: (n, k)
        # Embedding B: (n, k)
        
        # We want to minimize sum || x_i - y_{p(i)} ||^2 ?
        # This is a linear assignment problem on the squared distances.
        
        # Compute squared Euclidean distances between all facility embeddings and location embeddings
        # dist[i, j] = || v_a_top[i] - v_b_top[j] ||^2
        # = ||v_a_top[i]||^2 + ||v_b_top[j]||^2 - 2 v_a_top[i] dot v_b_top[j]
        
        # To use scipy's linear_sum_assignment, we need a cost matrix.
        # Minimize sum dist[i, p[i]].
        
        # Let's compute the interaction matrix M = v_a_top @ v_b_top.T
        # dist[i,j] = A_norm[i] + B_norm[j] - 2*M[i,j]
        # Minimizing sum dist[i, p[i]] is equivalent to maximizing sum M[i, p[i]]
        # (since A_norm and B_norm sums are constant regardless of permutation)
        
        M = v_a_top @ v_b_top.T
        
        # Solve linear assignment to maximize M
        # linear_sum_assignment minimizes, so we pass -M
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-M)
        
        p_init = np.empty(n, dtype=np.int64)
        p_init[row_ind] = col_ind
        
    except Exception:
        # Fallback to random if spectral fails
        p_init = np.random.permutation(n)

    # 2. Fast Local Search with Optimized Delta Calculation
    # Current cost
    def calc_cost(p):
        return int(np.sum(a * b[p][:, p]))
    
    current_cost = calc_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    # Pre-compute row sums of A to speed up delta calculation?
    # Delta for swapping p[i] and p[j]:
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Let x = p[i], y = p[j].
    # Terms involving i or j change.
    # New p[i] = y, p[j] = x.
    # 
    # Change in cost = 
    #   - sum_{v != i, v != j} a[i,v] b[x, p[v]]  (old i-x)
    #   - sum_{v != i, v != j} a[j,v] b[y, p[v]]  (old j-y)
    #   - a[i,j] b[x,y]                            (old i-j)
    #   - a[j,i] b[y,x]                            (old j-i)
    #   + sum_{v != i, v != j} a[i,v] b[y, p[v]]  (new i-y)
    #   + sum_{v != i, v != j} a[j,v] b[x, p[v]]  (new j-x)
    #   + a[i,j] b[y,x]                            (new i-j)
    #   + a[j,i] b[x,y]                            (new j-i)
    #
    # Note b[x,y] = b[y,x].
    # a[i,j] and a[j,i] are fixed.
    # 
    # Let's group terms:
    # Delta = 
    #   sum_{v != i, v != j} a[i,v] (b[y, p[v]] - b[x, p[v]])
    #   + sum_{v != i, v != j} a[j,v] (b[x, p[v]] - b[y, p[v]])
    #   + (a[i,j] + a[j,i]) * (b[y,x] - b[x,y])  -> This is 0 if b is symmetric.
    # 
    # If b is symmetric (standard QAP), the diagonal cross terms cancel out.
    # So Delta = 
    #   sum_{v != i, v != j} (a[i,v] - a[j,v]) * (b[y, p[v]] - b[x, p[v]])
    #
    # This can be computed in O(n) instead of O(n^2) for cost evaluation.
    # We need to access b[y, :] and b[x, :] and a[i, :] and a[j, :].
    
    # Pre-compute nothing specifically, but ensure arrays are contiguous.
    a_c = np.ascontiguousarray(a)
    b_c = np.ascontiguousarray(b)
    
    def delta_swap(p, i, j):
        """
        Calculate cost change if we swap p[i] and p[j].
        p[i] = x, p[j] = y.
        New p[i] = y, p[j] = x.
        """
        x = p[i]
        y = p[j]
        
        # Vectorized calculation
        # We need sum_{v != i, v != j} (a[i,v] - a[j,v]) * (b[y, p[v]] - b[x, p[v]])
        
        # Create mask for v != i, v != j
        # Instead of mask, we can compute full sum and subtract i and j terms.
        
        # Full sum term:
        # S = sum_v (a[i,v] - a[j,v]) * (b[y, p[v]] - b[x, p[v]])
        
        # Let diff_a = a[i, :] - a[j, :]
        # Let diff_b = b[y, p] - b[x, p]
        
        diff_a = a_c[i, :] - a_c[j, :]
        diff_b = b_c[y, p] - b_c[x, p]
        
        total = np.dot(diff_a, diff_b)
        
        # Subtract contributions from v=i and v=j
        # v=i: a[i,i]-a[j,i] * (b[y, p[i]] - b[x, p[i]])
        # p[i] = x
        # term_i = (a_c[i,i] - a_c[j,i]) * (b_c[y,x] - b_c[x,x])
        
        # v=j: a[i,j]-a[j,j] * (b[y, p[j]] - b[x, p[j]])
        # p[j] = y
        # term_j = (a_c[i,j] - a_c[j,j]) * (b_c[y,y] - b_c[x,y])
        
        term_i = (a_c[i, i] - a_c[j, i]) * (b_c[y, x] - b_c[x, x])
        term_j = (a_c[i, j] - a_c[j, j]) * (b_c[y, y] - b_c[x, y])
        
        return total - term_i - term_j

    # Local Search Loop
    # We use a simple steepest descent with random restarts if stuck.
    # Given the time limit, we will do a single long run with random perturbations if no improvement.
    
    rng = np.random.default_rng(42)
    no_improve_count = 0
    max_no_improve = 1000 # Reset perturbation if stuck
    
    while time.time() < deadline:
        improved = False
        
        # Try a few random swaps
        # To be fast, we don't do full steepest descent (O(n^2) swaps per step)
        # We do random swaps and accept if better.
        # This is essentially Simulated Annealing with T=0 (Greedy).
        
        for _ in range(100): # 100 random trials per "iteration"
            i, j = rng.choice(n, 2, replace=False)
            
            # Check if i, j are distinct
            if i == j:
                continue
                
            d = delta_swap(p_init, i, j)
            
            if d < 0:
                # Accept swap
                p_init[i], p_init[j] = p_init[j], p_init[i]
                current_cost += d
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p_init.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            if time.time() > deadline - 5.0:
                break
                
        if not improved:
            no_improve_count += 100
            
        # If stuck for a long time, perturb the solution
        if no_improve_count > max_no_improve:
            # Perform a random 3-swap or 5-swap to escape local minimum
            # This is a "kick"
            indices = rng.choice(n, size=min(5, n), replace=False)
            for k in range(len(indices)-1):
                i, j = indices[k], indices[k+1]
                # Don't check cost, just swap to escape
                p_init[i], p_init[j] = p_init[j], p_init[i]
            
            # Recalculate cost (expensive, but rare)
            current_cost = calc_cost(p_init)
            no_improve_count = 0
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
