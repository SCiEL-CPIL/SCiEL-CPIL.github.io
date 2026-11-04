"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Helper to calculate exact cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
    
    # Helper to get a random permutation
    def random_perm():
        p = np.arange(n)
        np.random.shuffle(p)
        return p

    # 1. Initialization: Spectral Relaxation
    # This often provides a much better starting point than simple rank matching.
    # We align the principal components of the flow and distance matrices.
    try:
        # Eigendecomposition for symmetric matrices
        # For large n, using eigh is O(n^3) but n=150 is small enough.
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Sort eigenvalues in descending order to align "most important" modes
        idx_a = np.argsort(-w_a)
        idx_b = np.argsort(-w_b)
        
        v_a = v_a[:, idx_a]
        v_b = v_b[:, idx_b]
        
        # Project facilities onto location eigenvectors
        # The assignment matrix X should approximate V_a V_b^T
        # Since we need a permutation, we solve the linear assignment problem on this matrix
        # Cost matrix for Hungarian: we want to maximize sum X[i, j] * (V_a[i, k] * V_b[j, k])
        # This is equivalent to minimizing - (V_a @ V_b.T)
        # However, linear_sum_assignment minimizes cost.
        # So cost_matrix = - (V_a @ V_b.T)
        
        # Note: V_a and V_b are n x n.
        # The objective is sum_{i, j, k} a[i, j] b[p[i], p[j]]
        # Spectral relaxation suggests p should align the eigenvectors.
        # Let's construct the linear assignment cost matrix.
        # The term to maximize is trace(V_a^T X V_b) = sum_{i, j} X[i, j] (V_a^T V_b)[i, j]
        # So we minimize the negative of this.
        
        cost_matrix = - (v_a.T @ v_b)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
    except Exception:
        # Fallback to rank matching if spectral fails
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        heavy_order = np.argsort(-a_sum, kind="stable")
        central_order = np.argsort(b_sum, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # 2. Iterated Linear Assignment (ILAP) with Local Search
    # Strategy:
    # - Loop until time runs out.
    # - In each iteration:
    #   a. Linearize the QAP: Construct C where C[i, j] = sum_k (a[i, k] + a[k, i]) * b[j, p[k]]
    #      This approximates the marginal cost of placing facility i at location j.
    #   b. Solve the Linear Assignment Problem (LAP) using Hungarian to get a new permutation.
    #   c. Perform a few random swaps (local search) to refine the solution.
    #   d. Update best if improved.
    #   e. Occasionally perturb the current solution to escape local minima.

    # Precompute symmetric A if needed for speed, though a is likely symmetric.
    # a_sym = (a + a.T) / 2.0
    
    # We will use a "greedy" delta update for local search to be fast.
    # But first, let's implement the ILAP loop.
    
    # Hungarian algorithm is O(n^3). For n=150, it's very fast (milliseconds).
    # We can run many iterations.
    
    iter_count = 0
    
    while time.time() < deadline:
        # --- Phase 1: Linearization and Hungarian ---
        # Construct the cost matrix for the linear assignment problem.
        # The QAP cost is sum_{i, j} a[i, j] b[p[i], p[j]].
        # Linearization: For a fixed assignment p, the "effective" cost of assigning i to location l
        # is approximated by the sum of interactions with all other facilities k.
        # Cost(i, l) = sum_{k != i} a[i, k] b[l, p[k]]
        
        # We can compute this matrix M where M[i, l] = sum_k a[i, k] b[l, p[k]].
        # M = a @ b[:, p]  (Wait, dimensions)
        # a is n x n. b[:, p] is n x n (columns permuted).
        # M[i, l] = sum_k a[i, k] * b[l, p[k]]
        # This is (a @ b[:, p]).T ?
        # Let's check indices.
        # (a @ B_p)[i, k] = sum_m a[i, m] B_p[m, k]. This is not right.
        
        # We want M[i, l] = sum_k a[i, k] b[l, p[k]].
        # Let B_p be the matrix b with columns permuted by p? No, b[p, p] is the permuted matrix.
        # Let's use the current assignment p to build the cost matrix.
        
        # Efficient computation:
        # M[i, l] = sum_k a[i, k] b[l, p[k]]
        # This is equivalent to (a @ b.T[:, p]).T ?
        # Let's just do:
        # b_perm = b[:, p]  # This is n x n. b_perm[l, k] = b[l, p[k]]
        # M = a @ b_perm
        # M[i, k] = sum_m a[i, m] b_perm[m, k] = sum_m a[i, m] b[m, p[k]]. 
        # This is NOT what we want. We want b[l, p[k]].
        
        # Correct formula:
        # M[i, l] = sum_k a[i, k] b[l, p[k]]
        # Let B_p be the matrix where B_p[k, l] = b[l, p[k]].
        # Then M[i, l] = sum_k a[i, k] B_p[k, l].
        # So M = a @ B_p.
        
        # How to get B_p?
        # b is n x n.
        # b[p, :] is rows permuted.
        # b[:, p] is cols permuted.
        # We need b[l, p[k]].
        # If we take b.T, then b.T[p, :] is rows of b.T permuted by p, which is cols of b permuted by p?
        # b.T[p, l] = b[l, p].
        # So B_p = b.T[:, p].T ?
        # Let's stick to vectorized numpy ops which are clear.
        
        # b_perm_k_l = b[:, p]  # b_perm_k_l[k, l] = b[k, p[l]] -- No, b[:, p] means columns are permuted.
        # b[:, p][k, l] = b[k, p[l]].
        # We want b[l, p[k]].
        # This is the transpose of b[:, p].
        # So B_p = b[:, p].T
        # B_p[k, l] = b[:, p].T[k, l] = b[:, p][l, k] = b[l, p[k]]. Correct.
        
        B_p = b[:, p].T
        M = a @ B_p
        
        # M[i, l] is the cost contribution of i to l, given other assignments.
        # We want to minimize sum_i M[i, p_new[i]].
        # So we solve LAP on M.
        
        # To avoid trivial solutions or getting stuck, we can add a small random noise
        # or use the current best to guide. But standard ILAP just solves M.
        
        # Optimization: We don't need to solve LAP from scratch if we just want to improve.
        # But LAP is fast.
        
        try:
            row_ind, col_ind = linear_sum_assignment(M)
            p_new = np.empty(n, dtype=np.int64)
            p_new[row_ind] = col_ind
        except Exception:
            continue
            
        cost_new = calc_cost(p_new)
        
        # --- Phase 2: Local Refinement (Greedy Swaps) ---
        # If the new solution is better, or even if it's close, we try to refine it.
        # We perform a limited number of best-improvement swaps.
        
        # We'll use a simple loop of random swaps to refine p_new, keeping the best.
        # This helps escape the "linearization" local minima.
        
        p_curr = p_new.copy()
        cost_curr = cost_new
        
        # Refine with 50 random swaps
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            u = p_curr[i]
            v = p_curr[j]
            
            # Calculate delta for swap i, j in p_curr
            # delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, p_curr[k]] - b[u, p_curr[k]])
            
            diff_a = a[i, :] - a[j, :]
            diff_b = b[v, p_curr] - b[u, p_curr]
            
            dot_full = np.dot(diff_a, diff_b)
            
            term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
            term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
            
            sum_k = dot_full - term_i - term_j
            delta = 2.0 * sum_k
            
            if delta < 0:
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                cost_curr += int(delta)
                
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr.copy()
            
        # Update current p for next iteration. 
        # Usually, we keep the best p found so far to build the next linearization,
        # or we keep the p from the LAP solution. 
        # Using the best p ensures we always linearize around the best known structure.
        p = best_p.copy()
        
        iter_count += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
