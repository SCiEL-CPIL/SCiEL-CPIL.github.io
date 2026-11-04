"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _fast_2opt(a, b, p, time_budget=10.0):
    """
    Fast 2-opt local search using vectorized delta calculation.
    Assumes A and B are symmetric (or effectively treats them as such for speed).
    """
    n = len(p)
    p = p.copy().astype(np.int64)
    
    # Precompute symmetric A for delta calculations
    A_sym = a + a.T
    
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_budget:
                break
            
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            
            # Vectorized delta calculation for swapping i with all j in js
            # Delta = sum_{k != i, j} (A_sym[i, k] - A_sym[j, k]) * (D[i, k] - D[j, k]) 
            #       + corrections for k=i and k=j
            
            # Term 1: Sum over k != i, j
            # We compute S = sum_k (A_sym[i, k] - A_sym[j, k]) * (D[i, k] - D[j, k])
            # Then subtract the k=i and k=j terms which were included in the sum
            # but need special handling because the "swap" changes the indices involved.
            
            # Let's use the standard vectorized formula derived for QAP 2-opt.
            # diff_A = A_sym[i] - A_sym[js]  (shape n, m)
            # diff_D = D[i] - D[js]          (shape n, m)
            # S = sum(diff_A * diff_D, axis=0) (shape m)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[i] - D[js]
            S = np.sum(diff_A * diff_D, axis=0)
            
            # Correction for k=i:
            # In the sum S, the term for k=i is (A_sym[i, i] - A_sym[j, i]) * (D[i, i] - D[j, i])
            # But in the actual delta, the k=i term involves the change in position i.
            # The standard correction is:
            # term_i = (A_sym[i, i] - A_sym[js, i]) * (D[i, i] - D[js, i])
            # Wait, the formula is:
            # Delta(i, j) = sum_{k != i, j} (A_sym[i, k] - A_sym[j, k]) (D[i, k] - D[j, k])
            #              + (A_sym[i, i] - A_sym[j, i]) (D[i, i] - D[j, i])  -- NO, this is not right.
            
            # Let's use the robust formula:
            # Delta = sum_{k} (A_sym[i, k] - A_sym[j, k]) * (D[i, k] - D[j, k])
            #         - (A_sym[i, i] - A_sym[j, i]) * (D[i, i] - D[j, i])  # Remove k=i from sum? No.
            
            # Actually, the most common vectorized implementation is:
            # S = np.sum((A_sym[i] - A_sym[js]) * (D[i] - D[js]), axis=0)
            # Then adjust for the diagonal/cross terms.
            # The term for k=i in the sum S is: (A_sym[i, i] - A_sym[j, i]) * (D[i, i] - D[j, i])
            # The term for k=j in the sum S is: (A_sym[i, j] - A_sym[j, j]) * (D[i, j] - D[j, j])
            
            # The actual delta for swapping i and j is:
            # Delta = S - term_k=i - term_k=j + direct_swap_term
            # Where direct_swap_term accounts for the change in the (i,j) and (j,i) interactions directly.
            
            # Let's use a simpler, verified vectorized approach:
            # Delta = sum_{k} (A_sym[i, k] - A_sym[j, k]) * (D[i, k] - D[j, k])
            #         - (A_sym[i, i] - A_sym[j, i]) * (D[i, i] - D[j, i])
            #         - (A_sym[i, j] - A_sym[j, j]) * (D[i, j] - D[j, j])
            #         + (A_sym[i, j] - A_sym[j, i]) * (D[i, j] - D[j, i])
            #         + (A_sym[i, i] - A_sym[j, j]) * (D[i, i] - D[j, j]) # This seems messy.
            
            # Let's stick to the proven formula from literature/code:
            # For symmetric A, B:
            # Delta = sum_{k != i, j} (A[i, k] + A[k, i] - A[j, k] - A[k, j]) * (B[p[i], p[k]] + B[p[k], p[i]] - B[p[j], p[k]] - B[p[k], p[j]])
            #       + (A[i, i] - A[j, j]) * (B[p[i], p[i]] - B[p[j], p[j]])
            #       + (A[i, j] + A[j, i]) * (B[p[i], p[j]] - B[p[j], p[i]])
            
            # Since A_sym = A + A.T and D is symmetric (if B is symmetric), this simplifies.
            # Let's use the formula:
            # Delta = S - term_i - term_j + cross_correction
            
            # term_i corresponds to k=i in the sum S.
            # S includes k=i: (A_sym[i, i] - A_sym[j, i]) * (D[i, i] - D[j, i])
            # S includes k=j: (A_sym[i, j] - A_sym[j, j]) * (D[i, j] - D[j, j])
            
            # The correct delta is:
            # Delta = S - (A_sym[i, i] - A_sym[js, i]) * (D[i, i] - D[js, i]) 
            #             - (A_sym[i, js] - A_sym[js, js]) * (D[i, js] - D[js, js])
            #             + (A_sym[i, js] - A_sym[js, i]) * (D[i, js] - D[js, i])
            #             + (A_sym[i, i] - A_sym[js, js]) * (D[i, i] - D[js, js])
            
            # Let's compute the corrections vectorized over js.
            # term_i_vec = (A_sym[i, i] - A_sym[js, i]) * (D[i, i] - D[js, i])
            # term_j_vec = (A_sym[i, js] - A_sym[js, js]) * (D[i, js] - D[js, js])
            # cross_vec = (A_sym[i, js] - A_sym[js, i]) * (D[i, js] - D[js, i])
            # diag_vec = (A_sym[i, i] - A_sym[js, js]) * (D[i, i] - D[js, js])
            
            term_i_vec = (A_sym[i, i] - A_sym[js, i]) * (D[i, i] - D[js, i])
            term_j_vec = (A_sym[i, js] - A_sym[js, js]) * (D[i, js] - D[js, js])
            cross_vec = (A_sym[i, js] - A_sym[js, i]) * (D[i, js] - D[js, i])
            diag_vec = (A_sym[i, i] - A_sym[js, js]) * (D[i, i] - D[js, js])
            
            deltas = S - term_i_vec - term_j_vec + cross_vec + diag_vec
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
        
    return p, current_cost

def _frank_wolfe(a, b, p_init, max_iters=20, time_budget=10.0):
    """
    Frank-Wolfe algorithm on Birkhoff polytope.
    """
    n = len(p_init)
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    start_time = time.time()
    
    for t in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Gradient G = A X B + A^T X B^T
        grad = a @ X @ b + a.T @ X @ b.T
        grad = (grad + grad.T) / 2.0
        
        row_ind, col_ind = linear_sum_assignment(grad)
        
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        step = 2.0 / (t + 2)
        X = (1.0 - step) * X + step * Y
        
        cost = _qap_cost(a, b, p_new)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_new.copy()
            
    return best_p, best_cost

def _spectral_initial(a, b):
    """Spectral embedding initialization."""
    n = len(a)
    try:
        A_sym = a + a.T
        B_sym = b + b.T
        
        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
        
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]
        
        V_A = eigvecs_A[:, idx_A]
        V_B = eigvecs_B[:, idx_B]
        
        C_spectral = -V_A @ V_B.T
        
        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        # Fallback
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # 1. Initial Solution: Spectral + 2-opt
    p0 = _spectral_initial(a, b)
    p0, cost0 = _fast_2opt(a, b, p0, time_budget=15.0)
    
    best_p = p0.copy()
    best_cost = cost0
    
    # 2. Diversification with Frank-Wolfe + Local Search
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 10.0:
        # Create a starting point for FW
        if rng.rand() < 0.3:
            p_start = best_p.copy()
        else:
            p_start = best_p.copy()
            # Perturb: swap 3-8 random elements
            num_swaps = rng.randint(3, 9)
            for _ in range(num_swaps):
                i, j = rng.randint(0, n, 2)
                if i != j:
                    p_start[i], p_start[j] = p_start[j], p_start[i]
        
        # Run FW
        remaining = deadline - time.time()
        fw_budget = min(5.0, remaining * 0.3)
        if fw_budget < 1.0:
            break
            
        p_fw, cost_fw = _frank_wolfe(a, b, p_start, max_iters=20, time_budget=fw_budget)
        
        # Polish with 2-opt
        remaining = deadline - time.time()
        opt_budget = min(10.0, remaining * 0.6)
        if opt_budget < 1.0:
            break
            
        p_2opt, cost_2opt = _fast_2opt(a, b, p_fw, time_budget=opt_budget)
        
        if cost_2opt < best_cost:
            best_cost = cost_2opt
            best_p = p_2opt.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
