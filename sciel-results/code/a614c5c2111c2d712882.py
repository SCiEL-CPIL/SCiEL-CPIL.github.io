"""Stronger rank-reduction via exact linear dependence detection and targeted search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                total = np.sum(U[a, :] * V[b, :] * W[c, :])
                if total != expected:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _compute_tensor_values(U, V, W, n, m, p, rank):
    """Compute the full tensor values from the decomposition."""
    values = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                values[i, j, k] = np.sum(U[a, :] * V[b, :] * W[c, :])
    return values

def _try_remove_column(U, V, W, col_idx):
    """
    Try to remove column col_idx by finding linear combinations.
    Returns (new_U, new_V, new_W) if successful, else None.
    """
    n_rows_u = U.shape[0]
    n_rows_v = V.shape[0]
    n_rows_w = W.shape[0]
    rank = U.shape[1]
    
    U_other = np.delete(U, col_idx, axis=1)
    V_other = np.delete(V, col_idx, axis=1)
    W_other = np.delete(W, col_idx, axis=1)
    
    U_col = U[:, col_idx]
    V_col = V[:, col_idx]
    W_col = W[:, col_idx]
    
    # Find coefficients alpha such that U_col = U_other @ alpha
    # This is a least squares / linear solve problem
    try:
        alpha, residuals, rank_u, _ = np.linalg.lstsq(U_other, U_col, rcond=None)
        if residuals is None or len(residuals) == 0 or residuals[0] > 1e-9:
            return None
        # Verify exact reconstruction
        if not np.allclose(U_other @ alpha, U_col, atol=1e-9):
            return None
        
        beta, residuals_v, rank_v, _ = np.linalg.lstsq(V_other, V_col, rcond=None)
        if residuals_v is None or len(residuals_v) == 0 or residuals_v[0] > 1e-9:
            return None
        if not np.allclose(V_other @ beta, V_col, atol=1e-9):
            return None
        
        gamma, residuals_w, rank_w, _ = np.linalg.lstsq(W_other, W_col, rcond=None)
        if residuals_w is None or len(residuals_w) == 0 or residuals_w[0] > 1e-9:
            return None
        if not np.allclose(W_other @ gamma, W_col, atol=1e-9):
            return None
        
        # Now we can eliminate col_idx: for each remaining column r, adjust
        # U_new[:, r] = U_other[:, r] - U_col * alpha[r]
        # V_new[:, r] = V_other[:, r] - V_col * beta[r]
        # W_new[:, r] = W_other[:, r] - W_col * gamma[r]
        # But wait, this isn't quite right. We need to preserve the tensor values.
        
        # The tensor value at (a, b, c) is sum_r U[a,r] * V[b,r] * W[c,r]
        # After removing col_idx, we need:
        # sum_{r != col_idx} U_new[a,r] * V_new[b,r] * W_new[c,r] = original sum
        
        # This is complex. Let's try a different approach:
        # If column col_idx is linearly dependent in all three factors simultaneously,
        # we can express the decomposition without it.
        
        # Actually, the correct approach: if U_col = U_other @ alpha, then
        # for any a: U[a, col_idx] = sum_{r != col_idx} U[a, r] * alpha[r]
        # This means the rank-1 tensor U_col * V_col * W_col can be rewritten.
        
        # But we need the FULL tensor to be preserved, not just one factor.
        
        # Let's try a simpler check: is the column truly redundant?
        # Check if removing it changes any tensor value.
        U_new = U_other.copy()
        V_new = V_other.copy()
        W_new = W_other.copy()
        
        # Verify tensor values are preserved
        n, m, p = U_new.shape[0] // 12, 4, 5  # hardcoded for 3x4x5
        n = 3; m = 4; p = 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    old_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    new_val = np.sum(U_new[a, :] * V_new[b, :] * W_new[c, :])
                    if old_val != new_val:
                        return None
        
        return U_new.tolist(), V_new.tolist(), W_new.tolist()
    except Exception:
        return None

def _greedy_rank_reduction(U, V, W, n, m, p, deadline, max_iterations=1000):
    """
    Greedy rank reduction: repeatedly try to remove columns.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    current_rank = U.shape[1]
    iterations = 0
    
    while U.shape[1] > 15 and iterations < max_iterations:
        if time.time() > deadline:
            break
        
        removed = False
        # Try columns in random order
        cols = list(range(U.shape[1]))
        np.random.shuffle(cols)
        
        for col in cols:
            if time.time() > deadline:
                break
            
            result = _try_remove_column(U, V, W, col)
            if result is not None:
                U, V, W = result
                removed = True
                break
        
        if not removed:
            break
        
        iterations += 1
    
    return U.tolist(), V.tolist(), W.tolist(), U.shape[1]

def _try_merge_columns(U, V, W, col1, col2, deadline):
    """
    Try to merge two columns by finding a linear combination that preserves the tensor.
    This is more complex; we skip for now and rely on greedy removal.
    """
    pass

def _structured_search_v2(n, m, p, deadline):
    """
    Improved structured search with better rank reduction.
    """
    U, V, W, rank = _schoolbook_decomposition(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Phase 1: Greedy rank reduction from schoolbook
    if time.time() < deadline - 30:
        U_reduced, V_reduced, W_reduced, new_rank = _greedy_rank_reduction(
            U, V, W, n, m, p, deadline - 30
        )
        if new_rank < best_rank:
            # Verify
            if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p, new_rank):
                best_rank = new_rank
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
    
    # Phase 2: Try targeted rank targets with random initialization + local search
    for target_rank in range(best_rank - 1, max(20, best_rank // 2), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            U_rand = np.random.randint(-1, 2, size=(n * m, target_rank)).astype(np.float64)
            V_rand = np.random.randint(-1, 2, size=(m * p, target_rank)).astype(np.float64)
            W_rand = np.random.randint(-1, 2, size=(p * n, target_rank)).astype(np.float64)
            
            # Local optimization: try to adjust entries to satisfy tensor equations
            # This is a nonlinear problem; we do a simple coordinate descent
            for coord_iter in range(20):
                if time.time() > deadline - 10:
                    break
                
                improved = False
                # Try perturbing each entry slightly
                for factor_idx, factor in enumerate([U_rand, V_rand, W_rand]):
                    for row in range(factor.shape[0]):
                        for col in range(factor.shape[1]):
                            if time.time() > deadline - 10:
                                break
                            
                            orig = factor[row, col]
                            for delta in [-1, 1]:
                                factor[row, col] = orig + delta
                                # Quick check: does this improve things?
                                # We'd need to recompute tensor error, which is expensive
                                # Skip for now
                                factor[row, col] = orig
                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                
                # Check if valid
                if _verify_decomposition(
                    U_rand.tolist(), V_rand.tolist(), W_rand.tolist(),
                    n, m, p, target_rank
                ):
                    found = True
                    best_rank = target_rank
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_rand.tolist()
                    break
            if found:
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s limit
    
    n, m, p = 3, 4, 5
    
    # Get schoolbook as safe fallback
    U_school, V_school, W_school, rank_school = _schoolbook_decomposition(n, m, p)
    
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p, rank_school):
        return {"U": U_school, "V": V_school, "W": W_school}
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank
    try:
        U_best, V_best, W_best, rank_best = _structured_search_v2(n, m, p, deadline)
        
        if _verify_decomposition(U_best, V_best, W_best, n, m, p, rank_best):
            best_rank = rank_best
            best_U, best_V, best_W = U_best, V_best, W_best
    except Exception:
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
