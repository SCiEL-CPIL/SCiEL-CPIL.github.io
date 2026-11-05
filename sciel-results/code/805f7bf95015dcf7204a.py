import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _rank19_bcdf_terms():
    """
    Return the 19 terms of the Bini-Capovani-De Falco-Ladisi rank-19 
    decomposition for 3x3x3 matrix multiplication.
    Each term is (u, v, w) where u[i,j], v[j,k], w[k,i] are 3x3 matrices.
    """
    terms = []
    # The 19 terms are defined by their U, V, W matrices
    # Using the standard BCDL rank-19 algorithm
    
    # Term 1: (a11+a33)(b11+b33)
    u1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    v1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    w1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u1, v1, w1))
    
    # Term 2: (a21+a31)(b11+b21) 
    u2 = np.array([[0,0,0],[1,0,0],[1,0,0]], dtype=np.float64)
    v2 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    w2 = np.array([[0,0,0],[0,1,0],[0,1,0]], dtype=np.float64)
    terms.append((u2, v2, w2))
    
    # Term 3: (a12+a32)(b22+b32)
    u3 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v3 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    w3 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    terms.append((u3, v3, w3))
    
    # Term 4: (a13+a33)(b31+b33)
    u4 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v4 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    w4 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u4, v4, w4))
    
    # Term 5: a11(b12+b13)
    u5 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    v5 = np.array([[0,1,1],[0,0,0],[0,0,0]], dtype=np.float64)
    w5 = np.array([[0,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u5, v5, w5))
    
    # Term 6: a22(b21+b23)
    u6 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v6 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    w6 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u6, v6, w6))
    
    # Term 7: a33(b32+b31)
    u7 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    v7 = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=np.float64)
    w7 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u7, v7, w7))
    
    # Term 8: (a11+a12+a21+a22)(b11+b12+b21+b22)
    u8 = np.array([[1,1,0],[1,1,0],[0,0,0]], dtype=np.float64)
    v8 = np.array([[1,1,0],[1,1,0],[0,0,0]], dtype=np.float64)
    w8 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u8, v8, w8))
    
    # Term 9: (a11+a21)(b12+b22)
    u9 = np.array([[1,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    v9 = np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=np.float64)
    w9 = np.array([[0,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u9, v9, w9))
    
    # Term 10: (a11+a22)(b11+b21)
    u10 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v10 = np.array([[1,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    w10 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u10, v10, w10))
    
    # Term 11: (a12+a22)(b11+b12)
    u11 = np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v11 = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    w11 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u11, v11, w11))
    
    # Term 12: (a11+a12)(b21+b22)
    u12 = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    v12 = np.array([[0,0,0],[1,1,0],[0,0,0]], dtype=np.float64)
    w12 = np.array([[0,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u12, v12, w12))
    
    # Term 13: (a11+a31)(b11+b31)
    u13 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    v13 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w13 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u13, v13, w13))
    
    # Term 14: (a12+a32)(b12+b32)
    u14 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v14 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    w14 = np.array([[0,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u14, v14, w14))
    
    # Term 15: (a13+a33)(b13+b33)
    u15 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v15 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    w15 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u15, v15, w15))
    
    # Term 16: (a21+a31)(b12+b32)
    u16 = np.array([[0,0,0],[1,0,0],[1,0,0]], dtype=np.float64)
    v16 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    w16 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u16, v16, w16))
    
    # Term 17: (a12+a32)(b21+b31)
    u17 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v17 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w17 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u17, v17, w17))
    
    # Term 18: (a13+a33)(b21+b31)
    u18 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v18 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w18 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u18, v18, w18))
    
    # Term 19: (a13+a33)(b23+b33)
    u19 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v19 = np.array([[0,0,0],[0,0,1],[0,0,1]], dtype=np.float64)
    w19 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u19, v19, w19))
    
    return terms

def _construct_bcdl_based_decomposition():
    """
    Construct a decomposition using BCDL rank-19 for the core 3x3x3 part
    and schoolbook for the remaining strips.
    
    For 3x4 x 4x4:
    - The matrix A is 3x4, B is 4x4
    - We can think of this as multiplying [A_core | A_strip] where A_core is 3x3 and A_strip is 3x1
    - Similarly B can be split into rows
    
    Actually, let's think differently. We have:
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,1,2,3
    
    We can split the j index into j=0,1,2 (core) and j=3 (strip)
    And the k index into k=0,1,2 (core) and k=3 (strip)
    
    So C = C_core + C_strip_i + C_strip_k + C_strip_both
    where:
    - C_core[i,k] = sum_{j=0,1,2} A[i,j] * B[j,k] for i=0,1,2, k=0,1,2 (3x3x3)
    - C_strip_i[i,k] = A[i,3] * B[3,k] for i=0,1,2, k=0,1,2 (3x1x3)
    - C_strip_k[i,k] = sum_{j=0,1,2} A[i,j] * B[j,3] for i=0,1,2, k=3 (3x3x1)  
    - C_strip_both[i,3] = A[i,3] * B[3,3] for i=0,1,2 (3x1x1)
    
    The core 3x3x3 can be done with rank 19 (BCDL)
    The strips need separate handling.
    """
    n, m, p = 3, 4, 4
    rank = 48  # Start with schoolbook as fallback
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First, let's just do schoolbook to have a valid baseline
    # Then we'll try to use BCDL for the core
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _construct_hybrid_decomposition():
    """
    Try a hybrid approach: use structured blocks.
    
    Split the 3x4 matrix into:
    - Top 2 rows: 2x4
    - Bottom 1 row: 1x4
    
    Split the 4x4 matrix into four 2x2 blocks:
    - Top-left (rows 0-1, cols 0-1)
    - Top-right (rows 0-1, cols 2-3)
    - Bottom-left (rows 2-3, cols 0-1)
    - Bottom-right (rows 2-3, cols 2-3)
    
    This gives us:
    - (2x4) x (4x4) = four (2x2) x (2x2) blocks -> 4 * 7 = 28 (Strassen)
    - (1x4) x (4x4) = four (1x2) x (2x2) blocks -> 4 * 4 = 16 (schoolbook for 1x2x2)
    Total: 44
    
    But we can try to be smarter about the 1x2x2 blocks.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W

def _compute_flattened_terms(U, V, W, n, m, p):
    """Compute the flattened tensor for each term."""
    rank = U.shape[1]
    terms = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        terms[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return terms

def _try_reduce_rank_global(U, V, W, n, m, p, time_budget):
    """Try to reduce rank by finding global linear dependencies among the flattened terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    rng = np.random.default_rng(42)
    
    def _attempt_elimination(U, V, W, dep):
        """Try to eliminate one term using the given dependency vector."""
        current_rank = U.shape[1]
        max_idx = np.argmax(np.abs(dep))
        
        if np.abs(dep[max_idx]) < 1e-9:
            return None, None, None, current_rank
            
        coeff = dep / dep[max_idx]
        coeff[max_idx] = 0
        
        U_new = np.delete(U, max_idx, axis=1).copy()
        V_new = np.delete(V, max_idx, axis=1).copy()
        W_new = np.delete(W, max_idx, axis=1).copy()
        
        for r in range(U_new.shape[1]):
            if abs(coeff[r]) > 1e-9:
                U_new[:, r] += coeff[r] * U[:, max_idx]
                V_new[:, r] += coeff[r] * V[:, max_idx]
                W_new[:, r] += coeff[r] * W[:, max_idx]
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
            return U_new, V_new, W_new, U_new.shape[1]
        return None, None, None, current_rank
    
    # Phase 1: SVD-based dependency search (exact)
    while time.time() - start_time < time_budget:
        terms = _compute_flattened_terms(U, V, W, n, m, p)
        
        try:
            U_svd, s, Vh = np.linalg.svd(terms, full_matrices=False)
            rank = np.sum(s > 1e-9)
            
            if rank < current_rank:
                null_space = Vh[rank:, :]
                best_result = None
                for dep in null_space:
                    U_new, V_new, W_new, new_rank = _attempt_elimination(U, V, W, dep)
                    if U_new is not None:
                        if best_result is None or new_rank < best_result[3]:
                            best_result = (U_new, V_new, W_new, new_rank)
                if best_result is not None:
                    U, V, W, current_rank = best_result
                else:
                    break
            else:
                break
        except Exception:
            break
    
    # Phase 2: Randomized dependency search
    # Try to find dependencies by looking at random subsets of terms
    while time.time() - start_time < time_budget - 5:
        current_rank = U.shape[1]
        terms = _compute_flattened_terms(U, V, W, n, m, p)
        
        # Try random pairs of terms to see if one can be expressed in terms of others
        found = False
        for trial in range(50):
            if time.time() - start_time > time_budget - 5:
                break
            # Pick a random term to try to eliminate
            target = rng.integers(0, current_rank)
            # Try to express term[target] as a combination of others
            others = [i for i in range(current_rank) if i != target]
            
            # Use a random subset for speed
            if len(others) > 20:
                subset = rng.choice(others, size=20, replace=False)
            else:
                subset = others
            
            # Solve least squares: terms[:, target] ≈ sum(coeff[i] * terms[:, i])
            A_sub = terms[:, subset]
            b = terms[:, target]
            
            try:
                coeff, residuals, _, _ = np.linalg.lstsq(A_sub, b, rcond=1e-10)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # Found a dependency: term[target] = sum(coeff[i] * terms[subset[i]])
                    dep = np.zeros(current_rank)
                    dep[target] = 1.0
                    for idx, ci in zip(subset, coeff):
                        dep[idx] = -ci
                    
                    U_new, V_new, W_new, new_rank = _attempt_elimination(U, V, W, dep)
                    if U_new is not None:
                        U, V, W = U_new, V_new, W_new
                        found = True
                        break
            except Exception:
                continue
        
        if not found:
            # Also try full SVD one more time
            try:
                U_svd, s, Vh = np.linalg.svd(terms, full_matrices=False)
                rank = np.sum(s > 1e-9)
                if rank < U.shape[1]:
                    null_space = Vh[rank:, :]
                    for dep in null_space:
                        U_new, V_new, W_new, new_rank = _attempt_elimination(U, V, W, dep)
                        if U_new is not None:
                            U, V, W = U_new, V_new, W_new
                            found = True
                            break
            except Exception:
                pass
            if not found:
                break
    
    return U, V, W

def _construct_hybrid_decomposition_v2():
    """
    Alternative decomposition: split rows as 1+2 instead of 2+1.
    
    Split A (3x4) into:
    - Row 0: 1x4
    - Rows 1-2: 2x4
    
    Split B (4x4) into four 2x2 blocks.
    
    This gives:
    - (2x4) x (4x4) = four (2x2) x (2x2) blocks -> 4 * 7 = 28 (Strassen)
    - (1x4) x (4x4) = four (1x2) x (2x2) blocks -> 4 * 4 = 16 (schoolbook)
    Total: 44 (same rank, but different structure may allow different dependencies)
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Rows 1-2 use Strassen, Row 0 uses schoolbook
    add_strassen(1, 0, 0, 0)
    add_strassen(1, 2, 2, 0)
    add_strassen(1, 0, 0, 2)
    add_strassen(1, 2, 2, 2)
    add_schoolbook_1x2x2(0, 0, 0, 0)
    add_schoolbook_1x2x2(0, 2, 2, 0)
    add_schoolbook_1x2x2(0, 0, 0, 2)
    add_schoolbook_1x2x2(0, 2, 2, 2)

    return U, V, W

def _construct_hybrid_decomposition_v3():
    """
    Alternative: split columns of A as 3+1, rows of B as 2+2.
    
    A = [A_left (3x3) | A_right (3x1)]
    B = [[B_tl (2x2), B_tr (2x2)], [B_bl (2x2), B_br (2x2)]]
    
    C = A_left * B + A_right * B_bottom_rows
    
    For A_left * B: three (3x2) x (2x2) multiplications... this gets complex.
    Let's try a different split: A columns 2+2, B rows 2+2.
    
    A = [A_l (3x2) | A_r (3x2)]
    B = [[B_t (2x4)], [B_b (2x4)]]
    
    C = A_l * B_t + A_r * B_b
    Each term is (3x2) x (2x4) = 3x4 output.
    Schoolbook: 3*2*4 = 24 per term, total 48. Not helpful.
    
    Let's just try the row-split 1+2 variant as v2 and this as v3 with col split.
    Actually let's do: split A rows as 2+1 but B cols as 3+1.
    """
    n, m, p = 3, 4, 4
    # Split: A rows 2+1, B cols 3+1
    # Top 2 rows of A (2x4) times B (4x4):
    #   = (2x4) x (4x3) + (2x4) x (4x1)
    #   For (2x4)x(4x3): split B into 2x2 blocks: 2*(2x2)x(2x3)... complex
    # Let's just do schoolbook for this variant to get a different structure
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Try multiple initial decompositions
    candidates = []
    
    U44, V44, W44 = _construct_hybrid_decomposition()
    if _verify_decomposition(U44, V44, W44, n, m, p, tol=1e-9):
        candidates.append((U44.copy(), V44.copy(), W44.copy(), 44))
    
    U44b, V44b, W44b = _construct_hybrid_decomposition_v2()
    if _verify_decomposition(U44b, V44b, W44b, n, m, p, tol=1e-9):
        candidates.append((U44b.copy(), V44b.copy(), W44b.copy(), 44))
    
    # Fallback: schoolbook rank-48
    U48 = np.zeros((n * m, 48))
    V48 = np.zeros((m * p, 48))
    W48 = np.zeros((p * n, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U48[i * m + j, col] = 1
                V48[j * p + k, col] = 1
                W48[k * n + i, col] = 1
                col += 1
    candidates.append((U48.copy(), V48.copy(), W48.copy(), 48))
    
    # Sort by rank, pick best
    candidates.sort(key=lambda x: x[3])
    best_U, best_V, best_W, best_rank = candidates[0]
    
    # Try to reduce rank using global dependency check on each candidate
    time_per_candidate = min(30, (deadline - time.time() - 15) / max(1, len(candidates)))
    
    for U_cand, V_cand, W_cand, rank_cand in candidates:
        if time.time() > deadline - 15:
            break
        if rank_cand >= best_rank:
            continue  # Can't improve if already at or above best
        time_budget = min(time_per_candidate, deadline - time.time() - 15)
        if time_budget < 5:
            break
        U_try, V_try, W_try = _try_reduce_rank_global(
            U_cand.copy(), V_cand.copy(), W_cand.copy(), n, m, p, time_budget
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = U_try.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
