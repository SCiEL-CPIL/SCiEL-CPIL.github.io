import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _build_base_decomposition():
    """Build the rank-28 base decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    # Block 1: j=0, k=0..4 (schoolbook 2x1x5) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M2: (A10+A11)B00
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1

        # M3: A00(B01-B11)
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M4: A11(B10-B00)
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1

        # M5: (A00+A01)B11
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    add_strassen(i_off=0, j_off=1, k_off=0)

    # Block 2b: j={1,2}, k=2 (schoolbook 2x2x1) -> 4 terms
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1

    add_strassen(i_off=0, j_off=1, k_off=3)

    assert col == rank, f"Column mismatch: {col} != {rank}"
    return U, V, W, rank

def _verify_tensor(U, V, W, n=2, m=3, p=5):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(U, V, W, rank, deadline):
    """
    Try to find a lower-rank decomposition by identifying linear dependencies
    in the tensor space. The tensor equation is:
    T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We build a matrix where each column r contributes a vector of size n*m*p:
    M[:, r] = vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r]) (appropriately permuted)
    
    If columns are linearly dependent, we can reduce rank.
    """
    n, m, p = 2, 3, 5
    dim = n * m * p  # 30
    
    # Build the tensor matrix: M[idx, r] where idx = i*(m*p) + j*p + k
    M = np.zeros((dim, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    M[idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # Check if target vector (all ones) is in column space
    target = np.ones(dim)
    
    # SVD to find rank and null space
    try:
        U_svd, s, Vh = np.linalg.svd(M, full_matrices=True)
    except Exception:
        return None
    
    tol = 1e-10
    effective_rank = np.sum(s > tol)
    
    if effective_rank >= rank:
        return None
    
    # The null space of M^T gives directions in coefficient space that don't change the tensor
    # Actually, we need: find c such that M @ c = 0, then we can perturb
    # But for rank reduction, we need to find a column that's in the span of others
    
    # Check each column: is it in the span of the others?
    for remove_idx in range(rank):
        if time.time() > deadline:
            break
        
        M_other = np.delete(M, remove_idx, axis=1)
        u_other, s_other, vh_other = np.linalg.svd(M_other, full_matrices=True)
        rank_other = np.sum(s_other > tol)
        
        if rank_other < effective_rank:
            # Column remove_idx is linearly dependent on the others
            # Find the coefficients
            # M @ c = 0 where c[remove_idx] = 1
            # M_other @ c_other = -M[:, remove_idx]
            rhs = -M[:, remove_idx]
            # Least squares
            try:
                coeffs, residuals, rank_ls, sv = np.linalg.lstsq(M_other, rhs, rcond=tol)
            except Exception:
                continue
            
            if np.linalg.norm(M_other @ coeffs - rhs) < 1e-8:
                # We can eliminate column remove_idx
                new_rank = rank - 1
                U_new = np.zeros((n * m, new_rank), dtype=float)
                V_new = np.zeros((m * p, new_rank), dtype=float)
                W_new = np.zeros((p * n, new_rank), dtype=float)
                
                col_new = 0
                for j in range(rank):
                    if j == remove_idx:
                        continue
                    coeff = 1.0 - coeffs[col_new]  # Adjust for the dependency
                    # Actually, the relationship is M[:, remove_idx] = M_other @ coeffs
                    # So sum_j M[:, j] * delta_j = 0 where delta[remove_idx]=1, delta[j]=coeffs[j-remove_idx_adjusted]
                    # This means we can't simply scale; we need a different approach
                    
                    U_new[:, col_new] = U[:, j]
                    V_new[:, col_new] = V[:, j]
                    W_new[:, col_new] = W[:, j]
                    col_new += 1
                
                # Verify the reduced decomposition
                if _verify_tensor(U_new, V_new, W_new):
                    return U_new, V_new, W_new, new_rank
                else:
                    # The simple elimination doesn't work; the tensor equation
                    # is not linear in the factor matrices
                    continue
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # Build base rank-28 decomposition
    U, V, W, rank = _build_base_decomposition()
    
    # Verify base decomposition
    if not _verify_tensor(U, V, W):
        # Fallback: schoolbook rank 30
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Try rank reduction
    reduced = _try_rank_reduction(U, V, W, rank, DEADLINE)
    if reduced is not None:
        U_r, V_r, W_r, new_rank = reduced
        if _verify_tensor(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Try alternative: different block partitioning
    # Strategy: j={0,1}, k={0,1,2,3,4} as 2x2x5, and j=2 as 2x1x5
    # 2x2x5: try Strassen(2x2x2) + schoolbook(2x2x3) = 7 + 6 = 13
    # 2x1x5: schoolbook = 10
    # Total: 23? Let's try to construct this.
    
    if time.time() < DEADLINE - 10:
        rank_alt = 23
        U_alt = np.zeros((n * m, rank_alt), dtype=float)
        V_alt = np.zeros((m * p, rank_alt), dtype=float)
        W_alt = np.zeros((p * n, rank_alt), dtype=float)
        col = 0

        def add_strassen_alt(U, V, W, col, i_off, j_off, k_off):
            # Same Strassen but with explicit U,V,W args
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = -1.0
            col += 1

            U[(i_off+0)*m + (j_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = -1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            col += 1

            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            col += 1

            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+0), col] = -1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1

            U[(i_off+0)*m + (j_off+1), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = -1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            col += 1
            return col

        # j={0,1}, k={0,1}: Strassen (7 terms)
        col = add_strassen_alt(U_alt, V_alt, W_alt, col, 0, 0, 0)
        
        # j={0,1}, k={2,3}: Strassen (7 terms)
        col = add_strassen_alt(U_alt, V_alt, W_alt, col, 0, 0, 2)
        
        # j={0,1}, k={4}: schoolbook 2x2x1 (4 terms)
        for i in range(n):
            for j in [0, 1]:
                U_alt[i * m + j, col] = 1.0
                V_alt[j * p + 4, col] = 1.0
                W_alt[4 * n + i, col] = 1.0
                col += 1

        # j=2, k={0,1,2,3,4}: schoolbook 2x1x5 (10 terms)
        for i in range(n):
            for k in range(p):
                U_alt[i * m + 2, col] = 1.0
                V_alt[2 * p + k, col] = 1.0
                W_alt[k * n + i, col] = 1.0
                col += 1

        assert col == rank_alt, f"Alt column mismatch: {col} != {rank_alt}"
        
        if _verify_tensor(U_alt, V_alt, W_alt):
            return {"U": U_alt.tolist(), "V": V_alt.tolist(), "W": W_alt.tolist()}

    # Return the base rank-28 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
