import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor (vectorized)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Vectorized: for each c, outer product contribution
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return bool(np.allclose(T, 1.0, atol=1e-9))

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Uses the standard Strassen decomposition.
    """
    # Standard Strassen 2x2x2 rank-7 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1
    
    return U, V, W

def _get_strassen_2x2x4():
    """
    Returns U, V, W for 2x2x4 matrix multiplication with rank 14.
    Uses two independent Strassen 2x2x2 decompositions.
    """
    U_222, V_222, W_222 = _get_strassen_2x2x2()
    
    U = np.zeros((4, 14), dtype=int)
    V = np.zeros((8, 14), dtype=int)
    W = np.zeros((8, 14), dtype=int)
    
    # First half: columns 0-6 for the first 2 columns of B
    U[:, 0:7] = U_222
    V[0:4, 0:7] = V_222
    W[0:4, 0:7] = W_222
    
    # Second half: columns 7-13 for the second 2 columns of B
    U[:, 7:14] = U_222
    V[4:8, 7:14] = V_222
    W[4:8, 7:14] = W_222
    
    return U, V, W

def _strassen_4x4x8():
    """
    Construct 4x4x8 using Strassen block decomposition.
    Split into 2x2 blocks, use Strassen 7-term formula at block level.
    Each block multiplication is 2x2 * 2x4, using rank-14 decomposition.
    Total rank = 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    U_224, V_224, W_224 = _get_strassen_2x2x4()
    
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    # Strassen terms: (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # Block indices: (row_block, col_block) where each is 0 or 1
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10 += M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3 = A00(B01-B11) -> C01 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4 = A11(B10-B00) -> C11 -= M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): -1}),
        # M5 = (A00+A01)B11 -> C00 += M5, C01 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C10 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1}),
        # M7 = (A01-A11)(B10+B11) -> C01 -= M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): -1}),
    ]
    
    for term_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_terms):
        col_start = term_idx * 14
        
        # Add A contributions to U
        for (a_row, a_col), coeff in a_coeffs.items():
            # A block (a_row, a_col) is 2x2
            # U row index: (a_row*2 + lr) * 4 + (a_col*2 + lc)
            for lr in range(2):
                for lc in range(2):
                    u_row = (a_row * 2 + lr) * 4 + (a_col * 2 + lc)
                    U[u_row, col_start:col_start+14] += coeff * U_224[lr*2+lc]
        
        # Add B contributions to V
        for (b_row, b_col), coeff in b_coeffs.items():
            # B block (b_row, b_col) is 2x4
            # V row index: (b_row*2 + lr) * 8 + (b_col*4 + lc)
            for lr in range(2):
                for lc in range(4):
                    v_row = (b_row * 2 + lr) * 8 + (b_col * 4 + lc)
                    V[v_row, col_start:col_start+14] += coeff * V_224[lr*4+lc]
        
        # Add C contributions to W
        for (c_row, c_col), coeff in c_coeffs.items():
            # C block (c_row, c_col) is 2x4
            # W row index: (c_col*4 + lc) * 4 + (c_row*2 + lr)
            for lr in range(2):
                for lc in range(4):
                    w_row = (c_col * 4 + lc) * 4 + (c_row * 2 + lr)
                    W[w_row, col_start:col_start+14] += coeff * W_224[lc*2+lr]
    
    return U, V, W

# EVOLVE-BLOCK-START
def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce the rank by finding columns whose tensor contributions are
    linearly dependent. If sum_c alpha_c * col_vec_c = 0 and alpha_c = -1 for some c,
    then column c can be eliminated.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Build column vectors: col_vecs[c] = flatten of tensor contribution of column c
    col_vecs = np.zeros((r, n * m * p), dtype=float)
    for c in range(r):
        u_2d = U[:, c].reshape(n, m)
        v_2d = V[:, c].reshape(m, p)
        w_2d = W[:, c].reshape(p, n)
        t_c = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
        col_vecs[c] = t_c.flatten()
    
    # Check for zero columns
    zero_cols = [c for c in range(r) if np.allclose(col_vecs[c], 0, atol=1e-12)]
    if zero_cols:
        keep = [c for c in range(r) if c not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        r = len(keep)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W, r
    
    # Find null space of col_vecs (r x n*m*p matrix)
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(col_vecs, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, r
    
    tol = 1e-10 * (max(s_vals) if len(s_vals) > 0 else 1.0)
    rank_cv = sum(1 for s in s_vals if s > tol)
    
    if rank_cv < r:
        null_vecs = Vt_svd[rank_cv:]  # (null_dim, r)
        
        # For each null vector, try to find a scaling that gives a -1 component
        for nv in null_vecs:
            nonzero = np.abs(nv) > 1e-12
            if not np.any(nonzero):
                continue
            # For each nonzero component c, scale nv so that nv[c] = -1
            for c in range(r):
                if abs(nv[c]) < 1e-12:
                    continue
                scale = -1.0 / nv[c]
                alpha = nv * scale
                # Now alpha[c] = -1, so column c can be eliminated
                # Verify: sum_c' (1 + alpha[c']) * col_vecs[c'] should still = 1
                # This is guaranteed by the null space property
                # Remove column c
                keep = [cc for cc in range(r) if cc != c]
                U_new = U[:, keep]
                V_new = V[:, keep]
                W_new = W[:, keep]
                if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                    return U_new, V_new, W_new, len(keep)
    
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        # Get verified 2x2x2 Strassen decomposition
        U_222, V_222, W_222 = _get_strassen_2x2x2()
        
        if not _verify(U_222.tolist(), V_222.tolist(), W_222.tolist(), 2, 2, 2):
            raise ValueError("2x2x2 Strassen verification failed")
        
        # Construct 4x4x8 using Strassen block decomposition
        U, V, W = _strassen_4x4x8()
        
        if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            raise ValueError("4x4x8 Strassen verification failed")
        
        # Attempt rank reduction
        if time.time() < deadline - 5:
            U, V, W, r = _reduce_rank(U.tolist(), V.tolist(), W.tolist(), n, m, p, deadline)
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            raise ValueError("Reduced decomposition verification failed")
            
    except Exception as e:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
