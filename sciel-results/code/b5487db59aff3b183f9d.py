import time
import numpy as np

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a dict with 'U', 'V', 'W' keys, each mapping (row, col) to coefficient.
    U: (i,j) -> coeff for A[i,j]
    V: (j,k) -> coeff for B[j,k]
    W: (k,i) -> coeff for C[k,i] (trace convention: W[k*n+i, r])
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (0,0): -1}
    })
    
    # M3 = A00(B01-B11)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A11(B10-B00)
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00+A01)B11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): 1, (0,1): -1}
    })
    
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    return terms

def _build_4x4x4_decomposition():
    """
    Builds a rank-49 decomposition for 4x4x4 matrix multiplication
    using the Kronecker product of two 2x2x2 Strassen decompositions.
    
    Returns U (16,49), V (16,49), W (16,49) where:
    U[i*4+j, r] = coefficient of A[i,j]
    V[j*4+k, r] = coefficient of B[j,k]
    W[k*4+i, r] = coefficient of C[k,i] (trace convention)
    """
    strassen_terms = _strassen_2x2_terms()
    rank = 49
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    col = 0
    for meta_term in strassen_terms:
        for inner_term in strassen_terms:
            # U part: outer[i1,j1] * inner[i0,j0]
            for (bi, bj), cu in meta_term['U'].items():
                for (ii, jj), cu_inner in inner_term['U'].items():
                    row_a = bi * 2 + ii
                    col_a = bj * 2 + jj
                    U[row_a * 4 + col_a][col] += cu * cu_inner
            
            # V part: outer[j1,k1] * inner[j0,k0]
            for (bj, bk), cv in meta_term['V'].items():
                for (jj, kk), cv_inner in inner_term['V'].items():
                    row_b = bj * 2 + jj
                    col_b = bk * 2 + kk
                    V[row_b * 4 + col_b][col] += cv * cv_inner
            
            # W part: outer[k1,i1] * inner[k0,i0]
            for (bi, bk), cw in meta_term['W'].items():
                for (ii, kk), cw_inner in inner_term['W'].items():
                    row_c = bk * 2 + kk
                    col_c = bi * 2 + ii
                    W[row_c * 4 + col_c][col] += cw * cw_inner
            
            col += 1
    
    return U, V, W

def _build_4x4x8_decomposition():
    """
    Builds a rank-98 decomposition for 4x4x8 matrix multiplication
    by applying the 49-rank 4x4x4 decomposition to both halves of the 8-column B matrix.
    
    Returns U (16,98), V (32,98), W (32,98) where:
    U[i*4+j, r] = coefficient of A[i,j]
    V[j*8+k, r] = coefficient of B[j,k]
    W[k*4+i, r] = coefficient of C[k,i] (trace convention)
    """
    U_444, V_444, W_444 = _build_4x4x4_decomposition()
    rank_444 = 49
    rank_total = 2 * rank_444  # 98
    
    U = [[0.0] * rank_total for _ in range(16)]
    V = [[0.0] * rank_total for _ in range(32)]
    W = [[0.0] * rank_total for _ in range(32)]
    
    # First half: columns 0-3 of B
    for r in range(rank_444):
        for idx in range(16):
            U[idx][r] = U_444[idx][r]
        for j in range(4):
            for k in range(4):
                V[j * 8 + k][r] = V_444[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][r] = W_444[k * 4 + i][r]
    
    # Second half: columns 4-7 of B
    for r in range(rank_444):
        r2 = rank_444 + r
        for idx in range(16):
            U[idx][r2] = U_444[idx][r]
        for j in range(4):
            for k in range(4, 8):
                V[j * 8 + k][r2] = V_444[j * 4 + (k - 4)][r]
        for k in range(4, 8):
            for i in range(4):
                W[k * 4 + i][r2] = W_444[(k - 4) * 4 + i][r]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition is correct."""
    rank = len(U[0])
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        U_r = U_mat[:, r].reshape(n, m)
        V_r = V_mat[:, r].reshape(m, p)
        W_r = W_mat[:, r].reshape(p, n)
        
        # Contract A with U_r: a_i = sum_j U_r[i,j] * A[i,j]
        a_part = np.sum(U_r * A_test, axis=1)
        # Contract B with V_r: b_k = sum_j V_r[j,k] * B[j,k]
        b_part = np.sum(V_r * B_test, axis=0)
        
        # C += a_part * b_part * W_r.T (since W_r is (p,n) and we need (n,p))
        C_computed += np.outer(a_part, b_part) * W_r.T
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    # Primary: 98-rank decomposition using Kronecker Strassen
    try:
        U, V, W = _build_4x4x8_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook 128-rank decomposition
    rank_schoolbook = n * m * p  # 128
    U_sb = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0.0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    
    if _verify_decomposition(U_sb, V_sb, W_sb, n, m, p):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Last resort: return schoolbook without verification
    return {"U": U_sb, "V": V_sb, "W": W_sb}
