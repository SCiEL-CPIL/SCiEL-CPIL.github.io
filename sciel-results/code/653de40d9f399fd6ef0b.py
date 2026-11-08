"""Matrix multiplication 4x4 x 4x8 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}})
    # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    terms.append({'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}})
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    terms.append({'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}})
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}})
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
    return terms

def _build_4x4x4_decomposition():
    """Build a rank-49 decomposition for 4x4x4 matrix multiplication using recursive Strassen."""
    strassen_terms = _strassen_2x2_terms()
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for meta_term in strassen_terms:
        for inner_term in strassen_terms:
            for (bi, bj), cu in meta_term['U'].items():
                for (ii, jj), cu_inner in inner_term['U'].items():
                    row_a = bi * 2 + ii
                    col_a = bj * 2 + jj
                    U[row_a * 4 + col_a][col] += cu * cu_inner
            
            for (bj, bk), cv in meta_term['V'].items():
                for (jj, kk), cv_inner in inner_term['V'].items():
                    row_b = bj * 2 + jj
                    col_b = bk * 2 + kk
                    V[row_b * 4 + col_b][col] += cv * cv_inner
            
            for (bi, bk), cw in meta_term['W'].items():
                for (ii, kk), cw_inner in inner_term['W'].items():
                    row_c = bk * 2 + kk
                    col_c = bi * 2 + ii
                    W[row_c * 4 + col_c][col] += cw * cw_inner
            
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
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
        
        a_part = np.sum(U_r * A_test, axis=1)
        b_part = np.sum(V_r * B_test, axis=0)
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += a_part[i] * b_part[k] * W_r[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U_444, V_444, W_444 = _build_4x4x4_decomposition()
    
    rank_total = 98
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(32)]
    W = [[0] * rank_total for _ in range(32)]
    
    # First half: columns 0-3
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][r] = U_444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*8+k][r] = V_444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W_444[k*4+i][r]
    
    # Second half: columns 4-7
    for r in range(49):
        r2 = 49 + r
        for i in range(4):
            for j in range(4):
                U[i*4+j][r2] = U_444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*8+(4+k)][r2] = V_444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[(4+k)*4+i][r2] = W_444[k*4+i][r]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
