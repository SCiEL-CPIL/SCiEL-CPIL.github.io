import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) where keys are (row, col).
    U, V are 2x2, W is 2x2 (W[(i,k)] for output C[i,k]).
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    })
    # M2 = (A10+A11)B00
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): 1}
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
        'W': {(0,0): 1, (0,1): 1}
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
        'W': {(1,0): 1}
    })
    return terms

def _build_4x4x4_decomposition():
    """
    Builds a rank-49 decomposition for 4x4x4 using nested Strassen.
    """
    strassen_terms = _strassen_2x2_terms()
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    col = 0
    for meta_term in strassen_terms:
        for inner_term in strassen_terms:
            for (bi, bj), cu in meta_term['U'].items():
                for (ii, jj), cu_inner in inner_term['U'].items():
                    row_a = bi * 2 + ii
                    col_a = bj * 2 + jj
                    U[row_a * 4 + col_a, col] += cu * cu_inner
            
            for (bj, bk), cv in meta_term['V'].items():
                for (jj, kk), cv_inner in inner_term['V'].items():
                    row_b = bj * 2 + jj
                    col_b = bk * 2 + kk
                    V[row_b * 4 + col_b, col] += cv * cv_inner
            
            for (bi, bk), cw in meta_term['W'].items():
                for (ii, kk), cw_inner in inner_term['W'].items():
                    row_c = bk * 2 + kk
                    col_c = bi * 2 + ii
                    W[row_c * 4 + col_c, col] += cw * cw_inner
            
            col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition."""
    rank = U.shape[1]
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        
        a_part = np.sum(U_r * A_test)
        b_part = np.sum(V_r * B_test)
        
        C_computed += np.outer(a_part, b_part) * W_r.T

    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U_444, V_444, W_444 = _build_4x4x4_decomposition()
    
    rank_total = 98
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # First half (k=0..3)
    for r in range(49):
        U[:, r] = U_444[:, r]
        for j in range(m):
            for k in range(4):
                V[j * p + k, r] = V_444[j * 4 + k, r]
        W[:, r] = W_444[:, r]
    
    # Second half (k=4..7)
    for r in range(49):
        r2 = 49 + r
        U[:, r2] = U_444[:, r]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k), r2] = V_444[j * 4 + k, r]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * n + i, r2] = W_444[k * 4 + i, r]
                
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
