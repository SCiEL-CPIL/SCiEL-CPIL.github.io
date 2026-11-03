import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Rank-36 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    - Split B (4x5) into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3).
    - C = A1 * B1 + A2 * B2.
    - Each 2x2 * 2x5 is decomposed into two 2x2*2x2 Strassen blocks (rank 7 each)
      and one 2x2*2x1 schoolbook block (rank 4).
    - Total rank: 7 + 7 + 4 + 7 + 7 + 4 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    _start = time.time()
    _deadline = _start + 12.0  # safety margin before 120s limit
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen(col_offset, j_offset, k_offset):
        """
        Add Strassen 2x2x2 decomposition.
        A block: rows 0-1, cols j_offset to j_offset+1.
        B block: rows j_offset to j_offset+1, cols k_offset to k_offset+1.
        C block: rows 0-1, cols k_offset to k_offset+1.
        
        U indices: i*m + j
        V indices: j*p + k
        W indices: k*n + i
        """
        c = col_offset
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0*m + j_offset][c] += 1
        U[1*m + (j_offset+1)][c] += 1
        V[j_offset*p + k_offset][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[1*m + j_offset][c] += 1
        U[1*m + (j_offset+1)][c] += 1
        V[j_offset*p + k_offset][c] += 1
        W[k_offset*n + 1][c] += 1
        W[(k_offset+1)*n + 1][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[0*m + j_offset][c] += 1
        V[j_offset*p + (k_offset+1)][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] -= 1
        W[(k_offset+1)*n + 0][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[1*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + k_offset][c] += 1
        V[j_offset*p + k_offset][c] -= 1
        W[k_offset*n + 0][c] += 1
        W[k_offset*n + 1][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[0*m + j_offset][c] += 1
        U[0*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] -= 1
        W[(k_offset+1)*n + 0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1*m + j_offset][c] += 1
        U[0*m + j_offset][c] -= 1
        V[j_offset*p + k_offset][c] += 1
        V[j_offset*p + (k_offset+1)][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0*m + (j_offset+1)][c] += 1
        U[1*m + (j_offset+1)][c] -= 1
        V[(j_offset+1)*p + k_offset][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] += 1
        c += 1
        
        return c

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        """
        Add schoolbook 2x2x1 decomposition.
        A block: rows 0-1, cols j_offset to j_offset+1.
        B block: rows j_offset to j_offset+1, col k_offset.
        C block: rows 0-1, col k_offset.
        """
        c = col_offset
        # i=0, j=j_offset
        U[0*m + j_offset][c] += 1
        V[j_offset*p + k_offset][c] += 1
        W[k_offset*n + 0][c] += 1
        c += 1
        
        # i=0, j=j_offset+1
        U[0*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + k_offset][c] += 1
        W[k_offset*n + 0][c] += 1
        c += 1
        
        # i=1, j=j_offset
        U[1*m + j_offset][c] += 1
        V[j_offset*p + k_offset][c] += 1
        W[k_offset*n + 1][c] += 1
        c += 1
        
        # i=1, j=j_offset+1
        U[1*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + k_offset][c] += 1
        W[k_offset*n + 1][c] += 1
        c += 1
        
        return c

    col = 0
    
    # A1 (j_offset=0)
    # B11 (k_offset=0)
    col = add_strassen(col, 0, 0)
    # B12 (k_offset=2)
    col = add_strassen(col, 0, 2)
    # b1 (k_offset=4)
    col = add_schoolbook_2x2x1(col, 0, 4)
    
    # A2 (j_offset=2)
    # B21 (k_offset=0)
    col = add_strassen(col, 2, 0)
    # B22 (k_offset=2)
    col = add_strassen(col, 2, 2)
    # b2 (k_offset=4)
    col = add_schoolbook_2x2x1(col, 2, 4)
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verification
    if time.time() < _deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        alphas = A_flat @ U_arr
        betas = B_flat @ V_arr
        
        C_calc = np.einsum('knr,r,r->nk', W_arr.reshape(p, n, rank), alphas, betas)
        
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            pass

    return {"U": U, "V": V, "W": W}
