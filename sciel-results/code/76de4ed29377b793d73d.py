"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _get_strassen_222_terms():
    """Return verified Strassen rank-7 terms for 2x2 * 2x2.
    Each term: (list of (i,j,coeff) for A, list of (j,k,coeff) for B, list of (k,i,coeff) for C)
    """
    terms = [
        # m1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # m2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # m3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # m4 = A11(B01-B00)
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # m5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # m6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # m7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
    ]
    # Verify
    np.random.seed(0)
    A = np.random.randn(2, 2)
    B = np.random.randn(2, 2)
    C_exp = A @ B
    C_comp = np.zeros((2, 2))
    for u_d, v_d, w_d in terms:
        uv = sum(A[i,j]*c for (i,j,c) in u_d) * sum(B[i,j]*c for (i,j,c) in v_d)
        for (k,i,c) in w_d:
            C_comp[i,k] += c * uv
    if not np.allclose(C_comp, C_exp, atol=1e-10):
        return None
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    strassen = _get_strassen_222_terms()
    
    # Block partition:
    # A: rows 0-1 (top), 2-3 (bottom); cols 0-1 (left), 2-4 (right)
    # B: rows 0-1 (top), 2-4 (bottom); cols 0-1 (left), 2-4 (right)
    # C: rows 0-1 (top), 2-3 (bottom); cols 0-1 (left), 2-4 (right)
    #
    # C11(2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    # C12(2x3) = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    # C21(2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    # C22(2x3) = A21(2x2)*B12(2x3) + A22(2x3)*B22(3x3)
    
    if strassen is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Rank calculation:
    # A11*B11: 7 (Strassen)
    # A12*B21: 2*3*2 = 12
    # A11*B12: 2*2*3 = 12
    # A12*B22: 2*3*3 = 18
    # A21*B11: 7 (Strassen)
    # A22*B21: 2*3*2 = 12
    # A21*B12: 2*2*3 = 12
    # A22*B22: 2*3*3 = 18
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 20 x 98
    V = [[0] * rank for _ in range(m * p)]  # 25 x 98
    W = [[0] * rank for _ in range(p * n)]  # 20 x 98
    col = 0
    
    def add_strassen(a_row_off, b_row_off, c_row_off):
        """Add Strassen for 2x2*2x2. A cols 0-1, B cols 0-1, C cols 0-1."""
        nonlocal col
        for u_d, v_d, w_d in strassen:
            for (i, j, c) in u_d:
                U[(a_row_off + i) * m + j][col] = c
            for (j, k, c) in v_d:
                V[(b_row_off + j) * p + k][col] = c
            for (k, i, c) in w_d:
                W[k * n + (c_row_off + i)][col] = c
            col += 1
    
    def add_schoolbook(a_row_off, a_col_off, a_r, a_c, b_row_off, b_col_off, b_r, b_c, c_row_off, c_col_off):
        """Add schoolbook for a_r x a_c * b_r x b_c -> a_r x b_c."""
        nonlocal col
        for i in range(a_r):
            for j in range(a_c):
                for k in range(b_c):
                    U[(a_row_off + i) * m + (a_col_off + j)][col] = 1
                    V[(b_row_off + j) * p + (b_col_off + k)][col] = 1
                    W[(c_col_off + k) * n + (c_row_off + i)][col] = 1
                    col += 1
    
    # C11 = A11*B11 (Strassen) + A12*B21 (schoolbook)
    add_strassen(0, 0, 0)
    add_schoolbook(0, 2, 2, 3, 2, 0, 3, 2, 0, 0)
    
    # C12 = A11*B12 (schoolbook) + A12*B22 (schoolbook)
    add_schoolbook(0, 0, 2, 2, 0, 2, 2, 3, 0, 2)
    add_schoolbook(0, 2, 2, 3, 2, 2, 3, 3, 0, 2)
    
    # C21 = A21*B11 (Strassen) + A22*B21 (schoolbook)
    add_strassen(2, 0, 2)
    add_schoolbook(2, 2, 2, 3, 2, 0, 3, 2, 2, 0)
    
    # C22 = A21*B12 (schoolbook) + A22*B22 (schoolbook)
    add_schoolbook(2, 0, 2, 2, 0, 2, 2, 3, 2, 2)
    add_schoolbook(2, 2, 2, 3, 2, 2, 3, 3, 2, 2)
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verify
    np.random.seed(42)
    A_t = np.random.randn(n, m)
    B_t = np.random.randn(m, p)
    C_exp = A_t @ B_t
    C_comp = np.zeros((n, p))
    U_a = np.array(U, dtype=float)
    V_a = np.array(V, dtype=float)
    W_a = np.array(W, dtype=float)
    A_flat = A_t.reshape(-1)
    B_flat = B_t.reshape(-1)
    for r in range(rank):
        C_comp += np.outer(W_a[:, r].reshape(p, n), U_a[:, r] @ A_flat * (V_a[:, r] @ B_flat))
    
    if not np.allclose(C_comp, C_exp, atol=1e-10):
        # Fallback
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
