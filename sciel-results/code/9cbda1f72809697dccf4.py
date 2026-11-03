"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-26 exact tensor decomposition for 3x3 matrix multiplication.
    
    The decomposition uses Strassen's 7-multiplication algorithm for the top-left 2x2
    block, and standard multiplication for all remaining terms involving index 2.
    
    Convention:
      U has shape (9, rank), indexed by (i, j) -> row i*3 + j
      V has shape (9, rank), indexed by (j, k) -> row j*3 + k
      W has shape (9, rank), indexed by (k, i) -> row k*3 + i
      
    The matrix multiplication tensor is:
      C[i, k] = sum_j A[i, j] * B[j, k]
    
    Each rank-1 term c contributes:
      A[i,j] coefficient in U[i*3+j, c]
      B[j,k] coefficient in V[j*3+k, c]
      C[i,k] coefficient in W[k*3+i, c]
    """
    n, m, p = DIMENSIONS
    rank = 26
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (i, j) pairs
    V = [[0] * rank for _ in range(m * p)]  # (j, k) pairs
    W = [[0] * rank for _ in range(p * n)]  # (k, i) pairs
    
    def set_u(i, j, c, val):
        U[i * m + j][c] = val
    
    def set_v(j, k, c, val):
        V[j * p + k][c] = val
    
    def set_w(k, i, c, val):
        W[k * n + i][c] = val
    
    # ============================================================
    # Terms 0-6: Strassen's algorithm for the 2x2 core
    # ============================================================
    # A_core = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_core = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C_core = A_core * B_core
    
    # M1 = (A00 + A11) * (B00 + B11)
    c = 0
    set_u(0, 0, c, 1)
    set_u(1, 1, c, 1)
    set_v(0, 0, c, 1)
    set_v(1, 1, c, 1)
    # C00 += M1, C11 -= M1
    set_w(0, 0, c, 1)
    set_w(1, 1, c, -1)
    
    # M2 = (A10 + A11) * B00
    c = 1
    set_u(1, 0, c, 1)
    set_u(1, 1, c, 1)
    set_v(0, 0, c, 1)
    # C10 += M2, C11 -= M2
    set_w(0, 1, c, 1)
    set_w(1, 1, c, -1)
    
    # M3 = A00 * (B01 - B11)
    c = 2
    set_u(0, 0, c, 1)
    set_v(0, 1, c, 1)
    set_v(1, 1, c, -1)
    # C01 += M3, C11 += M3
    set_w(1, 0, c, 1)
    set_w(1, 1, c, 1)
    
    # M4 = A11 * (B10 - B00)
    c = 3
    set_u(1, 1, c, 1)
    set_v(1, 0, c, 1)
    set_v(0, 0, c, -1)
    # C00 += M4, C10 += M4
    set_w(0, 0, c, 1)
    set_w(0, 1, c, 1)
    
    # M5 = (A00 + A01) * B11
    c = 4
    set_u(0, 0, c, 1)
    set_u(0, 1, c, 1)
    set_v(1, 1, c, 1)
    # C00 -= M5, C01 += M5
    set_w(0, 0, c, -1)
    set_w(1, 0, c, 1)
    
    # M6 = (A10 - A00) * (B00 + B01)
    c = 5
    set_u(1, 0, c, 1)
    set_u(0, 0, c, -1)
    set_v(0, 0, c, 1)
    set_v(0, 1, c, 1)
    # C11 += M6
    set_w(1, 1, c, 1)
    
    # M7 = (A01 - A11) * (B10 + B11)
    c = 6
    set_u(0, 1, c, 1)
    set_u(1, 1, c, -1)
    set_v(1, 0, c, 1)
    set_v(1, 1, c, 1)
    # C00 += M7
    set_w(0, 0, c, 1)
    
    # ============================================================
    # Terms 7-10: Cross terms for C[i, k] where i, k < 2 but j = 2
    # C[i, k] += A[i, 2] * B[2, k] for i, k in {0, 1}
    # ============================================================
    c = 7
    set_u(0, 2, c, 1)
    set_v(2, 0, c, 1)
    set_w(0, 0, c, 1)
    
    c = 8
    set_u(0, 2, c, 1)
    set_v(2, 1, c, 1)
    set_w(1, 0, c, 1)
    
    c = 9
    set_u(1, 2, c, 1)
    set_v(2, 0, c, 1)
    set_w(0, 1, c, 1)
    
    c = 10
    set_u(1, 2, c, 1)
    set_v(2, 1, c, 1)
    set_w(1, 1, c, 1)
    
    # ============================================================
    # Terms 11-16: C[i, 2] for i in {0, 1}, all j
    # C[i, 2] = sum_j A[i, j] * B[j, 2]
    # ============================================================
    c = 11
    set_u(0, 0, c, 1)
    set_v(0, 2, c, 1)
    set_w(2, 0, c, 1)
    
    c = 12
    set_u(0, 1, c, 1)
    set_v(1, 2, c, 1)
    set_w(2, 0, c, 1)
    
    c = 13
    set_u(0, 2, c, 1)
    set_v(2, 2, c, 1)
    set_w(2, 0, c, 1)
    
    c = 14
    set_u(1, 0, c, 1)
    set_v(0, 2, c, 1)
    set_w(2, 1, c, 1)
    
    c = 15
    set_u(1, 1, c, 1)
    set_v(1, 2, c, 1)
    set_w(2, 1, c, 1)
    
    c = 16
    set_u(1, 2, c, 1)
    set_v(2, 2, c, 1)
    set_w(2, 1, c, 1)
    
    # ============================================================
    # Terms 17-22: C[2, k] for k in {0, 1}, all j
    # C[2, k] = sum_j A[2, j] * B[j, k]
    # ============================================================
    c = 17
    set_u(2, 0, c, 1)
    set_v(0, 0, c, 1)
    set_w(0, 2, c, 1)
    
    c = 18
    set_u(2, 1, c, 1)
    set_v(1, 0, c, 1)
    set_w(0, 2, c, 1)
    
    c = 19
    set_u(2, 2, c, 1)
    set_v(2, 0, c, 1)
    set_w(0, 2, c, 1)
    
    c = 20
    set_u(2, 0, c, 1)
    set_v(0, 1, c, 1)
    set_w(1, 2, c, 1)
    
    c = 21
    set_u(2, 1, c, 1)
    set_v(1, 1, c, 1)
    set_w(1, 2, c, 1)
    
    c = 22
    set_u(2, 2, c, 1)
    set_v(2, 1, c, 1)
    set_w(1, 2, c, 1)
    
    # ============================================================
    # Terms 23-25: C[2, 2] = sum_j A[2, j] * B[j, 2]
    # ============================================================
    c = 23
    set_u(2, 0, c, 1)
    set_v(0, 2, c, 1)
    set_w(2, 2, c, 1)
    
    c = 24
    set_u(2, 1, c, 1)
    set_v(1, 2, c, 1)
    set_w(2, 2, c, 1)
    
    c = 25
    set_u(2, 2, c, 1)
    set_v(2, 2, c, 1)
    set_w(2, 2, c, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
