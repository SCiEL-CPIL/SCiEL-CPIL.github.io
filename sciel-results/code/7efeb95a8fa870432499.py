"""Matrix multiplication 4x4x6 with Strassen-based block decomposition."""
DIMENSIONS = [4, 4, 6]

# Strassen's 7-term decomposition for 2x2x2 matrix multiplication
# Each term (U_factor, V_factor, W_factor) is a set of (local_idx, value) tuples
# U: 2x2 matrix A, indexed as row*2+col -> (0,1,2,3)
# V: 2x2 matrix B, indexed as row*2+col -> (0,1,2,3)
# W: 2x2 matrix C, indexed as row*2+col -> (0,1,2,3)
#
# M1 = (A00+A11)(B00+B11): C00+=M1, C11+=M1
# M2 = (A10+A11)B00:       C10+=M2, C11-=M2
# M3 = A00(B10-B11):       C01+=M3, C11+=M3
# M4 = A11(B01-B00):       C00+=M4, C10+=M4
# M5 = (A00+A10)B11:       C00-=M5, C01+=M5
# M6 = (A01-A00)(B00+B10): C11+=M6
# M7 = (A11-A01)(B11+B01): C00+=M7, C11+=M7

STRASSEN_TERMS = [
    # (U_coeffs, V_coeffs, W_coeffs)
    # M1
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    # M2
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    # M3
    ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
    # M4
    ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
    # M5
    ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (1, 1)]),
    # M6
    ([(1, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
    # M7
    ([(3, 1), (1, -1)], [(3, 1), (1, 1)], [(0, 1), (3, 1)]),
]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block sizes: 2x2 for the 4x4 and 4x6 dimensions
    # A is 4x4: split into 2x2 blocks -> 2x2 grid of 2x2 blocks
    # B is 4x6: split into 2x2 (col dim 2, row dim 2) -> 2x3 grid of 2x2 blocks
    # C is 4x6: split into 2x2 (col dim 2, row dim 2) -> 2x3 grid of 2x2 blocks
    
    rank = 12 * 7  # 12 sub-problems * 7 Strassen terms each = 84
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # For each sub-problem (pi, rj, qk):
    #   pi in 0..1 (row block of A/C)
    #   rj in 0..1 (col block of A / row block of B)
    #   qk in 0..2 (col block of B/C)
    # 
    # Local 2x2 indices:
    #   A[u][v] -> global A[2*pi+u][2*rj+v] -> U index (2*pi+u)*m + (2*rj+v)
    #   B[v][w] -> global B[2*rj+v][2*qk+w] -> V index (2*rj+v)*p + (2*qk+w)
    #   C[u][w] -> global C[2*pi+u][2*qk+w] -> W index (2*qk+w)*n + (2*pi+u)
    #     (W uses trace convention W[k*n+i, r] where k is col, i is row)
    
    col = 0
    for pi in range(2):       # row block of A and C
        for rj in range(2):   # col block of A, row block of B
            for qk in range(3):  # col block of B and C
                for (u_coeffs, v_coeffs, w_coeffs) in STRASSEN_TERMS:
                    # Fill U
                    for (local_idx, val) in u_coeffs:
                        u_local, v_local = local_idx // 2, local_idx % 2
                        u_global = 2 * pi + u_local
                        v_global = 2 * rj + v_local
                        U[u_global * m + v_global][col] += val
                    
                    # Fill V
                    for (local_idx, val) in v_coeffs:
                        v_local, w_local = local_idx // 2, local_idx % 2
                        v_global = 2 * rj + v_local
                        w_global = 2 * qk + w_local
                        V[v_global * p + w_global][col] += val
                    
                    # Fill W (trace convention: W[k*n+i, r] where k=col, i=row)
                    for (local_idx, val) in w_coeffs:
                        u_local, w_local = local_idx // 2, local_idx % 2
                        u_global = 2 * pi + u_local
                        w_global = 2 * qk + w_local
                        W[w_global * n + u_global][col] += val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
