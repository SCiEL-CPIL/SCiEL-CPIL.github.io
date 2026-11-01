"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
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

    # Try to construct rank 25 decomposition
    # Strategy: 3 pairs of columns using Strassen (rank 7 each) + 1 single column (rank 4)
    # Total rank = 3 * 7 + 4 = 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We have 3 pairs: (0,1), (2,3), (4,5) and single 6
    # For each pair, we add 7 terms
    # For the single, we add 4 terms
    
    def add_strassen_terms(col_offset):
        nonlocal col_idx
        # M1
        U[0][col_idx] += 1 # A00
        U[3][col_idx] += 1 # A11
        V[0 * p + col_offset][col_idx] += 1 # B00
        V[1 * p + col_offset + 1][col_idx] += 1 # B11
        W[col_offset * n + 0][col_idx] += 1 # C00
        W[(col_offset + 1) * n + 1][col_idx] += 1 # C11
        col_idx += 1
        
        # M2
        U[2][col_idx] += 1 # A10
        U[3][col_idx] += 1 # A11
        V[0 * p + col_offset + 1][col_idx] += 1 # B01
        W[col_offset * n + 1][col_idx] += 1 # C10
        W[(col_offset + 1) * n + 1][col_idx] -= 1 # C11
        col_idx += 1
        
        # M3
        U[0][col_idx] += 1 # A00
        V[1 * p + col_offset][col_idx] += 1 # B10
        V[1 * p + col_offset + 1][col_idx] -= 1 # B11
        W[col_offset * n + 1][col_idx] += 1 # C01
        W[(col_offset + 1) * n + 1][col_idx] += 1 # C11
        col_idx += 1
        
        # M4
        U[3][col_idx] += 1 # A11
        V[0 * p + col_offset][col_idx] += 1 # B00
        V[0 * p + col_offset + 1][col_idx] -= 1 # B01
        W[col_offset * n + 0][col_idx] += 1 # C00
        W[col_offset * n + 1][col_idx] += 1 # C10
        col_idx += 1
        
        # M5
        U[0][col_idx] += 1 # A00
        U[1][col_idx] += 1 # A01
        V[1 * p + col_offset + 1][col_idx] += 1 # B11
        W[col_offset * n + 0][col_idx] -= 1 # C00
        W[col_offset * n + 1][col_idx] += 1 # C01
        col_idx += 1
        
        # M6
        U[2][col_idx] += 1 # A10
        U[0][col_idx] -= 1 # A00
        V[0 * p + col_offset][col_idx] += 1 # B00
        V[0 * p + col_offset + 1][col_idx] += 1 # B01
        W[(col_offset + 1) * n + 1][col_idx] += 1 # C11
        col_idx += 1
        
        # M7
        U[1][col_idx] += 1 # A01
        U[3][col_idx] -= 1 # A11
        V[1 * p + col_offset][col_idx] += 1 # B10
        V[1 * p + col_offset + 1][col_idx] += 1 # B11
        W[col_offset * n + 0][col_idx] += 1 # C00
        col_idx += 1

    def add_single_terms(col_offset):
        nonlocal col_idx
        # M1 = A00 B0k
        U[0][col_idx] += 1
        V[0 * p + col_offset][col_idx] += 1
        W[col_offset * n + 0][col_idx] += 1
        col_idx += 1
        
        # M2 = A01 B1k
        U[1][col_idx] += 1
        V[1 * p + col_offset][col_idx] += 1
        W[col_offset * n + 0][col_idx] += 1
        col_idx += 1
        
        # M3 = A10 B0k
        U[2][col_idx] += 1
        V[0 * p + col_offset][col_idx] += 1
        W[col_offset * n + 1][col_idx] += 1
        col_idx += 1
        
        # M4 = A11 B1k
        U[3][col_idx] += 1
        V[1 * p + col_offset][col_idx] += 1
        W[col_offset * n + 1][col_idx] += 1
        col_idx += 1

    add_strassen_terms(0)
    add_strassen_terms(2)
    add_strassen_terms(4)
    add_single_terms(6)
    
    # Verify
    def verify(U, V, W):
        for _ in range(10):
            A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
            C = [[0 for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C[i][k] += A[i][j] * B[j][k]
            
            C_calc = [[0 for _ in range(p)] for _ in range(n)]
            for r in range(rank):
                u_val = 0
                for i in range(n):
                    for j in range(m):
                        u_val += U[i * m + j][r] * A[i][j]
                v_val = 0
                for j in range(m):
                    for k in range(p):
                        v_val += V[j * p + k][r] * B[j][k]
                prod = u_val * v_val
                for k in range(p):
                    for i in range(n):
                        C_calc[i][k] += W[k * n + i][r] * prod
            
            if C != C_calc:
                return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
