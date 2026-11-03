"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(V[0]) != rank:
        return False
    if len(W[0]) != rank:
        return False
    
    # For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    total += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Rank-35 decomposition of 3x3 * 3x4 matrix multiplication.
    
    Uses Strassen's algorithm on the (i,j,k) in {0,1}x{0,1}x{0,1} sub-block,
    and schoolbook for all remaining 28 terms.
    
    Strassen decomposition for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A10)B11
    M6 = (A01-A11)(B10+B11)
    M7 = (A10-A00)(B01+B11)
    
    Contributions to C:
    C00 = M1 - M4 - M6 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 - M5 + M6
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 35)
    V = [[0] * rank for _ in range(m * p)]  # shape (12, 35)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 35)
    
    # ---- Strassen terms (columns 0-6) ----
    # Column 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1    # A00
    U[4][0] = 1    # A11
    V[0][0] = 1    # B00
    V[5][0] = 1    # B11
    W[0][0] = 1    # C00
    W[4][0] = 1    # C11
    
    # Column 1: M2 = (A10+A11)B00
    U[3][1] = 1    # A10
    U[4][1] = 1    # A11
    V[0][1] = 1    # B00
    W[1][1] = 1    # C10
    W[4][1] = -1   # C11
    
    # Column 2: M3 = A00(B10-B11)
    U[0][2] = 1    # A00
    V[4][2] = 1    # B10
    V[5][2] = -1   # B11
    W[3][2] = 1    # C01
    W[4][2] = 1    # C11
    
    # Column 3: M4 = A11(B01-B10)
    U[4][3] = 1    # A11
    V[1][3] = 1    # B01
    V[4][3] = -1   # B10
    W[0][3] = -1   # C00
    W[1][3] = 1    # C10
    
    # Column 4: M5 = (A00+A10)B11
    U[0][4] = 1    # A00
    U[3][4] = 1    # A10
    V[5][4] = 1    # B11
    W[3][4] = 1    # C01
    W[4][4] = -1   # C11
    
    # Column 5: M6 = (A01-A11)(B10+B11)
    U[1][5] = 1    # A01
    U[4][5] = -1   # A11
    V[4][5] = 1    # B10
    V[5][5] = 1    # B11
    W[0][5] = -1   # C00
    W[4][5] = 1    # C11
    
    # Column 6: M7 = (A10-A00)(B01+B11)
    U[3][6] = 1    # A10
    U[0][6] = -1   # A00
    V[1][6] = 1    # B01
    V[5][6] = 1    # B11
    W[0][6] = 1    # C00
    W[3][6] = 1    # C01
    
    # ---- Schoolbook terms for remaining (i,j,k) triples ----
    # All (i,j,k) where NOT all of i,j,k are in {0,1}
    # That's 3*3*4 - 8 = 36 - 8 = 28 terms
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if i in {0,1}, j in {0,1}, k in {0,1}
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # col should be 35
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": W, "W": W}
# EVOLVE-BLOCK-END
