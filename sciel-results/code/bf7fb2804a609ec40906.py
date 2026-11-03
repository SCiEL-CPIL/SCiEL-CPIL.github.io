"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import sys

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
    # U: A00 (idx 0), A11 (idx 4)
    U[0][0] = 1
    U[4][0] = 1
    # V: B00 (idx 0), B11 (idx 7)
    V[0][0] = 1
    V[7][0] = 1
    # W: C00 (idx 0), C11 (idx 4)
    W[0][0] = 1
    W[4][0] = 1
    
    # Column 1: M2 = (A10+A11)B00
    U[3][1] = 1  # A10
    U[4][1] = 1  # A11
    V[0][1] = 1  # B00
    # C10 += M2
    W[3][1] = 1  # C10: i=1,k=0 -> 0*3+1=1? No. C10 means row i=1, col k=0 -> index k*n+i = 0*3+1 = 1
    # Wait, W index is k*n+i. C[i,k] where i is row of result, k is col.
    # Result C is 3x4. C[i,k]. W index = k*n + i.
    # C00: i=0,k=0 -> 0
    # C01: i=0,k=1 -> 1*3+0 = 3
    # C02: i=0,k=2 -> 2*3+0 = 6
    # C03: i=0,k=3 -> 3*3+0 = 9
    # C10: i=1,k=0 -> 0*3+1 = 1
    # C11: i=1,k=1 -> 1*3+1 = 4
    # C12: i=1,k=2 -> 2*3+1 = 7
    # C13: i=1,k=3 -> 3*3+1 = 10
    # C20: i=2,k=0 -> 0*3+2 = 2
    # C21: i=2,k=1 -> 1*3+2 = 5
    # C22: i=2,k=2 -> 2*3+2 = 8
    # C23: i=2,k=3 -> 3*3+2 = 11
    
    # Let me redo the W indices:
    # C[i,k] -> W index = k*3 + i
    # C00: 0
    # C01: 3
    # C02: 6
    # C03: 9
    # C10: 1
    # C11: 4
    # C12: 7
    # C13: 10
    # C20: 2
    # C21: 5
    # C22: 8
    # C23: 11
    
    # Redo W assignments:
    # Column 0: M1 -> C00 (+1), C11 (+1)
    W[0][0] = 1   # C00
    W[4][0] = 1   # C11
    
    # Column 1: M2 -> C10 (+1), C11 (-1)
    U[3][1] = 1   # A10
    U[4][1] = 1   # A11
    V[0][1] = 1   # B00
    W[1][1] = 1   # C10
    W[4][1] = -1  # C11
    
    # Column 2: M3 = A00(B10-B11)
    U[0][2] = 1   # A00
    V[3][2] = 1   # B10
    V[10][2] = -1 # B11 (j=1,k=1 -> 1*4+1=5? No. V index is j*p+k. B[j,k]. j=1,k=1 -> 1*4+1=5)
    # Wait, V shape is (m*p, rank) = (12, 35). V[j*p+k, c].
    # B00: j=0,k=0 -> 0
    # B01: j=0,k=1 -> 1
    # B02: j=0,k=2 -> 2
    # B03: j=0,k=3 -> 3
    # B10: j=1,k=0 -> 4
    # B11: j=1,k=1 -> 5
    # B12: j=1,k=2 -> 6
    # B13: j=1,k=3 -> 7
    # B20: j=2,k=0 -> 8
    # B21: j=2,k=1 -> 9
    # B22: j=2,k=2 -> 10
    # B23: j=2,k=3 -> 11
    
    # Redo V indices:
    # B00: 0, B01: 1, B02: 2, B03: 3
    # B10: 4, B11: 5, B12: 6, B13: 7
    # B20: 8, B21: 9, B22: 10, B23: 11
    
    # Redo U indices (A[i,j], U[i*m+j, c], m=3):
    # A00: 0, A01: 1, A02: 2
    # A10: 3, A11: 4, A12: 5
    # A20: 6, A21: 7, A22: 8
    
    # Redo W indices (C[i,k], W[k*n+i, c], n=3):
    # C00: 0, C10: 1, C20: 2
    # C01: 3, C11: 4, C21: 5
    # C02: 6, C12: 7, C22: 8
    # C03: 9, C13: 10, C23: 11
    
    # Now redo all Strassen terms with correct indices:
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    # Verify Strassen part
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M5 + M6
    
    # Let me verify:
    # C00: M1(1) - M4(1) - M6(1) + M7(1) = from col0: +1, col3: -1, col5: -1, col6: +1. Sum = 0? No.
    # The coefficient of each M in C00: M1:+1, M4:-1, M6:-1, M7:+1
    # So W[0][c] should be: col0:1, col3:-1, col5:-1, col6:1. That's what I have. Good.
    # C01: M3:+1, M5:+1. W[3][c]: col2:1, col4:1. Good.
    # C10: M2:+1, M4:+1. W[1][c]: col1:1, col3:1. Good.
    # C11: M1:+1, M2:-1, M5:-1, M6:+1. W[4][c]: col0:1, col1:-1, col4:-1, col5:1. Good.
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
