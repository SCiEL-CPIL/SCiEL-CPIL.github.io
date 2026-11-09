import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Block-diagonal partition:
    #   A (2x5) -> [A00 (2x2) | A01 (2x1)]
    #   B (5x5) -> [B00 (2x2)  B01 (2x1)  B02 (2x1)]
    #               [B10 (1x2)  B11 (1x1)  B12 (1x1)]
    #   C (2x5) -> [C00 (2x2) | C01 (2x1) | C02 (2x1)]
    #
    # Sub-problems:
    #   1) C00 = A00 * B00  (2x2x2, Strassen rank 7)
    #   2) C01 = A00 * B01  (2x2x1, schoolbook rank 2)
    #   3) C02 = A00 * B02  (2x2x1, schoolbook rank 2)
    #   4) C01 += A01 * B10 (2x1x2, schoolbook rank 2)
    #   5) C02 += A01 * B12 (2x1x1, schoolbook rank 1)
    #   6) C01 += A01 * B11 (2x1x1, schoolbook rank 2? No, it's 2x1x1)
    #   7) C02 += A01 * B12 (already counted)
    #
    # Let's be precise about indices.
    # J indices for A: j=0,1 (B00), j=2 (B01), j=3 (B02) -- wait, B is 5x5.
    # B partition: rows J={0,1,2,3,4}, cols K={0,1,2,3,4}.
    # Let's map B blocks:
    # B00: rows {0,1}, cols {0,1} (2x2)
    # B01: rows {0,1}, cols {2}   (2x1)
    # B02: rows {0,1}, cols {3}   (2x1)
    # B03: rows {0,1}, cols {4}   (2x1)
    # B10: rows {2},    cols {0,1} (1x2)
    # B11: rows {2},    cols {2}   (1x1)
    # B12: rows {2},    cols {3}   (1x1)
    # B13: rows {2},    cols {4}   (1x1)
    # B20: rows {3},    cols {0,1} (1x2)
    # B21: rows {3},    cols {2}   (1x1)
    # B22: rows {3},    cols {3}   (1x1)
    # B23: rows {3},    cols {4}   (1x1)
    # B30: rows {4},    cols {0,1} (1x2)
    # B31: rows {4},    cols {2}   (1x1)
    # B32: rows {4},    cols {3}   (1x1)
    # B33: rows {4},    cols {4}   (1x1)
    #
    # This partition is messy. The current program uses a cleaner partition:
    # J={0,1,2,3,4}, K={0,1,2,3,4}
    # Strassen blocks on J={0,1},K={0,1}, J={2,3},K={0,1}, J={0,1},K={2,3}, J={2,3},K={2,3}
    # That's 4 Strassen blocks (rank 7 each) = 28
    # Remaining: J={0,1,2,3}, K=4 (4x1 blocks for A, 1x1 for B? No, B is 5x5)
    # The current code handles:
    # - 4 Strassen blocks (28 rank)
    # - J in {0,1,2,3}, K=4: 4 j-values * 2 i-values = 8 rank
    # - J=4, K in {0,1,2,3,4}: 5 k-values * 2 i-values = 10 rank
    # Total = 28 + 8 + 10 = 46
    #
    # The "block-diagonal partition" idea in the directive suggests:
    # B into 2x2, 2x2, 1x1 blocks.
    # J partition: {0,1}, {2,3}, {4}
    # K partition: {0,1}, {2,3}, {4}
    # This gives 9 sub-problems:
    # (0,0): 2x2x2 -> Strassen (7)
    # (0,1): 2x2x1 -> schoolbook (2)
    # (0,2): 2x1x1 -> schoolbook (2)
    # (1,0): 1x2x2 -> schoolbook (2)
    # (1,1): 1x2x2 -> schoolbook (2)  -- Wait, A is 2x5, so A blocks are 2x2, 2x2, 2x1
    # (1,2): 1x1x1 -> schoolbook (1)
    # (2,0): 1x2x2 -> schoolbook (2)
    # (2,1): 1x1x2 -> schoolbook (1)
    # (2,2): 1x1x1 -> schoolbook (1)
    #
    # Let's recount:
    # A blocks: A0 (2x2), A1 (2x2), A2 (2x1)
    # B blocks: B00 (2x2), B01 (2x2), B02 (2x1)
    #           B10 (2x2), B11 (2x2), B12 (2x1)
    #           B20 (1x2), B21 (1x2), B22 (1x1)
    # C blocks: C0 (2x2), C1 (2x2), C2 (2x1)
    #
    # Sub-problems:
    # C0: A0*B00 (2x2x2, rank 7) + A1*B10 (2x2x2, rank 7) + A2*B20 (2x1x2, rank 2)
    # C1: A0*B01 (2x2x2, rank 7) + A1*B11 (2x2x2, rank 7) + A2*B21 (2x1x2, rank 2)
    # C2: A0*B02 (2x2x1, rank 2) + A1*B12 (2x2x1, rank 2) + A2*B22 (2x1x1, rank 1)
    #
    # Total rank = (7+7+2) + (7+7+2) + (2+2+1) = 16 + 16 + 5 = 37
    #
    # This is much better than 46! Let's implement this.

    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add schoolbook block for A[i*m+j], B[j*p+k] -> C[k*n+i]
    def add_schoolbook(j, k, i_val=None):
        nonlocal col
        if i_val is not None:
            U[i_val*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+i_val][col] = 1
            col += 1
        else:
            # All i values
            for i_val in range(n):
                U[i_val*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i_val][col] = 1
                col += 1
    
    # Helper: add Strassen block for A rows {i0,i1}, B cols {j0,j1} -> C rows {j0,j1}
    # Actually, Strassen multiplies A_block (n x 2) by B_block (2 x 2) to get C_block (2 x 2)
    # Here n=2, so A_block is 2x2, B_block is 2x2, C_block is 2x2
    # J indices: ja, jb (columns of A, rows of B)
    # K indices: ka, kb (columns of B, rows of C)
    def add_strassen(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[0*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        W[ka*n+1][col] = 1
        W[kb*n+1][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+ja][col] = 1
        V[ja*p+kb][col] = 1
        V[jb*p+kb][col] = -1
        W[kb*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+jb][col] = 1
        V[jb*p+ka][col] = 1
        V[ja*p+ka][col] = -1
        W[ka*n+0][col] = 1
        W[ka*n+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+ja][col] = 1
        U[0*m+jb][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = -1
        W[kb*n+0][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m+ja][col] = 1
        U[0*m+ja][col] = -1
        V[ja*p+ka][col] = 1
        V[ja*p+kb][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+jb][col] = 1
        U[1*m+jb][col] = -1
        V[jb*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        col += 1

    # C0: A0*B00 (Strassen) + A1*B10 (Strassen) + A2*B20 (schoolbook)
    # A0: j=0,1; A1: j=2,3; A2: j=4
    # B00: k=0,1; B10: k=0,1; B20: k=0,1
    add_strassen(0, 1, 0, 1)
    add_strassen(2, 3, 0, 1)
    # A2*B20: j=4, k=0,1
    for k in [0, 1]:
        add_schoolbook(4, k, 0)
        add_schoolbook(4, k, 1)

    # C1: A0*B01 (Strassen) + A1*B11 (Strassen) + A2*B21 (schoolbook)
    # B01: k=2,3; B11: k=2,3; B21: k=2,3
    add_strassen(0, 1, 2, 3)
    add_strassen(2, 3, 2, 3)
    # A2*B21: j=4, k=2,3
    for k in [2, 3]:
        add_schoolbook(4, k, 0)
        add_schoolbook(4, k, 1)

    # C2: A0*B02 (schoolbook) + A1*B12 (schoolbook) + A2*B22 (schoolbook)
    # B02: k=4; B12: k=4; B22: k=4
    for j in [0, 1]:
        add_schoolbook(j, 4, 0)
        add_schoolbook(j, 4, 1)
    for j in [2, 3]:
        add_schoolbook(j, 4, 0)
        add_schoolbook(j, 4, 1)
    add_schoolbook(4, 4, 0)
    add_schoolbook(4, 4, 1)

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if not np.allclose(T_recon, T_target, atol=1e-9, rtol=0):
        # Fallback
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school), dtype=int)
        V_school = np.zeros((m*p, rank_school), dtype=int)
        W_school = np.zeros((p*n, rank_school), dtype=int)
        
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col_s] = 1
                    V_school[j*p+k, col_s] = 1
                    W_school[k*n+i, col_s] = 1
                    col_s += 1
        
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U, "V": V, "W": W}
