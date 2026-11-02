import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, val
    def add_term(u_idx, v_idx, w_idx, val, col):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    col = 0
    
    # Block 1: A_left (2x2) * B_top_left (2x2)
    # A indices: i in 0..1, j in 0..1 -> U[i*3 + j]
    # B indices: j in 0..1, k in 0..1 -> V[j*5 + k]
    # C indices: k in 0..1, i in 0..1 -> W[k*2 + i]
    
    # Strassen 1
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, col) # A00 B00
    add_term(3, 0, 0, 1, col) # A11 B00 -- wait, U index for A11 is 1*3+1=4? No.
    
    # Let's map indices carefully.
    # A is 2x3. U row index = i*m + j = i*3 + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    # B is 3x5. V row index = j*p + k = j*5 + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 5
    # j=1, k=1 -> 6
    
    # C is 2x5. W row index = k*n + i = k*2 + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Strassen 1 (A_left, B_top_left)
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[3], U[4]
    # B_top_left: B[0,0], B[0,1], B[1,0], B[1,1] -> V[0], V[1], V[5], V[6]
    
    # M1 = (A00+A11)(B00+B11)
    # Terms: A00B00, A00B11, A11B00, A11B11
    # We need to add coefficients to U, V, W such that sum U*V*W gives the product.
    # For rank 1 term: U_vec * V_vec * W_vec.
    # We are building U, V, W columns.
    # Col c corresponds to one rank-1 term.
    
    # M1:
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C01(1), C10(1), C11(1) ? No.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1:
    # U contributions: A00: 1, A11: 1
    # V contributions: B00: 1, B11: 1
    # W contributions: C00: 1, C01: 0, C10: 0, C11: 1 ?
    # Wait, M1 contributes to C00 and C11.
    # C00 index: k=0, i=0 -> 0.
    # C11 index: k=1, i=1 -> 3.
    # C01 index: k=0, i=1 -> 1.
    # C10 index: k=1, i=0 -> 2.
    
    # M1:
    U[0][col] += 1
    U[4][col] += 1
    V[0][col] += 1
    V[6][col] += 1
    W[0][col] += 1 # C00
    W[3][col] += 1 # C11
    col += 1
    
    # M2 = (A10+A11)B00
    # U: A10(1), A11(1)
    # V: B00(1)
    # W: C10(1)
    U[3][col] += 1
    U[4][col] += 1
    V[0][col] += 1
    W[2][col] += 1 # C10
    col += 1
    
    # M3 = A00(B01-B11)
    # U: A00(1)
    # V: B01(1), B11(-1)
    # W: C01(1)
    U[0][col] += 1
    V[1][col] += 1
    V[6][col] += -1
    W[1][col] += 1 # C01
    col += 1
    
    # M4 = A11(B10-B00)
    # U: A11(1)
    # V: B10(1), B00(-1)
    # W: C00(1), C10(1)
    U[4][col] += 1
    V[5][col] += 1
    V[0][col] += -1
    W[0][col] += 1 # C00
    W[2][col] += 1 # C10
    col += 1
    
    # M5 = (A00+A01)B11
    # U: A00(1), A01(1)
    # V: B11(1)
    # W: C01(-1), C11(-1)
    U[0][col] += 1
    U[1][col] += 1
    V[6][col] += 1
    W[1][col] += -1 # C01
    W[3][col] += -1 # C11
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10(1), A00(-1)
    # V: B00(1), B01(1)
    # W: C11(1)
    U[3][col] += 1
    U[0][col] += -1
    V[0][col] += 1
    V[1][col] += 1
    W[3][col] += 1 # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01(1), A11(-1)
    # V: B10(1), B11(1)
    # W: C00(1)
    U[1][col] += 1
    U[4][col] += -1
    V[5][col] += 1
    V[6][col] += 1
    W[0][col] += 1 # C00
    col += 1
    
    # Block 2: A_left (2x2) * B_top_mid (2x2)
    # B_top_mid: B[0,2], B[0,3], B[1,2], B[1,3]
    # V indices:
    # B[0,2] -> 2
    # B[0,3] -> 3
    # B[1,2] -> 7
    # B[1,3] -> 8
    # C indices:
    # C[0,2] -> k=2, i=0 -> 4
    # C[0,3] -> k=3, i=0 -> 6
    # C[1,2] -> k=2, i=1 -> 5
    # C[1,3] -> k=3, i=1 -> 7
    
    # M1
    U[0][col] += 1
    U[4][col] += 1
    V[2][col] += 1
    V[8][col] += 1
    W[4][col] += 1 # C02
    W[7][col] += 1 # C13
    col += 1
    
    # M2
    U[3][col] += 1
    U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1 # C12
    col += 1
    
    # M3
    U[0][col] += 1
    V[3][col] += 1
    V[8][col] += -1
    W[6][col] += 1 # C03
    col += 1
    
    # M4
    U[4][col] += 1
    V[7][col] += 1
    V[2][col] += -1
    W[4][col] += 1 # C02
    W[5][col] += 1 # C12
    col += 1
    
    # M5
    U[0][col] += 1
    U[1][col] += 1
    V[8][col] += 1
    W[6][col] += -1 # C03
    W[7][col] += -1 # C13
    col += 1
    
    # M6
    U[3][col] += 1
    U[0][col] += -1
    V[2][col] += 1
    V[3][col] += 1
    W[7][col] += 1 # C13
    col += 1
    
    # M7
    U[1][col] += 1
    U[4][col] += -1
    V[7][col] += 1
    V[8][col] += 1
    W[4][col] += 1 # C02
    col += 1
    
    # Block 3: A_left (2x2) * B_top_right (2x1)
    # B_top_right: B[0,4], B[1,4]
    # V indices:
    # B[0,4] -> 4
    # B[1,4] -> 9
    # C indices:
    # C[0,4] -> k=4, i=0 -> 8
    # C[1,4] -> k=4, i=1 -> 9
    
    # Rank 4 terms:
    # T1 = A00 B04 -> C04
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1
    col += 1
    
    # T2 = A01 B14 -> C04
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1
    
    # T3 = A10 B04 -> C14
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1
    col += 1
    
    # T4 = A11 B14 -> C14
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1
    
    # Block 4: A_right (2x1) * B_bottom (1x5)
    # A_right: A[0,2], A[1,2]
    # U indices:
    # A[0,2] -> 2
    # A[1,2] -> 5
    # B_bottom: B[2,0]..B[2,4]
    # V indices:
    # B[2,0] -> 10
    # B[2,1] -> 11
    # B[2,2] -> 12
    # B[2,3] -> 13
    # B[2,4] -> 14
    # C indices:
    # C[0,k] -> k*2 + 0
    # C[1,k] -> k*2 + 1
    
    for k in range(5):
        # T1 = A02 B2k -> C0k
        U[2][col] += 1
        V[10+k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        # T2 = A12 B2k -> C1k
        U[5][col] += 1
        V[10+k][col] += 1
        W[2*k+1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
