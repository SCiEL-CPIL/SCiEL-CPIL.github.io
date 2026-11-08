"""Matrix multiplication 3,3,3 - rank 26 exact decomposition via 2x2+1x1 partitioning."""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 26
    
    # U: shape (n*m, rank) = (9, 26)
    # V: shape (m*p, rank) = (9, 26)
    # W: shape (p*n, rank) = (9, 26)
    #
    # Indexing conventions:
    #   U[i*m + j] corresponds to A[i][j]
    #   V[j*p + k] corresponds to B[j][k]
    #   W[k*n + i] corresponds to C[i][k]  (official trace convention)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # --- Block 1: Top-left 2x2 * Top-left 2x2 -> Rank 7 (Strassen) ---
    # A2 indices in U: A00=0, A01=1, A10=3, A11=4
    # B2 indices in V: B00=0, B01=1, B10=3, B11=4
    # C indices in W: C00=W[0], C01=W[3], C10=W[1], C11=W[4]
    #   W index for C[i][k] = k*3 + i
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # --- Block 2: A[0:2,2] * B[2,0:2] -> Rank 4 ---
    # U: A02=2, A12=5
    # V: B20=6, B21=7
    # C00(W0) += A02*B20, C01(W3) += A02*B21
    # C10(W1) += A12*B20, C11(W4) += A12*B21
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # --- Block 3: A[0:2,0:2] * B[0:2,2] -> Rank 4 ---
    # U: A00=0, A01=1, A10=3, A11=4
    # V: B02=2, B12=5
    # C02(W6) += A00*B02 + A01*B12
    # C12(W7) += A10*B02 + A11*B12
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # --- Block 4: A[0:2,2] * B[2,2] -> Rank 2 ---
    # U: A02=2, A12=5
    # V: B22=8
    # C02(W6) += A02*B22, C12(W7) += A12*B22
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # --- Block 5: A[2,0:2] * B[0:2,0:2] -> Rank 4 ---
    # U: A20=6, A21=7
    # V: B00=0, B01=1, B10=3, B11=4
    # C20(W2) += A20*B00 + A21*B10
    # C21(W5) += A20*B01 + A21*B11
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # --- Block 6: A[2,2] * B[2,0:2] -> Rank 2 ---
    # U: A22=8
    # V: B20=6, B21=7
    # C20(W2) += A22*B20, C21(W5) += A22*B21
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # --- Block 7: A[2,0:2] * B[0:2,2] -> Rank 2 ---
    # U: A20=6, A21=7
    # V: B02=2, B12=5
    # C22(W8) += A20*B02 + A21*B12
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # --- Block 8: A[2,2] * B[2,2] -> Rank 1 ---
    # U: A22=8
    # V: B22=8
    # C22(W8) += A22*B22
    add_term(8, 8, 8, 1)
    
    # Verify col == rank
    # assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
