"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition of 3x3 matrix multiplication
    # Using Strassen for the 2x2 core plus border terms.
    # Index conventions:
    #   U index for A[i][j] = i*3 + j  (row-major, 9 entries)
    #   V index for B[i][j] = i*3 + j  (row-major, 9 entries)
    #   W index for C[i][k] = k*3 + i  (column-major per trace convention)
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
    
    # === Part 1: Top-left 2x2 * 2x2 via Strassen (7 terms) ===
    # A2 = A[0:2, 0:2], B2 = B[0:2, 0:2]
    # U: A00=0, A01=1, A10=3, A11=4
    # V: B00=0, B01=1, B10=3, B11=4
    # W: C00=0, C01=3, C10=1, C11=4
    
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
    
    # === Part 2: a * b^T (2x1 * 1x2) -> 4 terms ===
    # a = A[0:2, 2]: U indices A02=2, A12=5
    # b^T = B[2, 0:2]: V indices B20=6, B21=7
    # C00(W0) += A02*B20, C01(W3) += A02*B21
    # C10(W1) += A12*B20, C11(W4) += A12*B21
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # === Part 3: A2 * b (2x2 * 2x1) -> 4 terms ===
    # A2: U 0,1,3,4
    # b = B[0:2, 2]: V indices B02=2, B12=5
    # C02(W6) += A00*B02 + A01*B12
    # C12(W7) += A10*B02 + A11*B12
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # === Part 4: a * b3 (2x1 * 1x1) -> 2 terms ===
    # a: U 2,5
    # b3 = B[2,2]: V index 8
    # C02(W6) += A02*B22
    # C12(W7) += A12*B22
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # === Part 5: a^T * B2 (1x2 * 2x2) -> 4 terms ===
    # a^T = A[2, 0:2]: U indices A20=6, A21=7
    # B2: V 0,1,3,4
    # C20(W2) += A20*B00 + A21*B10
    # C21(W5) += A20*B01 + A21*B11
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # === Part 6: a3 * b^T (1x1 * 1x2) -> 2 terms ===
    # a3 = A[2,2]: U index 8
    # b^T = B[2, 0:2]: V indices B20=6, B21=7
    # C20(W2) += A22*B20
    # C21(W5) += A22*B21
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # === Part 7: a^T * b (1x2 * 2x1) -> 2 terms ===
    # a^T: U 6,7
    # b: V 2,5
    # C22(W8) += A20*B02 + A21*B12
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # === Part 8: a3 * b3 (1x1 * 1x1) -> 1 term ===
    # a3: U 8, b3: V 8
    # C22(W8) += A22*B22
    add_term(8, 8, 8, 1)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin before 120s limit
        pass  # We already have our result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
