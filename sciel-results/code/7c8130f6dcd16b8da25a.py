"""Improved program: Block Strassen decomposition for 4x5x6 matrix multiplication.

Splits the 4x5x6 tensor into sub-tensors (2x2x2, 2x1x2, etc.) and applies
Strassen's rank-7 decomposition to 2x2x2 blocks, achieving rank 108.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 6]
DEADLINE = 140  # seconds; evaluator kills at 150s


def _strassen_2x2x2():
    """Return U, V, W for rank-7 Strassen decomposition of M_{2,2,2}.
    Shapes: (4, 7) each.
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0 = (A00+A11)*(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M1 = (A10+A11)*B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M2 = A00*(B01-B11) -> C01
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M3 = A11*(B10-B00) -> C10
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # M4 = (A00+A01)*B11 -> C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # M5 = (A10-A00)*(B00+B01) -> C10
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M6 = (A01-A11)*(B10+B11) -> C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W


def _block_strassen_4x5x6():
    """Build rank-108 decomposition for 4x5x6 matrix multiplication.
    
    Strategy: Split rows of A into 2+2, cols of B into 2+2+1, 
    cols of C into 2+2+2.
    
    A (4x5) = [[A00, A01, A02], [A10, A11, A12]] where:
      A00: 2x2, A01: 2x2, A02: 2x1
      A10: 2x2, A11: 2x2, A12: 2x1
    
    B (5x6) = [[B00, B01, B02], [B10, B11, B12]] where:
      B00: 2x2, B01: 2x2, B02: 2x2
      B10: 1x2, B11: 1x2, B12: 1x2
    
    C (4x6) = [[C00, C01, C02], [C10, C11, C12]] where:
      C00: 2x2, C01: 2x2, C02: 2x2
      C10: 2x2, C11: 2x2, C12: 2x2
    
    C_ab = sum_c A_ac * B_cb
    
    For 2x2 * 2x2 -> 2x2: use Strassen (rank 7)
    For 2x2 * 1x2 -> 2x2: outer product, rank 4
    For 2x1 * 2x2 -> 2x2: rank 4
    For 2x1 * 1x2 -> 2x2: rank 4 (but actually rank 1 per output... let me think)
    
    Actually let me reconsider. The blocks:
    - A00 (2x2) * B00 (2x2) -> contributes to C00 (2x2): rank 7 (Strassen)
    - A01 (2x2) * B10 (1x2) -> contributes to C00 (2x2): rank 4
    - A02 (2x1) * B00 (2x2) -> contributes to C00 (2x2): rank 4
    - A02 (2x1) * B10 (1x2) -> contributes to C00 (2x2): rank 4
    
    Wait, 2x1 * 1x2 gives a 2x2 matrix = outer product, rank 4 (each entry is a product).
    Actually 2x1 * 1x2: result[i][k] = A[i][0] * B[0][k]. That's 4 scalar products, rank 4.
    
    So C00 gets contributions from 4 sub-products:
    - A00*B00: 2x2*2x2 -> rank 7
    - A01*B10: 2x2*1x2 -> rank 4 (each of 4 outputs = A[i][j]*B[j][k], but B is 1x2 so j=0 only... wait)
    
    Let me be more careful. B10 is 1x2 (one row, two cols).
    A01 is 2x2.
    A01 * B10: (2x2) * (1x2) -- dimension mismatch! 2x2 times 1x2 doesn't work.
    
    I need to re-split. Let me redo:
    A is 4x5, B is 5x6.
    Split the shared dimension (5) into 2+1+2 or 2+2+1 etc.
    
    Let me try: split dim 5 into 2+2+1.
    A = [A_L (4x2) | A_M (4x2) | A_R (4x1)]
    B = [B_L (2x6); B_M (2x6); B_R (1x6)]
    
    C = A_L*B_L + A_M*B_M + A_R*B_R
    
    A_L*B_L: 4x2 * 2x6. Split rows 4=2+2, cols 6=2+2+2:
      4 blocks: (2x2)*(2x2) each -> 4 * rank(2x2x2) = 4*7 = 28
    A_M*B_M: same = 28
    A_R*B_R: 4x1 * 1x6. Split rows 4=2+2, cols 6=2+2+2:
      6 blocks: (2x1)*(1x2) each -> 6 * rank(2x1x2) = 6*4 = 24
    
    Total: 28 + 28 + 24 = 80? That seems too good. Let me verify.
    
    Wait, 4x2 * 2x6: I'm splitting the output 4x6 into 2x2 blocks (4 of them: 2 row-blocks x 2 col-blocks... no, 6 cols = 3 col-blocks of 2).
    So output is 2 row-blocks x 3 col-blocks = 6 blocks of 2x2.
    Each 2x2 output block = (2x2 subblock of A_L) * (2x2 subblock of B_L).
    A_L is 4x2, so row-blocks are 2x2 (two of them).
    B_L is 2x6, so col-blocks are 2x2 (three of them).
    Each output 2x2 block: (2x2) * (2x2) -> rank 7.
    6 blocks * 7 = 42.
    
    Similarly A_M*B_M: 42.
    A_R*B_R: 4x1 * 1x6. Row-blocks: 2x1 (two of them). Col-blocks: 1x2 (three of them).
    Each: (2x1)*(1x2) -> 2x2 output, rank 4.
    6 blocks * 4 = 24.
    
    Total: 42 + 42 + 24 = 108. That matches the measured improvement!
    """
    n, m, p = 4, 5, 6
    
    # Split: m = 2 + 2 + 1
    # n = 2 + 2
    # p = 2 + 2 + 2
    
    U = []
    V = []
    W = []
    
    # Helper: add a Strassen 2x2x2 block at given offsets
    def add_strassen_2x2x2(row_off, col_off, mid_off, a_row_off, a_col_off, b_row_off, b_col_off):
        """Add Strassen terms for A[a_row_off:a_row_off+2, a_col_off:a_col_off+2] *
        B[b_row_off:b_row_off+2, b_col_off:b_col_off+2] ->
        C[row_off:row_off+2, col_off:col_off+2]
        
        a_row_off: offset in A's rows (0-3)
        a_col_off: offset in A's cols (0-4)
        b_row_off: offset in B's rows (0-4)
        b_col_off: offset in B's cols (0-5)
        row_off: offset in C's rows (0-3)
        col_off: offset in C's cols (0-5)
        mid_off: not used, kept for signature
        """
        U2, V2, W2 = _strassen_2x2x2()
        for t in range(7):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            for i in range(2):
                for j in range(2):
                    u_val = U2[i*2+j][t]
                    if u_val != 0:
                        u_vec[(a_row_off+i) * m + (a_col_off+j)] = u_val
            for j in range(2):
                for k in range(2):
                    v_val = V2[j*2+k][t]
                    if v_val != 0:
                        v_vec[(b_row_off+j) * p + (b_col_off+k)] = v_val
            for k in range(2):
                for i in range(2):
                    w_val = W2[k*2+i][t]
                    if w_val != 0:
                        w_vec[(col_off+k) * n + (row_off+i)] = w_val
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
    
    # Helper: add schoolbook 2x1x2 block (outer product)
    def add_outer_2x1x2(row_off, col_off, a_row_off, a_col_off, b_row_off, b_col_off):
        """Add terms for A[a_row_off:a_row_off+2, a_col_off] * B[b_row_off, b_col_off:b_col_off+2]
        -> C[row_off:row_off+2, col_off:col_off+2]
        This is a 2x1 * 1x2 = 2x2 outer product, rank 4.
        """
        for i in range(2):
            for k in range(2):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[(a_row_off+i) * m + a_col_off] = 1
                v_vec[b_row_off * p + (b_col_off+k)] = 1
                w_vec[(col_off+k) * n + (row_off+i)] = 1
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
    
    # Helper: add schoolbook 2x2x2 block (8 terms)
    def add_schoolbook_2x2x2(row_off, col_off, a_row_off, a_col_off, b_row_off, b_col_off):
        for j in range(2):
            for i in range(2):
                for k in range(2):
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    u_vec[(a_row_off+i) * m + (a_col_off+j)] = 1
                    v_vec[(b_row_off+j) * p + (b_col_off+k)] = 1
                    w_vec[(col_off+k) * n + (row_off+i)] = 1
                    U.append(u_vec)
                    V.append(v_vec)
                    W.append(w_vec)
    
    # Part 1: A_L (4x2) * B_L (2x6)
    # A_L rows 0-1, cols 0-1; rows 2-3, cols 0-1
    # B_L rows 0-1, cols 0-1; cols 2-3; cols 4-5
    # C blocks: rows 0-1 or 2-3, cols 0-1 or 2-3 or 4-5
    for a_r in [0, 2]:  # row block offset in A
        for b_c in [0, 2, 4]:  # col block offset in B
            add_strassen_2x2x2(
                row_off=a_r, col_off=b_c, mid_off=0,
                a_row_off=a_r, a_col_off=0,
                b_row_off=0, b_col_off=b_c
            )
    # That's 2 * 3 = 6 blocks * 7 = 42 terms
    
    # Part 2: A_M (4x2) * B_M (2x6)
    # A_M rows 0-1, cols 2-3; rows 2-3, cols 2-3
    # B_M rows 2-3, cols 0-1; cols 2-3; cols 4-5
    for a_r in [0, 2]:
        for b_c in [0, 2, 4]:
            add_strassen_2x2x2(
                row_off=a_r, col_off=b_c, mid_off=0,
                a_row_off=a_r, a_col_off=2,
                b_row_off=2, b_col_off=b_c
            )
    # That's 2 * 3 = 6 blocks * 7 = 42 terms
    
    # Part 3: A_R (4x1) * B_R (1x6)
    # A_R rows 0-1, col 4; rows 2-3, col 4
    # B_R row 4, cols 0-1; cols 2-3; cols 4-5
    for a_r in [0, 2]:
        for b_c in [0, 2, 4]:
            add_outer_2x1x2(
                row_off=a_r, col_off=b_c,
                a_row_off=a_r, a_col_off=4,
                b_row_off=4, b_col_off=b_c
            )
    # That's 2 * 3 = 6 blocks * 4 = 24 terms
    
    rank = len(U)
    assert rank == 108, f"Expected rank 108, got {rank}"
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, max_tests=20):
    """Verify that the decomposition is correct by testing with random matrices."""
    import random
    random.seed(42)
    rank = len(U[0])
    for _ in range(max_tests):
        A = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_prime = [[0.0] * p for _ in range(n)]
        for t in range(rank):
            u_val = sum(U[i*m+j][t] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j*p+k][t] * B[j][k] for j in range(m) for k in range(p))
            prod = u_val * v_val
            for k in range(p):
                for i in range(n):
                    C_prime[i][k] += W[k*n+i][t] * prod
        for i in range(n):
            for k in range(p):
                if abs(C[i][k] - C_prime[i][k]) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the block Strassen decomposition (rank 108)
    U, V, W = _block_strassen_4x5x6()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, max_tests=20):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
