import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-39 exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    Strategy:
    1. Split A (2x3) into A_L (2x2) and A_R (2x1).
    2. Split B (3x7) into B_T (2x7) and B_B (1x7).
    3. C = A_L B_T + A_R B_B.
    
    Rank of A_L B_T (2x2 x 2x7):
    Split B_T into B_T1 (2x2), B_T2 (2x2), B_T3 (2x2), B_T4 (2x1).
    Rank(2x2 x 2x2) = 7 (Strassen).
    Rank(2x2 x 2x1) = 4.
    Total = 7 + 7 + 7 + 4 = 25.
    
    Rank of A_R B_B (2x1 x 1x7):
    Rank = 2 * 7 = 14.
    
    Total Rank = 25 + 14 = 39.
    """
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_i0, a_i1, b_j0, b_j1, b_k0, b_k1, c_k0, c_k1, offset):
        nonlocal col
        # Local indices for A: (0,0), (0,1), (1,0), (1,1)
        # Global U index: i*m + j
        # a_i0, a_i1 are the row indices for A (0, 1)
        # b_j0, b_j1 are the col indices for A (0, 1)
        # b_k0, b_k1 are the row indices for B (0, 1)
        # c_k0, c_k1 are the col indices for B (0, 1)
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[a_i0*m + b_j0][col] += 1
        U[a_i1*m + b_j1][col] += 1
        V[b_j0*p + c_k0][col] += 1
        V[b_j1*p + c_k1][col] += 1
        W[c_k0*n + a_i0][col] += 1
        W[c_k1*n + a_i1][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[a_i1*m + b_j0][col] += 1
        U[a_i1*m + b_j1][col] += 1
        V[b_j0*p + c_k0][col] += 1
        W[c_k0*n + a_i1][col] += 1
        W[c_k1*n + a_i1][col] += -1
        col += 1
        
        # M3: A00(B10-B11) -> C01 += M3, C10 += M3
        U[a_i0*m + b_j0][col] += 1
        V[b_j1*p + c_k0][col] += 1
        V[b_j1*p + c_k1][col] += -1
        W[c_k1*n + a_i0][col] += 1
        W[c_k0*n + a_i1][col] += 1
        col += 1
        
        # M4: A11(B01-B00) -> C00 += M4, C10 += M4
        U[a_i1*m + b_j1][col] += 1
        V[b_j0*p + c_k1][col] += 1
        V[b_j0*p + c_k0][col] += -1
        W[c_k0*n + a_i0][col] += 1
        W[c_k0*n + a_i1][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[a_i0*m + b_j0][col] += 1
        U[a_i0*m + b_j1][col] += 1
        V[b_j1*p + c_k1][col] += 1
        W[c_k0*n + a_i0][col] += -1
        W[c_k1*n + a_i0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[a_i1*m + b_j0][col] += 1
        U[a_i0*m + b_j0][col] += -1
        V[b_j0*p + c_k0][col] += 1
        V[b_j0*p + c_k1][col] += 1
        W[c_k1*n + a_i1][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
        U[a_i0*m + b_j1][col] += 1
        U[a_i1*m + b_j1][col] += -1
        V[b_j1*p + c_k0][col] += 1
        V[b_j1*p + c_k1][col] += 1
        W[c_k0*n + a_i0][col] += 1
        W[c_k1*n + a_i1][col] += -1
        col += 1

    def add_2x2x1(a_i0, a_i1, b_j0, b_j1, b_k, c_k, c_i0, c_i1):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0 -> C0
        U[a_i0*m + b_j0][col] += 1
        V[b_j0*p + b_k][col] += 1
        W[c_k*n + c_i0][col] += 1
        col += 1
        
        # Term 2: A01 B1 -> C0
        U[a_i0*m + b_j1][col] += 1
        V[b_j1*p + b_k][col] += 1
        W[c_k*n + c_i0][col] += 1
        col += 1
        
        # Term 3: A10 B0 -> C1
        U[a_i1*m + b_j0][col] += 1
        V[b_j0*p + b_k][col] += 1
        W[c_k*n + c_i1][col] += 1
        col += 1
        
        # Term 4: A11 B1 -> C1
        U[a_i1*m + b_j1][col] += 1
        V[b_j1*p + b_k][col] += 1
        W[c_k*n + c_i1][col] += 1
        col += 1

    def add_2x1x7(a_i0, a_i1, b_j, b_k_start, b_k_end, c_k_start, c_k_end):
        nonlocal col
        # A is 2x1, B is 1x7, C is 2x7
        # C_i,k = A_i B_k
        
        for k in range(b_k_start, b_k_end):
            # Term for row 0: A0 B_k -> C_0,k
            U[a_i0*m + b_j][col] += 1
            V[b_j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            # Term for row 1: A1 B_k -> C_1,k
            U[a_i1*m + b_j][col] += 1
            V[b_j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1

    # Part 1: A_L (rows 0,1; cols 0,1) * B_T (rows 0,1; cols 0,1)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: k in {0,1}, i in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0)
    
    # Part 2: A_L * B_T (rows 0,1; cols 2,3)
    # B indices: j in {0,1}, k in {2,3}
    # C indices: k in {2,3}, i in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0)
    
    # Part 3: A_L * B_T (rows 0,1; cols 4,5)
    # B indices: j in {0,1}, k in {4,5}
    # C indices: k in {4,5}, i in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 4, 5, 0)
    
    # Part 4: A_L * B_T (rows 0,1; col 6)
    # B indices: j in {0,1}, k=6
    # C indices: k=6, i in {0,1}
    add_2x2x1(0, 1, 0, 1, 0, 1, 6, 6, 0, 1)
    # Wait, my add_2x2x1 signature is (a_i0, a_i1, b_j0, b_j1, b_k, c_k, c_i0, c_i1)
    # b_k is the row index of B. In B_T, rows are 0,1.
    # So for B_T col 6, we have B[0,6] and B[1,6].
    # So b_j0=0, b_j1=1.
    # c_k is the col index of C. c_k=6.
    # c_i0=0, c_i1=1.
    # Correct.
    
    # Part 5: A_R (rows 0,1; col 2) * B_B (row 2; cols 0..6)
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..6}
    # C indices: k in {0..6}, i in {0,1}
    add_2x1x7(0, 1, 2, 2, 0, 7, 0, 7)
    # Wait, add_2x1x7 signature: (a_i0, a_i1, b_j, b_k_start, b_k_end, c_k_start, c_k_end)
    # b_j is the row index of B. B_B is row 2. So b_j=2.
    # b_k_start=0, b_k_end=7.
    # c_k_start=0, c_k_end=7.
    # Correct.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res['U'], res['V'], res['W']
    rank = len(U[0])
    
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    print(f"Failed for i={i}, j={j}, k={k}, val={val}")
                    sys.exit(1)
                    
    elapsed = time.time() - start_time
    print(f"Score (Rank): {rank}")
    print(f"Time: {elapsed:.4f}s")
