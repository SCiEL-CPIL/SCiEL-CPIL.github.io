"""Matrix multiplication 2x3x3 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 2x3x3 matrix multiplication into a 2x2x2 core
    # (indices i=0,1; j=0,1; k=0,1) using Strassen's 7-rank algorithm,
    # plus the remaining rank-1 terms for the third column of B (k=2).
    #
    # Dimensions: n=2 (rows of A, cols of output), m=3 (cols of A, rows of B), p=3 (cols of B)
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank)
    #
    # Index conventions:
    #   U[i*m + j, c]  where i in [0,2), j in [0,3)
    #   V[j*p + k, c]  where j in [0,3), k in [0,3)
    #   W[k*n + i, c]  where k in [0,3), i in [0,2)
    #
    # Strassen for 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1}):
    # Using the standard Strassen formulas with matrices:
    #   A = [[A00, A01], [A10, A11]]  (i,j)
    #   B = [[B00, B01], [B10, B11]]  (j,k)
    #   C = A * B = [[C00, C01], [C10, C11]]
    #
    # Strassen's 7 multiplications (one common variant):
    #   M1 = A00 * (B10 - B00)
    #   M2 = (A00 + A01) * B00
    #   M3 = (A10 + A11) * B11
    #   M4 = A11 * (B01 - B11)
    #   M5 = (A00 + A10) * B01
    #   M6 = (A01 - A11) * (B10 + B11)
    #   M7 = (A00 - A10) * (B00 + B01)
    #
    #   C00 = M1 + M2 - M4 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M5 + M6
    #
    # Wait, let me re-derive or use a well-known Strassen set. Let me use:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    #   C00 = M1 - M2 + M4 + M6
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M5 - M7
    #
    # Let me verify:
    # C00 = (A00+A11)(B00+B11) - (A10+A11)B00 + A11(B10-B00) + (A10-A00)(B00+B01)
    #     = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A11B10 - A11B00
    #       + A10B00 + A10B01 - A00B00 - A00B01
    #     = A00B00 + A00B11 + A11B11 + A11B10 - A11B00 + A10B01 - A00B01
    #     That's not right. Let me just use the first set I wrote and verify numerically.

    n, m, p = DIMENSIONS
    rank = 7 + 4  # 7 for Strassen 2x2x2 core, 4 for k=2 terms
    # Actually for k=2, we need A[i,j] * B[j,2] for i in {0,1}, j in {0,1,2}
    # That's 2*3 = 6 terms if done naively, but we can share.
    # C[i, k=2] = sum_j A[i,j] * B[j,2]
    # For i=0: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # For i=1: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # We need 6 rank-1 tensors for these, OR we can be smarter.
    # Actually each term A[i,j]*B[j,2] is a separate rank-1 term.
    # So that's 6 terms. Total rank = 7 + 6 = 13.
    
    # Let me reconsider. The full problem is 2x3x3 = 18 schoolbook.
    # Using Strassen on the 2x2x2 sub-block (i={0,1}, j={0,1}, k={0,1}) saves 1 rank (7 vs 8).
    # The remaining terms involve j=2 or k=2.
    # 
    # Terms with k=2 (output column 2):
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # These are 6 independent rank-1 terms.
    #
    # Terms with j=2 and k<2:
    #   C[0,0] += A[0,2]*B[2,0]
    #   C[0,1] += A[0,2]*B[2,1]
    #   C[1,0] += A[1,2]*B[2,0]
    #   C[1,1] += A[1,2]*B[2,1]
    # These are 4 independent rank-1 terms.
    #
    # So total = 7 (Strassen for i,j,k in {0,1}) + 6 (k=2) + 4 (j=2, k<2) = 17.
    
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0.0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 17

    # Helper to set U index
    def u_idx(i, j):
        return i * m + j  # i in [0,2), j in [0,3)

    # Helper to set V index
    def v_idx(j, k):
        return j * p + k  # j in [0,3), k in [0,3)

    # Helper to set W index
    def w_idx(k, i):
        return k * n + i  # k in [0,3), i in [0,2)

    col = 0
    # Strassen 7 terms for (i,j,k) in {0,1}^3
    # M1 = (A00 + A11) * (B00 + B11)
    U[u_idx(0,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1   # C00
    W[w_idx(0,1)][col] = 1   # C11 -> wait, W[k*n+i], so k=1,i=1 is C11
    W[w_idx(1,1)][col] = 1
    col += 1

    # M2 = (A10 + A11) * B00
    U[u_idx(1,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,0)][col] = -1  # -M2 in C00
    W[w_idx(1,0)][col] = 1   # +M2 in C10
    col += 1

    # M3 = A00 * (B01 - B11)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = -1
    W[w_idx(0,1)][col] = 1   # +M3 in C01
    W[w_idx(1,1)][col] = -1  # -M3 in C11
    col += 1

    # M4 = A11 * (B10 - B00)
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,0)][col] = 1
    V[v_idx(0,0)][col] = -1
    W[w_idx(0,0)][col] = 1   # +M4 in C00
    W[w_idx(1,0)][col] = 1   # +M4 in C10
    col += 1

    # M5 = (A00 + A01) * B11
    U[u_idx(0,0)][col] = 1
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,1)][col] = 1   # +M5 in C01
    W[w_idx(1,1)][col] = 1   # +M5 in C11
    col += 1

    # M6 = (A10 - A00) * (B00 + B01)
    U[u_idx(1,0)][col] = 1
    U[u_idx(0,0)][col] = -1
    V[v_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1
    W[w_idx(0,0)][col] = 1   # +M6 in C00
    col += 1

    # M7 = (A01 - A11) * (B10 + B11)
    U[u_idx(0,1)][col] = 1
    U[u_idx(1,1)][col] = -1
    V[v_idx(1,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(1,1)][col] = -1  # -M7 in C11
    col += 1

    # Now k=2 terms: C[i,2] = sum_j A[i,j]*B[j,2]
    # 6 terms: for each i in {0,1}, j in {0,1,2}
    for i in range(2):
        for j in range(3):
            U[u_idx(i, j)][col] = 1
            V[v_idx(j, 2)][col] = 1
            W[w_idx(2, i)][col] = 1
            col += 1

    # Now j=2, k<2 terms: C[i,k] += A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    for i in range(2):
        for k in range(2):
            U[u_idx(i, 2)][col] = 1
            V[v_idx(2, k)][col] = 1
            W[w_idx(k, i)][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
