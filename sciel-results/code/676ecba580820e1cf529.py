"""Rewritten program: rank-minimal exact tensor decomposition for 2x2 * 2x5."""
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # We use the known optimal decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    # The matrix multiplication tensor T has entries T[i,j,k] = A[i,j] * B[j,k].
    # Dimensions: n=2 (rows of A), m=2 (cols of A / rows of B), p=5 (cols of B).
    #
    # Indexing conventions:
    #   U: shape (n*m, rank) = (4, rank).  U[i*m + j, c]
    #   V: shape (m*p, rank) = (10, rank). V[j*p + k, c]
    #   W: shape (p*n, rank) = (10, rank). W[k*n + i, c]
    #
    # The tensor T satisfies:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # We construct a rank-18 decomposition:
    #   - Split the 5 columns of B into two 2x2 blocks (cols 0,1) and (cols 2,3) plus one remaining col (col 4).
    #   - Apply Strassen's 7-multiplication algorithm to each 2x2 block.
    #   - Use schoolbook (rank 4) for the last column.
    #   Total rank = 7 + 7 + 4 = 18.
    #
    # Strassen's algorithm for C = A * B where A is 2x2, B is 2x2:
    #   M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    #   M2 = (A[1,0]+A[1,1]) * B[0,0]
    #   M3 = A[0,0] * (B[0,1]-B[1,1])
    #   M4 = A[1,1] * (B[1,0]-B[0,0])
    #   M5 = (A[0,0]+A[0,1]) * B[1,1]
    #   M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    #   M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #   C[0,0] = M1 + M4 - M5 + M7
    #   C[0,1] = M3 + M5
    #   C[1,0] = M2 + M4
    #   C[1,1] = M1 - M2 + M3 + M6
    #
    # Each M_c is a product of a linear combination of A entries and a linear combination of B entries.
    # For the tensor decomposition, each multiplication M_c corresponds to one rank-1 term.
    # The "U" factor picks out the linear combo of A entries (indexed by i*m+j),
    # the "V" factor picks out the linear combo of B entries (indexed by j*p+k),
    # and the "W" factor distributes the product to the output entries (indexed by k*n+i).

    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 4 x 18
    V = [[0] * rank for _ in range(m * p)]  # 10 x 18
    W = [[0] * rank for _ in range(p * n)]  # 10 x 18

    # Helper to add a Strassen block for columns offset by col_offset (0-based start column in B)
    def add_strassen(col_offset, start_rank):
        # Strassen's 7 multiplications for a 2x2 block of B starting at column col_offset
        # A is always the full 2x2 matrix (rows i=0,1; cols j=0,1)
        # B block is 2x2: rows j=0,1; cols k=col_offset, col_offset+1
        #
        # M1 = (A[0,0]+A[1,1]) * (B[0,co]+B[1,co+1])
        # M2 = (A[1,0]+A[1,1]) * B[0,co]
        # M3 = A[0,0] * (B[0,co+1]-B[1,co+1])
        # M4 = A[1,1] * (B[1,co]-B[0,co])
        # M5 = (A[0,0]+A[0,1]) * B[1,co+1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,co]+B[0,co+1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,co]+B[1,co+1])
        #
        # Output C[0,0] = M1 + M4 - M5 + M7  -> output entry (i=0, k=co)
        # Output C[0,1] = M3 + M5            -> output entry (i=0, k=co+1)
        # Output C[1,0] = M2 + M4            -> output entry (i=1, k=co)
        # Output C[1,1] = M1 - M2 + M3 + M6  -> output entry (i=1, k=co+1)

        co = col_offset
        r = start_rank

        # M1: A combo = A[0,0] + A[1,1]; B combo = B[0,co] + B[1,co+1]
        # U indices: 0*m+0=0 (A[0,0]), 1*m+1=3 (A[1,1])
        # V indices: 0*p+co=co, 1*p+(co+1)=5+co+1=co+6
        # W: contributes to C[0,co]=idx co*n+0=co, C[1,co+1]=idx (co+1)*n+1=2co+3
        U[0][r] = 1
        U[3][r] = 1
        V[co][r] = 1
        V[co + 6][r] = 1
        W[co][r] = 1        # C[0,co]
        W[2 * co + 3][r] = 1  # C[1,co+1]
        r += 1

        # M2: A combo = A[1,0] + A[1,1]; B combo = B[0,co]
        # U indices: 1*m+0=2 (A[1,0]), 1*m+1=3 (A[1,1])
        # V indices: 0*p+co=co
        # W: contributes to C[1,co]=idx co*n+1=co+1
        U[2][r] = 1
        U[3][r] = 1
        V[co][r] = 1
        W[co + 1][r] = 1
        r += 1

        # M3: A combo = A[0,0]; B combo = B[0,co+1] - B[1,co+1]
        # U indices: 0
        # V indices: 0*p+(co+1)=co+1, 1*p+(co+1)=co+6 -> +1 and -1
        # W: contributes to C[0,co+1]=idx (co+1)*n+0=2co+2, C[1,co+1]=idx (co+1)*n+1=2co+3
        U[0][r] = 1
        V[co + 1][r] = 1
        V[co + 6][r] = -1
        W[2 * co + 2][r] = 1   # C[0,co+1]
        W[2 * co + 3][r] = 1   # C[1,co+1]
        r += 1

        # M4: A combo = A[1,1]; B combo = B[1,co] - B[0,co]
        # U indices: 3
        # V indices: 1*p+co=5+co, 0*p+co=co -> +1 and -1
        # W: contributes to C[0,co]=idx co, C[1,co]=idx co+1
        U[3][r] = 1
        V[co + 5][r] = 1
        V[co][r] = -1
        W[co][r] = 1       # C[0,co]
        W[co + 1][r] = 1   # C[1,co]
        r += 1

        # M5: A combo = A[0,0] + A[0,1]; B combo = B[1,co+1]
        # U indices: 0, 1
        # V indices: 1*p+(co+1)=co+6
        # W: contributes to C[0,co]=idx co (with -1), C[0,co+1]=idx 2co+2
        U[0][r] = 1
        U[1][r] = 1
        V[co + 6][r] = 1
        W[co][r] = -1       # C[0,co] (negative in C[0,0] formula)
        W[2 * co + 2][r] = 1  # C[0,co+1]
        r += 1

        # M6: A combo = A[1,0] - A[0,0]; B combo = B[0,co] + B[0,co+1]
        # U indices: 2 (+1), 0 (-1)
        # V indices: co, co+1
        # W: contributes to C[1,co+1]=idx 2co+3
        U[2][r] = 1
        U[0][r] = -1
        V[co][r] = 1
        V[co + 1][r] = 1
        W[2 * co + 3][r] = 1  # C[1,co+1]
        r += 1

        # M7: A combo = A[0,1] - A[1,1]; B combo = B[1,co] + B[1,co+1]
        # U indices: 1 (+1), 3 (-1)
        # V indices: co+5, co+6
        # W: contributes to C[0,co]=idx co
        U[1][r] = 1
        U[3][r] = -1
        V[co + 5][r] = 1
        V[co + 6][r] = 1
        W[co][r] = 1       # C[0,co]
        r += 1

        return r

    # Add Strassen for columns 0,1 (col_offset=0)
    r = add_strassen(0, 0)
    # Add Strassen for columns 2,3 (col_offset=2)
    r = add_strassen(2, r)

    # Schoolbook for column 4 (col_offset=4, only 1 column)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # 4 terms: (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    for i in range(n):
        for j in range(m):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1  # k=4, i -> index 4*2+i = 8+i
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
