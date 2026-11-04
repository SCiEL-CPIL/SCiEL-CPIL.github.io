"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Build the tensor T_{i,j,k} = delta_{jk} (matrix multiplication tensor)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Convention: W[k*n + i, c]
    # The tensor is: T[i,j,k] = 1 if j==k else 0
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]

    # Strategy: Use Strassen's algorithm on 2x2 blocks where possible.
    # Decompose the 2x3x8 multiplication.
    # 
    # A is 2x3, B is 3x8. Split A into [A1 | A2] where A1 is 2x1, A2 is 2x2.
    # Split B into [B1; B2] where B1 is 1x8, B2 is 2x8.
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x1 times 1x8 = outer product, rank = 2*8 = 16 terms (but can be done more efficiently)
    # Actually A1*B1: sum over the single element of the middle dimension.
    # A1*B1[i,k] = A1[i,0] * B1[0,k]. This is an outer product of a 2-vector and 8-vector.
    # Rank of outer product 2x8 is 16 in the tensor sense? No.
    # The tensor for outer product 2x1x8 has rank 16 (each element is a separate term).
    # Wait, no. The tensor T_{i,0,k} = A[i,0]*B[0,k]. In the decomposition framework:
    # We need U[i*m+0, c] * V[0*p+k, c] * W[k*n+i, c] to produce A[i,0]*B[0,k] for all i,k.
    # This is just the outer product tensor of a 2-vector and 8-vector, which has rank 2*8 = 16.
    #
    # A2*B2: A2 is 2x2, B2 is 2x8. 
    # Split B2 into four 2x2 blocks: B2 = [B2_11 B2_12 B2_13 B2_14]
    # Each A2*B2_jj is 2x2 times 2x2, which can be done with Strassen rank 7.
    # So total rank for A2*B2 = 4 * 7 = 28.
    # Total rank = 16 + 28 = 44.
    #
    # Alternatively, we can use the known rank-11 for 2x2x2 Strassen and combine differently.
    # Let's try: split A into two 2x1 columns and one 2x1 column.
    # A = [a1, a2, a3] where each is 2x1.
    # B = [b1; b2; b3] where each is 1x8.
    # C = a1*b1 + a2*b2 + a3*b3. Each term is outer product 2x8, rank 16. Total 48. No improvement.
    #
    # Better: Group columns of A. A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    # C = A1*B1 + A2*B2.
    # A1*B1: 2x2 times 2x8. Split B1 into four 2x2 blocks. Each 2x2*2x2 has rank 7 (Strassen).
    # So rank = 4*7 = 28.
    # A2*B2: 2x1 times 1x8. Outer product, rank 16.
    # Total = 28 + 16 = 44.
    #
    # Can we do better? What about using the rank-11 result for 2x2x2?
    # Actually, let's just implement the 44-rank decomposition.
    # Or even better, let's try to get rank 40 or so.
    #
    # Let me implement the straightforward 44-rank decomposition using Strassen for 2x2 blocks.

    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Part 1: A2*B2 where A2 is columns 0,1 of A (2x2), B2 is rows 0,1 of B (2x8)
    # B2 is 2x8, split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # Strassen for 2x2 * 2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,1]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*B[1,0]
    # M4 = A[1,1]*B[1,1]
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    def add_strassen(a_row0, a_row1, b_col_block, base_col):
        """Add Strassen terms for a 2x2 * 2x2 multiplication.
        a_row0, a_row1: indices into A for rows 0 and 1 of the 2x2 block (within the 2x2 sub-block)
        b_col_block: list of 4 column indices in B for the 2x2 block
        base_col: starting column index in the factor matrices
        Returns: new base_col
        """
        nonlocal col
        c = base_col

        # M1 = (A[0,0]+A[1,1])*(B[0,1]+B[1,1])
        U[a_row0[0]][c] += 1  # A[0,0]
        U[a_row0[1]][c] += 1  # A[1,1]
        V[b_col_block[1]][c] += 1  # B[0,1]
        V[b_col_block[3]][c] += 1  # B[1,1]
        W[1 * n + 0][c] += 1  # W[k*n+i] for k=1,i=0 (C[0,1])
        W[1 * n + 1][c] += 1  # W[1*n+1] for k=1,i=1 (C[1,1])
        W[0 * n + 0][c] += 1  # W[0*n+0] for k=0,i=0 (C[0,0])
        W[0 * n + 1][c] -= 1  # W[0*n+1] for k=0,i=1 (C[0,1])... 
        # Wait, this is getting complicated. Let me think about this differently.
        # 
        # The tensor T[i,j,k] = delta_{j,k} means that for each j, we need to compute
        # sum_j A[i,j]*B[j,k] for each i,k.
        # 
        # In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For a fixed j, U[i*m+j, c] depends on i, V[j*p+k, c] depends on k, W[k*n+i, c] depends on both.
        # 
        # Actually, the standard approach is:
        # For each (i,j,k) with j==k, the tensor entry is 1.
        # The rank-1 term for (i,j,k) has U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
        # 
        # Strassen works by grouping terms. Let me just implement the known 44-rank directly.

        return c + 7

    # This approach is getting complex. Let me just use the simple 44-rank decomposition.
    # Actually, let me reconsider. The schoolbook is 48. Strassen on 2x2 blocks gives 44.
    # Let me implement it carefully.

    # Reset
    col = 0
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Split: A = [A_left | A_right] where A_left is 2x2 (cols 0,1), A_right is 2x1 (col 2)
    # B = [B_top; B_bottom] where B_top is 2x8 (rows 0,1), B_bottom is 1x8 (row 2)
    # C = A_left @ B_top + A_right @ B_bottom

    # Part 1: A_left (2x2) @ B_top (2x8)
    # Split B_top into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # For each 2x2 block, use Strassen (rank 7)
    
    for block in range(4):
        # B block columns: block*2, block*2+1
        b0 = block * 2
        b1 = block * 2 + 1
        
        # Strassen 7 terms for 2x2 * 2x2
        # M1 = (A[0,0]+A[1,1]) * (B[0,1]+B[1,1])
        c = col
        U[0 * m + 0][c] = 1  # A[0,0]
        U[1 * m + 1][c] = 1  # A[1,1]
        V[0 * p + b1][c] = 1  # B[0,1] (within block, so actual col b1)
        V[1 * p + b1][c] = 1  # B[1,1]
        # C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] -= M1
        W[b0 * n + 0][c] = 1   # k=b0, i=0
        W[b1 * n + 0][c] = 1   # k=b1, i=0
        W[b0 * n + 1][c] = 1   # k=b0, i=1
        W[b1 * n + 1][c] = -1  # k=b1, i=1
        col += 1

        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        c = col
        U[1 * m + 0][c] = 1  # A[1,0]
        U[1 * m + 1][c] = 1  # A[1,1]
        V[0 * p + b0][c] = 1  # B[0,0]
        # C[1,0] += M2, C[1,1] -= M2
        W[b0 * n + 1][c] = 1
        W[b1 * n + 1][c] = -1
        col += 1

        # M3 = A[0,0] * B[1,0]
        c = col
        U[0 * m + 0][c] = 1  # A[0,0]
        V[1 * p + b0][c] = 1  # B[1,0]
        # C[0,1] += M3, C[1,0] += M3
        W[b0 * n + 0][c] = 1
        W[b0 * n + 1][c] = 1
        col += 1

        # M4 = A[1,1] * B[1,1]
        c = col
        U[1 * m + 1][c] = 1  # A[1,1]
        V[1 * p + b1][c] = 1  # B[1,1]
        # C[0,0] += M4, C[1,1] += M4
        W[b0 * n + 0][c] = 1
        W[b1 * n + 1][c] = 1
        col += 1

        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        c = col
        U[0 * m + 0][c] = 1  # A[0,0]
        U[0 * m + 1][c] = 1  # A[0,1]
        V[1 * p + b1][c] = 1  # B[1,1]
        # C[0,0] -= M5, C[0,1] += M5
        W[b0 * n + 0][c] = -1
        W[b1 * n + 0][c] = 1
        col += 1

        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        c = col
        U[1 * m + 0][c] = 1  # A[1,0]
        U[0 * m + 0][c] = -1  # A[0,0]
        V[0 * p + b0][c] = 1  # B[0,0]
        V[0 * p + b1][c] = 1  # B[0,1]
        # C[1,1] += M6
        W[b1 * n + 1][c] = 1
        col += 1

        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        c = col
        U[0 * m + 1][c] = 1  # A[0,1]
        U[1 * m + 1][c] = -1  # A[1,1]
        V[1 * p + b0][c] = 1  # B[1,0]
        V[1 * p + b1][c] = 1  # B[1,1]
        # C[0,0] += M7
        W[b0 * n + 0][c] = 1
        col += 1

    # Part 2: A_right (2x1) @ B_bottom (1x8)
    # A_right is column 2 of A, B_bottom is row 2 of B
    # C[i,k] += A[i,2] * B[2,k]
    # This is an outer product: for each i in {0,1} and k in {0,...,7}, we need a term.
    # Rank = 2 * 8 = 16
    for i in range(n):
        for k in range(p):
            c = col
            U[i * m + 2][c] = 1  # A[i,2]
            V[2 * p + k][c] = 1  # B[2,k]
            W[k * n + i][c] = 1
            col += 1

    # Verify col == rank
    # col should be 4*7 + 16 = 28 + 16 = 44

    # Verify the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For j != k, T should be 0
    # For j == k, T should be 1
    
    # Let's verify numerically
    U_arr = U
    V_arr = V
    W_arr = W
    r = col
    U_arr = U_arr[:, :r]
    V_arr = V_arr[:, :r]
    W_arr = W_arr[:, :r]
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                actual = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                if abs(expected - actual) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    U = np.zeros((n*m, n*m*p), dtype=float)
                    V = np.zeros((m*p, n*m*p), dtype=float)
                    W = np.zeros((p*n, n*m*p), dtype=float)
                    c2 = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m+j2][c2] = 1
                                V[j2*p+k2][c2] = 1
                                W[k2*n+i2][c2] = 1
                                c2 += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
