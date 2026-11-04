"""Matrix multiplication tensor decomposition 4x4 x 4x9."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p  # 4 * 4 * 9 = 144
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def strassen_4x4x4():
        """
        Strassen's algorithm for 4x4 * 4x4 = 4x4.
        Returns (U, V, W) of shape (16, 7), (16, 7), (16, 7)
        where U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
        
        Strassen splits each 4x4 into 2x2 blocks:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # For 4x4 matrices, we split into 2x2 blocks of 2x2 submatrices
        # A has entries A[i,j] for i,j in 0..3
        # B has entries B[j,k] for j,k in 0..3
        
        # Block structure:
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
        # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
        
        # Each 2x2 block multiplication uses 4 schoolbook terms
        # But Strassen uses 7 multiplications of 2x2 blocks
        
        # For each Strassen term, we need to expand the 2x2 block multiplication
        # using schoolbook (4 terms each)
        
        U = np.zeros((16, 7))  # 4*4 entries, 7 Strassen terms
        V = np.zeros((16, 7))  # 4*4 entries, 7 Strassen terms
        W = np.zeros((16, 7))  # 4*4 entries, 7 Strassen terms
        
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Let's define the contributions more carefully.
        # For a 2x2 block multiplication X*Y where X and Y are 2x2:
        # (X*Y)[i,j] = sum_k X[i,k] * Y[k,j]
        
        # We'll build the decomposition by tracking which A and B entries
        # contribute to each Strassen term.
        
        # Helper: add contribution to U, V, W for a bilinear term
        # U[i*4+j, col] *= coeff_u
        # V[j*4+k, col] *= coeff_v  
        # W[k*4+i, col] += coeff_w * (product of coeff_u and coeff_v)
        
        def add_term(col, u_entries, v_entries, w_coeff=1.0):
            """
            u_entries: list of (i, j, coeff) for A[i,j]
            v_entries: list of (j, k, coeff) for B[j,k]
            """
            # This adds a rank-1 term where:
            # U picks from A entries, V picks from B entries
            # The product gives the contribution to C
            
            # For simplicity, let's handle single-entry terms
            pass
        
        # Actually, let me think about this differently.
        # The matrix multiplication tensor for 4x4*4x4 has rank 7 (Strassen).
        # I need to construct explicit U, V, W matrices.
        
        # Let me use a different approach: directly construct the Strassen
        # decomposition by expanding each 2x2 block multiplication.
        
        # For 2x2 * 2x2 = 2x2, schoolbook has rank 4.
        # Strassen for 2x2*2x2=2x2 has rank 7.
        
        # Actually, Strassen's original algorithm is for 2x2*2x2=2x2 with rank 7.
        # For 4x4*4x4=4x4, we can apply Strassen recursively or use it once.
        
        # Let me just use the standard Strassen construction for 4x4.
        
        # The key insight: 4x4*4x4 can be viewed as 2x2 block matrix multiplication
        # where each block is 2x2. Strassen reduces 8 block multiplications to 7.
        # Each block multiplication (2x2*2x2=2x2) has rank 4.
        # So total rank = 7 * 4 = 28? No, that's not right either.
        
        # Actually, the rank of the 4x4*4x4=4x4 tensor is at most 7*4=28
        # using Strassen at the block level, but we can do better by
        # applying Strassen at the element level.
        
        # Let me just use a known result: the rank of the 4x4*4x4=4x4
        # matrix multiplication tensor is 49 (this is the best known upper bound
        # from repeated Strassen, but actually it's lower).
        
        # Wait, I'm overcomplicating this. Let me just use the block approach:
        # Split 4x4 into four 2x2 blocks.
        # Use Strassen's 7 multiplications of 2x2 blocks.
        # Each 2x2 block multiplication uses 4 scalar multiplications.
        # Total: 7 * 4 = 28 rank-1 terms.
        
        # But we can do better by using Strassen for the 2x2 block multiplications too.
        # Strassen for 2x2*2x2=2x2 has rank 7.
        # So total rank = 7 * 7 = 49.
        
        # This matches the known result that the 4x4*4x4=4x4 tensor has rank ≤ 49.
        
        # Let me construct this explicitly.
        
        # Level 1: 4x4 blocks
        # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
        # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
        # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
        
        # Strassen terms (each is a product of 2x2 matrices):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Now, each Mj is a 2x2 matrix product.
        # For each Mj, we need to decompose the 2x2*2x2=2x2 tensor.
        # Using Strassen for each: rank 7 per Mj.
        # Total: 7 * 7 = 49.
        
        # Let me define Strassen for 2x2*2x2=2x2:
        # For X*Y where X,Y are 2x2:
        # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
        # N2 = (X[1,0]+X[1,1])Y[0,0]
        # N3 = X[0,0](Y[0,1]-Y[1,1])
        # N4 = X[1,1](Y[1,0]-Y[0,0])
        # N5 = (X[0,0]+X[0,1])Y[1,1]
        # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
        # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
        
        # (X*Y)[0,0] = N1 + N4 - N5 + N7
        # (X*Y)[0,1] = N3 + N5
        # (X*Y)[1,0] = N2 + N4
        # (X*Y)[1,1] = N1 - N2 + N3 + N6
        
        # Now I need to compose these.
        # This is getting complex. Let me use a programmatic approach.
        
        # For each of the 7 level-1 Strassen terms Mj, I have:
        # Mj = X_j * Y_j where X_j and Y_j are linear combinations of 2x2 blocks
        
        # For each Mj, I apply level-2 Strassen with 7 terms.
        # Each level-2 term is a scalar product of entries from X_j and Y_j.
        
        # Let me define the linear combinations explicitly.
        
        # Level 1:
        # M1: X1 = A11+A22, Y1 = B11+B22
        # M2: X2 = A21+A22, Y2 = B11
        # M3: X3 = A11, Y3 = B12-B22
        # M4: X4 = A22, Y4 = B21-B11
        # M5: X5 = A11+A12, Y5 = B22
        # M6: X6 = A21-A11, Y6 = B11+B12
        # M7: X7 = A12-A22, Y7 = B21+B22
        
        # Each Xj and Yj is a 2x2 matrix that's a linear combination of A or B blocks.
        
        # Let me represent each Xj as a 2x2 matrix of coefficients on A entries.
        # A entries: A[0,0], A[0,1], A[1,0], A[1,1] (for block A11)
        #            A[0,2], A[0,3], A[1,2], A[1,3] (for block A12)
        #            A[2,0], A[2,1], A[3,0], A[3,1] (for block A21)
        #            A[2,2], A[2,3], A[3,2], A[3,3] (for block A22)
        
        # This is getting very complex. Let me use a simpler approach:
        # Just use schoolbook for the 4x4*4x4 part, which gives rank 16.
        # Then for the full 4x4*4x9, I can split into 4x4*4x4 + 4x4*4x5.
        # But that doesn't help much.
        
        # Actually, let me just return schoolbook for now and focus on
        # a cleaner structural approach.
        
        # For now, let me return a placeholder and implement the real
        # Strassen construction separately.
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # I'll implement this properly in the main function
        return U, V, W

    def build_strassen_decomposition():
        """
        Build the full 4x4*4x9 decomposition using Strassen for the
        first 4 columns (4x4*4x4) and schoolbook for the remaining 5 columns.
        
        Rank: 49 (Strassen) + 4*5 (schoolbook) = 49 + 20 = 69
        """
        # Strassen part: 4x4 * 4x4 = 4x4, rank 49
        # Schoolbook part: 4x4 * 4x5, rank 4*4*5 = 80... wait that's worse.
        
        # Actually, for the last 5 columns, schoolbook is 4*4*5 = 80.
        # Total: 49 + 80 = 129. That's worse than 144.
        
        # Let me reconsider. Maybe I should use Strassen for more columns.
        
        # Alternative: split 4x4*4x9 as 4x4*(4x4 + 4x5)
        # = (4x4*4x4) + (4x4*4x5)
        # Rank of 4x4*4x4 ≤ 49 (Strassen)
        # Rank of 4x4*4x5: use schoolbook = 4*4*5 = 80
        # Total: 49 + 80 = 129
        
        # Or split as (4x4*4x2) + (4x4*4x2) + (4x4*4x1) + ...
        # This is getting complicated.
        
        # Let me just use the insight from the cell findings:
        # "Strassen-based block decomposition reduces rank from 144 to 114
        # by using rank-49 Strassen for two 4x4*4x4 subproblems and rank-16
        # schoolbook for the 4x4*4x1 subproblem"
        
        # So: 4x4*4x9 = 4x4*(4x4 + 4x4 + 4x1)
        # = (4x4*4x4) + (4x4*4x4) + (4x4*4x1)
        # Rank: 49 + 49 + 16 = 114
        
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3), rank 49
        U1, V1, W1 = _strassen_4x4x4_impl()
        U_parts.append(U1)
        V_parts.append(V1)
        W_parts.append(W1)
        
        # Part 2: 4x4 * 4x4 (columns 4-7), rank 49
        U2, V2, W2 = _strassen_4x4x4_impl()
        # Need to shift the column indices in V
        # V2[j*4+k, c] should become V[j*4+(k+4), c]
        V2_shifted = np.zeros((16, 49))
        for j in range(4):
            for k in range(4):
                V2_shifted[j*4 + (k+4), :] = V2[j*4 + k, :]
        U_parts.append(U2)
        V_parts.append(V2_shifted)
        W_parts.append(W2)
        
        # Part 3: 4x4 * 4x1 (column 8), rank 16 (schoolbook)
        U3 = np.zeros((16, 16))
        V3 = np.zeros((16, 16))
        W3 = np.zeros((16, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                U3[i*4+j, col] = 1
                V3[j*4+8, col] = 1  # column 8
                W3[8*4+i, col] = 1
                col += 1
        U_parts.append(U3)
        V_parts.append(V3)
        W_parts.append(W3)
        
        U_full = np.hstack(U_parts)
        V_full = np.hstack(V_parts)
        W_full = np.hstack(W_parts)
        
        rank = U_full.shape[1]
        return U_full, V_full, W_full, rank

    def _strassen_4x4x4_impl():
        """
        Implement Strassen's algorithm for 4x4*4x4=4x4 with rank 49.
        
        Uses two levels of Strassen:
        Level 1: 7 terms, each is a product of 2x2 blocks
        Level 2: For each level-1 term, apply Strassen to the 2x2 block product (7 terms each)
        Total: 7*7 = 49 rank-1 terms
        """
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Level 1 Strassen terms for 4x4 block multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mj, we have Xj * Yj where Xj and Yj are 2x2 matrices
        # that are linear combinations of A or B blocks.
        
        # Define the linear combinations for each Xj and Yj.
        # Each is a 2x2 matrix of coefficients.
        
        # A blocks:
        # A11 = A[0:2, 0:2], entries: A[0,0], A[0,1], A[1,0], A[1,1]
        # A12 = A[0:2, 2:4], entries: A[0,2], A[0,3], A[1,2], A[1,3]
        # A21 = A[2:4, 0:2], entries: A[2,0], A[2,1], A[3,0], A[3,1]
        # A22 = A[2:4, 2:4], entries: A[2,2], A[2,3], A[3,2], A[3,3]
        
        # B blocks similar.
        
        # For each level-1 term Mj = Xj * Yj:
        # Xj is a 2x2 matrix where each entry is a linear combination of A entries
        # Yj is a 2x2 matrix where each entry is a linear combination of B entries
        
        # Let me define Xj as a 4x4 matrix of coefficients (flattened 2x2 blocks)
        # Actually, let me be more careful.
        
        # For M1: X1 = A11 + A22
        # X1[0,0] = A[0,0] + A[2,2]
        # X1[0,1] = A[0,1] + A[2,3]
        # X1[1,0] = A[1,0] + A[3,2]
        # X1[1,1] = A[1,1] + A[3,3]
        
        # Y1 = B11 + B22
        # Y1[0,0] = B[0,0] + B[2,2]
        # Y1[0,1] = B[0,1] + B[2,3]
        # Y1[1,0] = B[1,0] + B[3,2]
        # Y1[1,1] = B[1,1] + B[3,3]
        
        # Now, M1 = X1 * Y1 is a 2x2 matrix.
        # M1[0,0] = X1[0,0]*Y1[0,0] + X1[0,1]*Y1[1,0]
        # M1[0,1] = X1[0,0]*Y1[0,1] + X1[0,1]*Y1[1,1]
        # M1[1,0] = X1[1,0]*Y1[0,0] + X1[1,1]*Y1[1,0]
        # M1[1,1] = X1[1,0]*Y1[0,1] + X1[1,1]*Y1[1,1]
        
        # Each of these is a product of two linear combinations.
        # To decompose this into rank-1 terms, I can use Strassen on the 2x2 product.
        
        # For a 2x2 product X*Y, Strassen gives:
        # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
        # N2 = (X[1,0]+X[1,1])Y[0,0]
        # N3 = X[0,0](Y[0,1]-Y[1,1])
        # N4 = X[1,1](Y[1,0]-Y[0,0])
        # N5 = (X[0,0]+X[0,1])Y[1,1]
        # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
        # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
        
        # (X*Y)[0,0] = N1 + N4 - N5 + N7
        # (X*Y)[0,1] = N3 + N5
        # (X*Y)[1,0] = N2 + N4
        # (X*Y)[1,1] = N1 - N2 + N3 + N6
        
        # Each Nj is a product of two scalar linear combinations.
        # So each Nj is a rank-1 term in the tensor decomposition.
        
        # For each level-1 term Mj, I have 7 level-2 terms N1..N7.
        # Each Nj contributes to specific entries of C.
        
        # Let me define the coefficients for each level-1 term.
        
        # For M1:
        # X1 entries (coefficients on A):
        # X1[0,0]: A[0,0] + A[2,2]
        # X1[0,1]: A[0,1] + A[2,3]
        # X1[1,0]: A[1,0] + A[3,2]
        # X1[1,1]: A[1,1] + A[3,3]
        #
        # Y1 entries (coefficients on B):
        # Y1[0,0]: B[0,0] + B[2,2]
        # Y1[0,1]: B[0,1] + B[2,3]
        # Y1[1,0]: B[1,0] + B[3,2]
        # Y1[1,1]: B[1,1] + B[3,3]
        
        # This is getting very tedious to do by hand. Let me use a programmatic approach.
        
        # Define the linear combinations as dictionaries mapping (block, row, col) to coefficient
        # where block is one of A11, A12, A21, A22 or B11, B12, B21, B22
        
        # Actually, let me just implement this computationally.
        
        # Define helper to create a 2x2 matrix of linear combinations
        # Each entry is a dict: {(matrix, i, j): coeff}
        
        def make_lc(entries):
            """Create a linear combination as a dict."""
            d = {}
            for (mat, i, j), coeff in entries:
                d[(mat, i, j)] = d.get((mat, i, j), 0) + coeff
            return d
        
        # Define the 7 level-1 terms.
        # Each term has X (linear combo of A blocks) and Y (linear combo of B blocks).
        # X and Y are 2x2 matrices of linear combinations.
        
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        # For each block, define the mapping to global indices.
        # A11: (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
        # A12: (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
        # A21: (0,0)->(2,0), (0,1)->(2,1), (1,0)->(3,0), (1,1)->(3,1)
        # A22: (0,0)->(2,2), (0,1)->(2,3), (1,0)->(3,2), (1,1)->(3,3)
        
        # B similar.
        
        # Define block offsets:
        # A11: row_offset=0, col_offset=0
        # A12: row_offset=0, col_offset=2
        # A21: row_offset=2, col_offset=0
        # A22: row_offset=2, col_offset=2
        
        # B11: row_offset=0, col_offset=0
        # B12: row_offset=0, col_offset=2
        # B21: row_offset=2, col_offset=0
        # B22: row_offset=2, col_offset=2
        
        # For each level-1 term, define X and Y as 2x2 matrices of linear combinations.
        # Each entry is a list of (block, row_in_block, col_in_block, coeff)
        
        # M1: X1 = A11 + A22, Y1 = B11 + B22
        # X1[0,0] = A11[0,0] + A22[0,0]
        # X1[0,1] = A11[0,1] + A22[0,1]
        # X1[1,0] = A11[1,0] + A22[1,0]
        # X1[1,1] = A11[1,1] + A22[1,1]
        #
        # Y1[0,0] = B11[0,0] + B22[0,0]
        # Y1[0,1] = B11[0,1] + B22[0,1]
        # Y1[1,0] = B11[1,0] + B22[1,0]
        # Y1[1,1] = B11[1,1] + B22[1,1]
        
        # I'll define all 7 terms programmatically.
        
        # Let me create a list of the 7 level-1 terms.
        # Each term is (X, Y) where X and Y are 2x2 arrays of linear combinations.
        
        # Define linear combinations as lists of (block_id, r, c, coeff)
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22 for X
        #           0=B11, 1=B12, 2=B21, 3=B22 for Y
        
        # M1: X = A11+A22, Y = B11+B22
        # X[r,c] = A11[r,c] + A22[r,c] for r,c in 0,1
        # Y[r,c] = B11[r,c] + B22[r,c] for r,c in 0,1
        
        level1_terms = []
        
        # M1: X = A11+A22, Y = B11+B22
        X1 = [[[(0,r,c,1), (3,r,c,1)] for c in range(2)] for r in range(2)]
        Y1 = [[[(0,r,c,1), (3,r,c,1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X1, Y1))
        
        # M2: X = A21+A22, Y = B11
        X2 = [[[(2,r,c,1), (3,r,c,1)] for c in range(2)] for r in range(2)]
        Y2 = [[[(0,r,c,1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X2, Y2))
        
        # M3: X = A11, Y = B12-B22
        X3 = [[[(0,r,c,1)] for c in range(2)] for r in range(2)]
        Y3 = [[[(1,r,c,1), (3,r,c,-1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X3, Y3))
        
        # M4: X = A22, Y = B21-B11
        X4 = [[[(3,r,c,1)] for c in range(2)] for r in range(2)]
        Y4 = [[[(2,r,c,1), (0,r,c,-1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X4, Y4))
        
        # M5: X = A11+A12, Y = B22
        X5 = [[[(0,r,c,1), (1,r,c,1)] for c in range(2)] for r in range(2)]
        Y5 = [[[(3,r,c,1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X5, Y5))
        
        # M6: X = A21-A11, Y = B11+B12
        X6 = [[[(2,r,c,1), (0,r,c,-1)] for c in range(2)] for r in range(2)]
        Y6 = [[[(0,r,c,1), (1,r,c,1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X6, Y6))
        
        # M7: X = A12-A22, Y = B21+B22
        X7 = [[[(1,r,c,1), (3,r,c,-1)] for c in range(2)] for r in range(2)]
        Y7 = [[[(2,r,c,1), (3,r,c,1)] for c in range(2)] for r in range(2)]
        level1_terms.append((X7, Y7))
        
        # Now, for each level-1 term Mj = Xj * Yj, apply Strassen to the 2x2 product.
        # Strassen for 2x2*2x2=2x2:
        # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
        # N2 = (X[1,0]+X[1,1])Y[0,0]
        # N3 = X[0,0](Y[0,1]-Y[1,1])
        # N4 = X[1,1](Y[1,0]-Y[0,0])
        # N5 = (X[0,0]+X[0,1])Y[1,1]
        # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
        # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
        
        # (X*Y)[0,0] = N1 + N4 - N5 + N7
        # (X*Y)[0,1] = N3 + N5
        # (X*Y)[1,0] = N2 + N4
        # (X*Y)[1,1] = N1 - N2 + N3 + N6
        
        # Each Nj is a product of two linear combinations (scalars).
        # So Nj = (sum of A entries) * (sum of B entries)
        # This is a rank-1 term in the tensor.
        
        # For each level-1 term Mj, I have 7 level-2 terms N1..N7.
        # Each Nj contributes to specific entries of C with specific coefficients.
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mj, the contribution to C is:
        # If Mj appears in C11 with coeff c11, then:
        #   Mj[0,0] contributes to C[0,0] with coeff c11
        #   Mj[0,1] contributes to C[0,1] with coeff c11
        #   Mj[1,0] contributes to C[1,0] with coeff c11
        #   Mj[1,1] contributes to C[1,1] with coeff c11
        #
        # If Mj appears in C12 with coeff c12, then:
        #   Mj[0,0] contributes to C[0,2] with coeff c12
        #   Mj[0,1] contributes to C[0,3] with coeff c12
        #   Mj[1,0] contributes to C[1,2] with coeff c12
        #   Mj[1,1] contributes to C[1,3] with coeff c12
        #
        # If Mj appears in C21 with coeff c21, then:
        #   Mj[0,0] contributes to C[2,0] with coeff c21
        #   Mj[0,1] contributes to C[2,1] with coeff c21
        #   Mj[1,0] contributes to C[3,0] with coeff c21
        #   Mj[1,1] contributes to C[3,1] with coeff c21
        #
        # If Mj appears in C22 with coeff c22, then:
        #   Mj[0,0] contributes to C[2,2] with coeff c22
        #   Mj[0,1] contributes to C[2,3] with coeff c22
        #   Mj[1,0] contributes to C[3,2] with coeff c22
        #   Mj[1,1] contributes to C[3,3] with coeff c22
        
        # Define the coefficients for each Mj in each C block.
        # C11: M1(+1), M4(+1), M5(-1), M7(+1)
        # C12: M3(+1), M5(+1)
        # C21: M2(+1), M4(+1)
        # C22: M1(+1), M2(-1), M3(+1), M6(+1)
        
        c_coeffs = {
            0: {0: 1, 1: 0, 2: 0, 3: 1},  # M1: C11(+1), C12(0), C21(0), C22(+1)
            1: {0: 0, 1: 0, 2: 1, 3: -1},  # M2: C11(0), C12(0), C21(+1), C22(-1)
            2: {0: 0, 1: 1, 2: 0, 3: 1},   # M3: C11(0), C12(+1), C21(0), C22(+1)
            3: {0: 1, 1: 0, 2: 1, 3: 0},   # M4: C11(+1), C12(0), C21(+1), C22(0)
            4: {0: -1, 1: 1, 2: 0, 3: 0},  # M5: C11(-1), C12(+1), C21(0), C22(0)
            5: {0: 0, 1: 0, 2: 0, 3: 1},   # M6: C11(0), C12(0), C21(0), C22(+1)
            6: {0: 1, 1: 0, 2: 0, 3: 0},   # M7: C11(+1), C12(0), C21(0), C22(0)
        }
        
        # Now, for each level-1 term Mj and each level-2 term Nk,
        # I need to determine the contribution to each C entry.
        
        # Strassen for 2x2 product X*Y:
        # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
        # N2 = (X[1,0]+X[1,1])Y[0,0]
        # N3 = X[0,0](Y[0,1]-Y[1,1])
        # N4 = X[1,1](Y[1,0]-Y[0,0])
        # N5 = (X[0,0]+X[0,1])Y[1,1]
        # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
        # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
        
        # (X*Y)[0,0] = N1 + N4 - N5 + N7
        # (X*Y)[0,1] = N3 + N5
        # (X*Y)[1,0] = N2 + N4
        # (X*Y)[1,1] = N1 - N2 + N3 + N6
        
        # So for each Nk, the contribution to (X*Y)[r,c] is:
        n_to_block = {
            0: {0: 1, 1: 0, 2: 0, 3: 1},  # N1: [0,0](+1), [0,1](0), [1,0](0), [1,1](+1)
            1: {0: 0, 1: 0, 2: 1, 3: -1},  # N2: [0,0](0), [0,1](0), [1,0](+1), [1,1](-1)
            2: {0: 0, 1: 1, 2: 0, 3: 1},   # N3: [0,0](0), [0,1](+1), [1,0](0), [1,1](+1)
            3: {0: 1, 1: 0, 2: 1, 3: 0},   # N4: [0,0](+1), [0,1](0), [1,0](+1), [1,1](0)
            4: {0: -1, 1: 1, 2: 0, 3: 0},  # N5: [0,0](-1), [0,1](+1), [1,0](0), [1,1](0)
            5: {0: 0, 1: 0, 2: 0, 3: 1},   # N6: [0,0](0), [0,1](0), [1,0](0), [1,1](+1)
            6: {0: 1, 1: 0, 2: 0, 3: 0},   # N7: [0,0](+1), [0,1](0), [1,0](0), [1,1](0)
        }
        
        # Now I need to construct the actual U, V, W matrices.
        # Each rank-1 term corresponds to a product of a linear combination of A entries
        # and a linear combination of B entries, contributing to specific C entries.
        
        # For a rank-1 term:
        # U[i*4+j, col] = sum of coefficients for A[i,j]
        # V[j*4+k, col] = sum of coefficients for B[j,k]
        # W[k*4+i, col] = sum of contributions to C[i,k]
        
        # Actually, the tensor decomposition is:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where T[i,j,k] = 1 if i==k else 0 (for the identity in the j dimension)
        
        # Wait, I need to be more careful. The matrix multiplication tensor is:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] has T[i,j,k] = 1 for all i,j,k (it's the identity tensor).
        # Actually no, T[i,j,k] = delta_{i,k} is not right either.
        
        # The matrix multiplication tensor T is defined by:
        # T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
        # So T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
        
        # The rank-1 decomposition is:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # For each rank-1 term c, we have:
        # U[i,j,c] is the coefficient of A[i,j] in this term
        # V[j,k,c] is the coefficient of B[j,k] in this term
        # W[k,i,c] is the coefficient of C[i,k] in this term
        
        # Now, for each level-2 term Nk within level-1 term Mj:
        # Nk = (linear combo of A entries) * (linear combo of B entries)
        # This contributes to specific C entries with specific coefficients.
        
        # Let me construct this computationally.
        
        col = 0
        for j_idx, (X, Y) in enumerate(level1_terms):
            # For this level-1 term, compute the 7 level-2 Strassen terms.
            # Each Nk is a product of two linear combinations.
            
            # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
            # N2 = (X[1,0]+X[1,1])Y[0,0]
            # N3 = X[0,0](Y[0,1]-Y[1,1])
            # N4 = X[1,1](Y[1,0]-Y[0,0])
            # N5 = (X[0,0]+X[0,1])Y[1,1]
            # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
            # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
            
            # Each of these is a product of two linear combinations.
            # The first linear combination is a sum of A entries.
            # The second linear combination is a sum of B entries.
            
            # For Nk, let:
            # A_lc = sum of (coeff * A[i,j]) for various (i,j)
            # B_lc = sum of (coeff * B[j,k]) for various (j,k)
            # Then Nk = A_lc * B_lc
            
            # This contributes to C entries according to n_to_block[k] and c_coeffs[j_idx].
            
            # Let me define the 7 level-2 terms for this level-1 term.
            # Each is (A_entries, B_entries) where A_entries is a list of (i,j,coeff)
            # and B_entries is a list of (j,k,coeff).
            
            # Helper to combine linear combinations
            def combine_lcs(lcs):
                """Combine a list of linear combinations into one."""
                result = {}
                for lc in lcs:
                    for key, coeff in lc.items():
                        result[key] = result.get(key, 0) + coeff
                return result
            
            # X[r][c] is a list of (block, row_in_block, col_in_block, coeff)
            # I need to convert this to a dict of (global_i, global_j) -> coeff
            
            def block_to_global(block, r, c, is_A=True):
                """Convert block coordinates to global coordinates."""
                if is_A:
                    # A blocks
                    if block == 0:  # A11
                        return (r, c)
                    elif block == 1:  # A12
                        return (r, c + 2)
                    elif block == 2:  # A21
                        return (r + 2, c)
                    elif block == 3:  # A22
                        return (r + 2, c + 2)
                else:
                    # B blocks
                    if block == 0:  # B11
                        return (r, c)
                    elif block == 1:  # B12
                        return (r, c + 2)
                    elif block == 2:  # B21
                        return (r + 2, c)
                    elif block == 3:  # B22
                        return (r + 2, c + 2)
            
            def convert_X_to_dict(X):
                """Convert X (2x2 array of lists) to a dict of (i,j) -> coeff."""
                result = {}
                for r in range(2):
                    for c in range(2):
                        for (block, br, bc, coeff) in X[r][c]:
                            gi, gj = block_to_global(block, br, bc, is_A=True)
                            result[(gi, gj)] = result.get((gi, gj), 0) + coeff
                return result
            
            def convert_Y_to_dict(Y):
                """Convert Y (2x2 array of lists) to a dict of (j,k) -> coeff."""
                result = {}
                for r in range(2):
                    for c in range(2):
                        for (block, br, bc, coeff) in Y[r][c]:
                            gi, gj = block_to_global(block, br, bc, is_A=False)
                            result[(gi, gj)] = result.get((gi, gj), 0) + coeff
                return result
            
            # Now compute the 7 level-2 terms.
            # X[r][c] is a list, but I need to treat each X[r][c] as a linear combination.
            # Actually, X[r][c] is already a list of terms. Let me convert to dict.
            
            X_dict = convert_X_to_dict(X)
            Y_dict = convert_Y_to_dict(Y)
            
            # But wait, X and Y are 2x2 matrices of linear combinations.
            # X[r][c] is the linear combination for the (r,c) entry of the 2x2 matrix X.
            # So X_dict should be a 2x2 array of dicts, not a single dict.
            
            # Let me redo this.
            
            def convert_X_to_2x2_dict(X):
                """Convert X to a 2x2 array of dicts."""
                result = [[{} for _ in range(2)] for _ in range(2)]
                for r in range(2):
                    for c in range(2):
                        for (block, br, bc, coeff) in X[r][c]:
                            gi, gj = block_to_global(block, br, bc, is_A=True)
                            result[r][c][(gi, gj)] = result[r][c].get((gi, gj), 0) + coeff
                return result
            
            def convert_Y_to_2x2_dict(Y):
                """Convert Y to a 2x2 array of dicts."""
                result = [[{} for _ in range(2)] for _ in range(2)]
                for r in range(2):
                    for c in range(2):
                        for (block, br, bc, coeff) in Y[r][c]:
                            gi, gj = block_to_global(block, br, bc, is_A=False)
                            result[r][c][(gi, gj)] = result[r][c].get((gi, gj), 0) + coeff
                return result
            
            X_2x2 = convert_X_to_2x2_dict(X)
            Y_2x2 = convert_Y_to_2x2_dict(Y)
            
            # Now compute the 7 level-2 Strassen terms.
            # Each is a product of two linear combinations (dicts).
            
            level2_terms = []
            
            # N1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1])
            A_N1 = combine_lcs([X_2x2[0][0], X_2x2[1][1]])
            B_N1 = combine_lcs([Y_2x2[0][0], Y_2x2[1][1]])
            level2_terms.append((A_N1, B_N1))
            
            # N2 = (X[1,0]+X[1,1])Y[0,0]
            A_N2 = combine_lcs([X_2x2[1][0], X_2x2[1][1]])
            B_N2 = Y_2x2[0][0]
            level2_terms.append((A_N2, B_N2))
            
            # N3 = X[0,0](Y[0,1]-Y[1,1])
            A_N3 = X_2x2[0][0]
            B_N3 = combine_lcs([Y_2x2[0][1], {k: -v for k, v in Y_2x2[1][1].items()}])
            level2_terms.append((A_N3, B_N3))
            
            # N4 = X[1,1](Y[1,0]-Y[0,0])
            A_N4 = X_2x2[1][1]
            B_N4 = combine_lcs([Y_2x2[1][0], {k: -v for k, v in Y_2x2[0][0].items()}])
            level2_terms.append((A_N4, B_N4))
            
            # N5 = (X[0,0]+X[0,1])Y[1,1]
            A_N5 = combine_lcs([X_2x2[0][0], X_2x2[0][1]])
            B_N5 = Y_2x2[1][1]
            level2_terms.append((A_N5, B_N5))
            
            # N6 = (X[1,0]-X[0,0])(Y[0,0]+Y[0,1])
            A_N6 = combine_lcs([X_2x2[1][0], {k: -v for k, v in X_2x2[0][0].items()}])
            B_N6 = combine_lcs([Y_2x2[0][0], Y_2x2[0][1]])
            level2_terms.append((A_N6, B_N6))
            
            # N7 = (X[0,1]-X[1,1])(Y[1,0]+Y[1,1])
            A_N7 = combine_lcs([X_2x2[0][1], {k: -v for k, v in X_2x2[1][1].items()}])
            B_N7 = combine_lcs([Y_2x2[1][0], Y_2x2[1][1]])
            level2_terms.append((A_N7, B_N7))
            
            # Now, for each level-2 term Nk, add a rank-1 term to U, V, W.
            for k_idx, (A_lc, B_lc) in enumerate(level2_terms):
                # A_lc is a dict of (i,j) -> coeff for A entries
                # B_lc is a dict of (j,k) -> coeff for B entries
                
                # The contribution to C is determined by n_to_block[k_idx] and c_coeffs[j_idx].
                # n_to_block[k_idx] gives the contribution to (X*Y)[r,c] for r,c in 0,1.
                # c_coeffs[j_idx] gives the coefficient for this level-1 term in each C block.
                
                # For each (r,c) in 0,1, the contribution to C block (block_idx) is:
                # n_to_block[k_idx][block_idx] * c_coeffs[j_idx][block_idx]
                
                # C block 0 = C11, block 1 = C12, block 2 = C21, block 3 = C22
                # C11: rows 0-1, cols 0-1
                # C12: rows 0-1, cols 2-3
                # C21: rows 2-3, cols 0-1
                # C22: rows 2-3, cols 2-3
                
                # For (r,c) in 0,1:
                # C11[r,c] += n_to_block[k_idx][0] * c_coeffs[j_idx][0] * Nk
                # C12[r,c] += n_to_block[k_idx][1] * c_coeffs[j_idx][1] * Nk
                # C21[r,c] += n_to_block[k_idx][2] * c_coeffs[j_idx][2] * Nk
                # C22[r,c] += n_to_block[k_idx][3] * c_coeffs[j_idx][3] * Nk
                
                # Nk = A_lc * B_lc
                # So the contribution to C[i,k] is:
                # sum over (i,j) in A_lc and (j,k) in B_lc of:
                #   A_lc[(i,j)] * B_lc[(j,k)] * n_to_block[k_idx][block] * c_coeffs[j_idx][block]
                
                # For the rank-1 term:
                # U[i*4+j, col] = A_lc[(i,j)] for all (i,j) in A_lc
                # V[j*4+k, col] = B_lc[(j,k)] for all (j,k) in B_lc
                # W[k*4+i, col] = sum of contributions to C[i,k]
                
                # Let me compute W contributions.
                W_contrib = {}  # (i,k) -> coeff
                
                for (i, j), a_coeff in A_lc.items():
                    for (j2, k), b_coeff in B_lc.items():
                        if j != j2:
                            continue
                        # This term contributes to C[i,k]
                        for block_idx in range(4):
                            n_coeff = n_to_block[k_idx][block_idx]
                            c_coeff = c_coeffs[j_idx][block_idx]
                            total_coeff = n_coeff * c_coeff
                            if total_coeff != 0:
                                # Determine the row and col in C
                                if block_idx == 0:  # C11
                                    ci, ck = i, k
                                elif block_idx == 1:  # C12
                                    ci, ck = i, k + 2
                                elif block_idx == 2:  # C21
                                    ci, ck = i + 2, k
                                elif block_idx == 3:  # C22
                                    ci, ck = i + 2, k + 2
                                else:
                                    continue
                                W_contrib[(ci, ck)] = W_contrib.get((ci, ck), 0) + a_coeff * b_coeff * total_coeff
                
                # Now fill in U, V, W for this rank-1 term.
                for (i, j), a_coeff in A_lc.items():
                    U[i * 4 + j, col] += a_coeff
                
                for (j, k), b_coeff in B_lc.items():
                    V[j * 4 + k, col] += b_coeff
                
                for (i, k), w_coeff in W_contrib.items():
                    W[k * 4 + i, col] += w_coeff
                
                col += 1
        
        return U, V, W

    def build_strassen_decomposition():
        """
        Build the full 4x4*4x9 decomposition using Strassen for the
        first 8 columns (two 4x4*4x4 subproblems) and schoolbook for the last column.
        
        Rank: 49 + 49 + 16 = 114
        """
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3), rank 49
        U1, V1, W1 = _strassen_4x4x4_impl()
        U_parts.append(U1)
        V_parts.append(V1)
        W_parts.append(W1)
        
        # Part 2: 4x4 * 4x4 (columns 4-7), rank 49
        U2, V2, W2 = _strassen_4x4x4_impl()
        # Need to shift the column indices in V
        # V2[j*4+k, c] should become V[j*4+(k+4), c]
        V2_shifted = np.zeros((16, 49))
        for j in range(4):
            for k in range(4):
                V2_shifted[j*4 + (k+4), :] = V2[j*4 + k, :]
        U_parts.append(U2)
        V_parts.append(V2_shifted)
        W_parts.append(W2)
        
        # Part 3: 4x4 * 4x1 (column 8), rank 16 (schoolbook)
        U3 = np.zeros((16, 16))
        V3 = np.zeros((16, 16))
        W3 = np.zeros((16, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                U3[i*4+j, col] = 1
                V3[j*4+8, col] = 1  # column 8
                W3[8*4+i, col] = 1
                col += 1
        U_parts.append(U3)
        V_parts.append(V3)
        W_parts.append(W3)
        
        U_full = np.hstack(U_parts)
        V_full = np.hstack(V_parts)
        W_full = np.hstack(W_parts)
        
        rank = U_full.shape[1]
        return U_full, V_full, W_full, rank

    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try the Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = build_strassen_decomposition()
        if check_validity(U_str, V_str, W_str):
            best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
        else:
            # Strassen decomposition is invalid, fall back to schoolbook
            pass
    except Exception as e:
        # If Strassen construction fails, fall back to schoolbook
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
