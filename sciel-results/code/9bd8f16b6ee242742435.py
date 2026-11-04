"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.

Design:
  - Decompose A (2x3) into a 2x2 block A11 and a 2x1 block A12.
  - Decompose B (3x4) into a 2x2 block B22 and a 1x2 block B21.
  - C = A11*B22 + A12*B21.
  - Use Strassen for each 2x2 multiplication (rank 7 each).
  - Use schoolbook for each 2x1x2 multiplication (rank 4 each).
  - Total rank = 7 + 7 + 4 + 4 = 22.

Convention: W[k*n+i, r] with n=2 (rows of A), p=4 (cols of B).
U shape: (n*m, rank) = (6, 22)
V shape: (m*p, rank) = (12, 22)
W shape: (p*n, rank) = (8, 22)
"""

import time

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Return a dict with keys 'U', 'V', 'W' representing a rank-22 exact
    tensor decomposition of 2x3 * 3x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22

    U = [[0] * rank for _ in range(n * m)]      # 6 x 22
    V = [[0] * rank for _ in range(m * p)]      # 12 x 22
    W = [[0] * rank for _ in range(p * n)]      # 8 x 22

    # Column layout:
    # Cols 0-6:   Strassen for A11 * B22 (top-left 2x2 * top-left 2x2)
    # Cols 7-13:  Strassen for A12 * B21 (right 2x1 * right 1x2)
    # Cols 14-17: Schoolbook for A12 * B21 first part (2x1 * 1x2 -> 2x2)
    # Wait, let me re-think the block decomposition.

    # A is 2x3:
    #   A = [A11 | A12] where A11 is 2x2 (cols 0,1), A12 is 2x1 (col 2)
    # B is 3x4:
    #   B = [B11; B21; B31] where B11 is 2x4 (rows 0,1), B21 is 1x4 (row 2)
    # Actually for matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need to split the summation index j.
    # 
    # Let's split j into j=0,1 (first 2) and j=2 (last 1).
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2] * B[2, :]
    #
    # A[:, 0:2] is 2x2, B[0:2, :] is 2x4.
    # A[:, 2] is 2x1, B[2, :] is 1x4.
    #
    # For the first product: (2x2) * (2x4)
    #   Split B[0:2, :] into B[0:2, 0:2] (2x2) and B[0:2, 2:4] (2x2)
    #   So A[:, 0:2] * B[0:2, 0:2] is 2x2 * 2x2 = use Strassen (rank 7)
    #   And A[:, 0:2] * B[0:2, 2:4] is 2x2 * 2x2 = use Strassen (rank 7)
    #
    # For the second product: (2x1) * (1x4)
    #   Split B[2, :] into B[2, 0:2] (1x2) and B[2, 2:4] (1x2)
    #   A[:, 2] * B[2, 0:2] is 2x1 * 1x2 = schoolbook (rank 4)
    #   A[:, 2] * B[2, 2:4] is 2x1 * 1x2 = schoolbook (rank 4)
    #
    # Total: 7 + 7 + 4 + 4 = 22. Correct.

    # Helper: Strassen decomposition for 2x2 * 2x2
    # Inputs: a 2x2 matrix X (indexed X[r, c]), a 2x2 matrix Y (indexed Y[r, c])
    # Output: 7 rank-1 terms (u_coeff, v_coeff, w_idx)
    # where the product Z = X * Y has entries Z[i, j] = sum_c u_c(i) * v_c(j) * w_c
    # 
    # Standard Strassen:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    def strassen_2x2(u_base, v_base, w_base):
        """
        Add 7 Strassen terms starting at column u_base for U, v_base for V, w_base for W.
        u_base, v_base, w_base are offsets into the U, V, W arrays.
        The U indices are u_base + 0..3 (for the 2x2 block of A)
        The V indices are v_base + 0..3 (for the 2x2 block of B)
        The W indices are w_base + 0..3 (for the 2x2 block of C)
        """
        # X indices in U: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Y indices in V: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Z indices in W: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # M1: (X00 + X11) * (Y00 + Y11)
        U[u_base + 0][w_base + 0] += 1
        U[u_base + 3][w_base + 0] += 1
        V[v_base + 0][w_base + 0] += 1
        V[v_base + 3][w_base + 0] += 1
        # Z00 += M1, Z11 += M1
        W[w_base + 0][w_base + 0] += 1
        W[w_base + 3][w_base + 0] += 1
        
        # M2: (X10 + X11) * Y00
        U[u_base + 2][w_base + 1] += 1
        U[u_base + 3][w_base + 1] += 1
        V[v_base + 0][w_base + 1] += 1
        # Z10 += M2, Z11 -= M2
        W[w_base + 2][w_base + 1] += 1
        W[w_base + 3][w_base + 1] -= 1
        
        # M3: X00 * (Y01 - Y11)
        U[u_base + 0][w_base + 2] += 1
        V[v_base + 1][w_base + 2] += 1
        V[v_base + 3][w_base + 2] -= 1
        # Z01 += M3, Z11 += M3
        W[w_base + 1][w_base + 2] += 1
        W[w_base + 3][w_base + 2] += 1
        
        # M4: X11 * (Y10 - Y00)
        U[u_base + 3][w_base + 3] += 1
        V[v_base + 2][w_base + 3] += 1
        V[v_base + 0][w_base + 3] -= 1
        # Z00 += M4, Z10 += M4
        W[w_base + 0][w_base + 3] += 1
        W[w_base + 2][w_base + 3] += 1
        
        # M5: (X00 + X01) * Y11
        U[u_base + 0][w_base + 4] += 1
        U[u_base + 1][w_base + 4] += 1
        V[v_base + 3][w_base + 4] += 1
        # Z00 -= M5, Z01 += M5
        W[w_base + 0][w_base + 4] -= 1
        W[w_base + 1][w_base + 4] += 1
        
        # M6: (X10 - X00) * (Y00 + Y01)
        U[u_base + 2][w_base + 5] += 1
        U[u_base + 0][w_base + 5] -= 1
        V[v_base + 0][w_base + 5] += 1
        V[v_base + 1][w_base + 5] += 1
        # Z11 += M6
        W[w_base + 3][w_base + 5] += 1
        
        # M7: (X01 - X11) * (Y10 + Y11)
        U[u_base + 1][w_base + 6] += 1
        U[u_base + 3][w_base + 6] -= 1
        V[v_base + 2][w_base + 6] += 1
        V[v_base + 3][w_base + 6] += 1
        # Z00 += M7
        W[w_base + 0][w_base + 6] += 1

    # Helper: Schoolbook for 2x1 * 1x2 -> 2x2
    # X is 2x1 (indices 0, 1 in U), Y is 1x2 (indices 0, 1 in V), Z is 2x2 (indices 0..3 in W)
    # Z[i, j] = X[i] * Y[j]
    # 4 terms:
    # Z00 = X0 * Y0
    # Z01 = X0 * Y1
    # Z10 = X1 * Y0
    # Z11 = X1 * Y1

    def schoolbook_2x1x2(u_base, v_base, w_base):
        """
        Add 4 schoolbook terms for (2x1) * (1x2).
        u_base: offset for U (2 rows)
        v_base: offset for V (2 cols)
        w_base: offset for W (4 entries, row-major: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1))
        """
        # Z00 = X0 * Y0
        U[u_base + 0][w_base + 0] += 1
        V[v_base + 0][w_base + 0] += 1
        W[w_base + 0][w_base + 0] += 1
        
        # Z01 = X0 * Y1
        U[u_base + 0][w_base + 1] += 1
        V[v_base + 1][w_base + 1] += 1
        W[w_base + 1][w_base + 1] += 1
        
        # Z10 = X1 * Y0
        U[u_base + 1][w_base + 2] += 1
        V[v_base + 0][w_base + 2] += 1
        W[w_base + 2][w_base + 2] += 1
        
        # Z11 = X1 * Y1
        U[u_base + 1][w_base + 3] += 1
        V[v_base + 1][w_base + 3] += 1
        W[w_base + 3][w_base + 3] += 1

    # Now map the blocks to actual indices.
    # A is 2x3, stored in U with row index = i * m + j = i * 3 + j
    #   i in {0, 1}, j in {0, 1, 2}
    #   A11 (i, j) for i in {0,1}, j in {0,1}: U indices (i*3 + j)
    #   A12 (i, j=2) for i in {0,1}: U indices (i*3 + 2)
    #
    # B is 3x4, stored in V with row index = j * p + k = j * 4 + k
    #   j in {0, 1, 2}, k in {0, 1, 2, 3}
    #   B[0:2, 0:2] (j in {0,1}, k in {0,1}): V indices (j*4 + k)
    #   B[0:2, 2:4] (j in {0,1}, k in {2,3}): V indices (j*4 + k)
    #   B[2, 0:2] (j=2, k in {0,1}): V indices (2*4 + k) = (8+k)
    #   B[2, 2:4] (j=2, k in {2,3}): V indices (2*4 + k) = (8+k)
    #
    # C is 2x4, stored in W with row index = k * n + i = k * 2 + i
    #   k in {0, 1, 2, 3}, i in {0, 1}
    #   C[0:2, 0:2] (k in {0,1}, i in {0,1}): W indices (k*2 + i)
    #   C[0:2, 2:4] (k in {2,3}, i in {0,1}): W indices (k*2 + i)

    # Block 1: A11 * B[0:2, 0:2] -> C[0:2, 0:2]
    # A11 U indices: (0*3+0)=0, (0*3+1)=1, (1*3+0)=3, (1*3+1)=4
    #   Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   So U offsets: [0, 1, 3, 4]
    # B[0:2, 0:2] V indices: (0*4+0)=0, (0*4+1)=1, (1*4+0)=4, (1*4+1)=5
    #   Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   So V offsets: [0, 1, 4, 5]
    # C[0:2, 0:2] W indices: (0*2+0)=0, (0*2+1)=1, (1*2+0)=2, (1*2+1)=3
    #   Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   So W offsets: [0, 1, 2, 3]
    
    if time.time() > deadline:
        # Safety: return what we have (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # We need to handle non-contiguous indices. Let's write a more general Strassen.
    
    def strassen_general(u_indices, v_indices, w_indices, col_offset):
        """
        u_indices: list of 4 U row indices for the 2x2 block (local order: 00, 01, 10, 11)
        v_indices: list of 4 V row indices for the 2x2 block
        w_indices: list of 4 W row indices for the 2x2 block
        col_offset: starting column in U, V, W (same for all)
        """
        u0, u1, u2, u3 = u_indices
        v0, v1, v2, v3 = v_indices
        w0, w1, w2, w3 = w_indices
        
        c = col_offset
        # M1: (X00 + X11) * (Y00 + Y11)
        U[u0][c] += 1
        U[u3][c] += 1
        V[v0][c] += 1
        V[v3][c] += 1
        W[w0][c] += 1
        W[w3][c] += 1
        
        # M2: (X10 + X11) * Y00
        U[u2][c + 1] += 1
        U[u3][c + 1] += 1
        V[v0][c + 1] += 1
        W[w2][c + 1] += 1
        W[w3][c + 1] -= 1
        
        # M3: X00 * (Y01 - Y11)
        U[u0][c + 2] += 1
        V[v1][c + 2] += 1
        V[v3][c + 2] -= 1
        W[w1][c + 2] += 1
        W[w3][c + 2] += 1
        
        # M4: X11 * (Y10 - Y00)
        U[u3][c + 3] += 1
        V[v2][c + 3] += 1
        V[v0][c + 3] -= 1
        W[w0][c + 3] += 1
        W[w2][c + 3] += 1
        
        # M5: (X00 + X01) * Y11
        U[u0][c + 4] += 1
        U[u1][c + 4] += 1
        V[v3][c + 4] += 1
        W[w0][c + 4] -= 1
        W[w1][c + 4] += 1
        
        # M6: (X10 - X00) * (Y00 + Y01)
        U[u2][c + 5] += 1
        U[u0][c + 5] -= 1
        V[v0][c + 5] += 1
        V[v1][c + 5] += 1
        W[w3][c + 5] += 1
        
        # M7: (X01 - X11) * (Y10 + Y11)
        U[u1][c + 6] += 1
        U[u3][c + 6] -= 1
        V[v2][c + 6] += 1
        V[v3][c + 6] += 1
        W[w0][c + 6] += 1

    def schoolbook_general(u_indices, v_indices, w_indices, col_offset):
        """
        u_indices: list of 2 U row indices (local 0, 1)
        v_indices: list of 2 V row indices (local 0, 1)
        w_indices: list of 4 W row indices (local 00, 01, 10, 11)
        col_offset: starting column
        """
        u0, u1 = u_indices
        v0, v1 = v_indices
        w0, w1, w2, w3 = w_indices
        
        c = col_offset
        # Z00 = X0 * Y0
        U[u0][c] += 1
        V[v0][c] += 1
        W[w0][c] += 1
        
        # Z01 = X0 * Y1
        U[u0][c + 1] += 1
        V[v1][c + 1] += 1
        W[w1][c + 1] += 1
        
        # Z10 = X1 * Y0
        U[u1][c + 2] += 1
        V[v0][c + 2] += 1
        W[w2][c + 2] += 1
        
        # Z11 = X1 * Y1
        U[u1][c + 3] += 1
        V[v1][c + 3] += 1
        W[w3][c + 3] += 1

    # Block 1: A11 * B[0:2, 0:2] -> C[0:2, 0:2], cols 0-6
    u1 = [0, 1, 3, 4]       # A11 local (0,0),(0,1),(1,0),(1,1)
    v1 = [0, 1, 4, 5]       # B[0:2, 0:2]
    w1 = [0, 1, 2, 3]       # C[0:2, 0:2]
    strassen_general(u1, v1, w1, 0)

    # Block 2: A11 * B[0:2, 2:4] -> C[0:2, 2:4], cols 7-13
    u2 = [0, 1, 3, 4]       # A11 same
    v2 = [2, 3, 6, 7]       # B[0:2, 2:4]: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    w2 = [4, 5, 6, 7]       # C[0:2, 2:4]: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7
    strassen_general(u2, v2, w2, 7)

    # Block 3: A12 * B[2, 0:2] -> C[0:2, 0:2], cols 14-17
    u3 = [2, 5]             # A12: (0,2)->2, (1,2)->5
    v3 = [8, 9]             # B[2, 0:2]: (2,0)->8, (2,1)->9
    w3 = [0, 1, 2, 3]       # C[0:2, 0:2]
    schoolbook_general(u3, v3, w3, 14)

    # Block 4: A12 * B[2, 2:4] -> C[0:2, 2:4], cols 18-21
    u4 = [2, 5]             # A12 same
    v4 = [10, 11]           # B[2, 2:4]: (2,2)->10, (2,3)->11
    w4 = [4, 5, 6, 7]       # C[0:2, 2:4]
    schoolbook_general(u4, v4, w4, 18)

    # Verify shapes
    assert len(U) == n * m, f"U rows {len(U)} != {n * m}"
    assert len(V) == m * p, f"V rows {len(V)} != {m * p}"
    assert len(W) == p * n, f"W rows {len(W)} != {p * n}"
    assert all(len(row) == rank for row in U), "U cols wrong"
    assert all(len(row) == rank for row in V), "V cols wrong"
    assert all(len(row) == rank for row in W), "W cols wrong"

    return {"U": U, "V": V, "W": W}
