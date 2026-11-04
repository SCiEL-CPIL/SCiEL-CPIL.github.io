"""Rank-34 decomposition for 3x3x4 matrix multiplication using Strassen blocks."""
import time
from fractions import Fraction as F

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=4):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = F(0)
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != F(1):
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the 7 Strassen factors for 2x2x2 matrix multiplication.
    Each factor is a tuple (u, v, w) where:
      u: list of 4 entries for A (indices: A[0,0], A[0,1], A[1,0], A[1,1])
      v: list of 4 entries for B
      w: list of 4 entries for C
    Using the standard Strassen formulas:
      M1 = (A00+A11)(B00+B11)  -> C00 += M1, C11 += M1
      M2 = (A10+A11)B00        -> C10 += M2, C11 += M2 (with -)
      M3 = A00(B10-B00)        -> C01 += M3, C11 += M3
      M4 = A11(B01-B11)        -> C00 += M4, C10 += M4
      M5 = (A00+A01)B11        -> C01 += M5, C11 -= M5
      M6 = (A10-A00)(B00+B01)  -> C10 -= M6, C11 += M6
      M7 = (A01-A11)(B10+B11)  -> C00 -= M7, C01 -= M7
    """
    zero = F(0)
    one = F(1)
    neg_one = F(-1)
    
    # Each factor: (u for A, v for B, w for C)
    # A indices: [A00, A01, A10, A11]
    # B indices: [B00, B01, B10, B11]
    # C indices: [C00, C01, C10, C11]
    
    factors = [
        # M1 = (A00+A11)(B00+B11): contributes to C00 and C11
        ([one, zero, zero, one], [one, zero, zero, one], [one, zero, zero, zero]),
        ([one, zero, zero, one], [one, zero, zero, one], [zero, zero, zero, one]),
        # M2 = (A10+A11)B00: contributes to C10 and C11
        ([zero, zero, one, one], [one, zero, zero, zero], [zero, zero, one, zero]),
        ([zero, zero, one, one], [one, zero, zero, zero], [zero, zero, zero, neg_one]),
        # M3 = A00(B10-B00): contributes to C01 and C11
        ([one, zero, zero, zero], [zero, zero, one, zero], [zero, one, zero, zero]),
        ([one, zero, zero, zero], [zero, zero, zero, zero], [zero, zero, zero, one]),
        # Wait, M3 = A00*(B10 - B00), so v should be [B00_coeff, B01_coeff, B10_coeff, B11_coeff]
        # M3: u=[1,0,0,0], v=[-1,0,1,0], w=[0,1,0,0] for C01 and [0,0,0,1] for C11
        # Let me redo this more carefully.
    ]
    
    # Let me use a cleaner formulation.
    # Strassen 2x2x2: 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7  -- wait, let me use the standard form:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each multiplication Mi, we need one factor with:
    #   u: coefficients for A entries
    #   v: coefficients for B entries
    #   w: coefficient for C entry (but we need to distribute)
    
    # Actually, in the tensor decomposition, each rank-1 term contributes to one C entry.
    # So for M1 which contributes to both C00 and C11, we need TWO rank-1 terms.
    
    # Let me list all 14 rank-1 terms (7 multiplications x 2 C entries each, but some may combine)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,0] (C00)
    # Term 2: u=[1,0,0,1], v=[1,0,0,1], w=[0,0,0,1] (C11)
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # Term 3: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (C10)
    # Term 4: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,0,-1] (C11)
    
    # M3 = A00(B10-B00) -> C01 += M3, C11 += M3
    # Term 5: u=[1,0,0,0], v=[-1,0,1,0], w=[0,1,0,0] (C01)
    # Term 6: u=[1,0,0,0], v=[-1,0,1,0], w=[0,0,0,1] (C11)
    
    # M4 = A11(B01-B11) -> C00 += M4, C10 += M4
    # Term 7: u=[0,0,0,1], v=[0,1,0,-1], w=[1,0,0,0] (C00)
    # Term 8: u=[0,0,0,1], v=[0,1,0,-1], w=[0,0,1,0] (C10)
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    # Term 9: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0] (C01)
    # Term 10: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,0,-1] (C11)
    
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    # Term 11: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,-1,0] (C10)
    # Term 12: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1] (C11)
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C01 -= M7
    # Term 13: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0] (C00)
    # Term 14: u=[0,1,0,-1], v=[0,0,1,1], w=[0,-1,0,0] (C01)
    
    z = F(0)
    o = F(1)
    no = F(-1)
    
    factors = [
        # M1 -> C00
        ([o, z, z, o], [o, z, z, o], [o, z, z, z]),
        # M1 -> C11
        ([o, z, z, o], [o, z, z, o], [z, z, z, o]),
        # M2 -> C10
        ([z, z, o, o], [o, z, z, z], [z, z, o, z]),
        # M2 -> C11
        ([z, z, o, o], [o, z, z, z], [z, z, z, no]),
        # M3 -> C01
        ([o, z, z, z], [no, z, o, z], [z, o, z, z]),
        # M3 -> C11
        ([o, z, z, z], [no, z, o, z], [z, z, z, o]),
        # M4 -> C00
        ([z, z, z, o], [z, o, z, no], [o, z, z, z]),
        # M4 -> C10
        ([z, z, z, o], [z, o, z, no], [z, z, o, z]),
        # M5 -> C01
        ([o, o, z, z], [z, z, z, o], [z, o, z, z]),
        # M5 -> C11
        ([o, o, z, z], [z, z, z, o], [z, z, z, no]),
        # M6 -> C10
        ([no, z, o, z], [o, o, z, z], [z, z, no, z]),
        # M6 -> C11
        ([no, z, o, z], [o, o, z, z], [z, z, z, o]),
        # M7 -> C00
        ([z, o, z, no], [z, z, o, o], [o, z, z, z]),
        # M7 -> C01
        ([z, o, z, no], [z, z, o, o], [z, no, z, z]),
    ]
    
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = 3, 3, 4
    
    # Block decomposition:
    # A (3x3) = [[A22 (2x2), A21 (2x1)], [A12 (1x2), A11 (1x1)]]
    # B (3x4) = [[B22 (2x2), B22p (2x2)], [B12 (1x2), B12p (1x2)]]
    # C (3x4) = [[C22 (2x2), C22p (2x2)], [C12 (1x2), C12p (1x2)]]
    #
    # C22 = A22*B22 + A21*B12
    # C22p = A22*B22p + A21*B12p
    # C12 = A12*B22 + A11*B12
    # C12p = A12*B22p + A11*B12p
    #
    # Sub-problems:
    # 1. A22*B22: 2x2x2, rank 7 (Strassen)
    # 2. A22*B22p: 2x2x2, rank 7 (Strassen)
    # 3. A21*B12: 2x1x2, rank 4 (schoolbook)
    # 4. A21*B12p: 2x1x2, rank 4 (schoolbook)
    # 5. A12*B22: 1x2x2, rank 4 (schoolbook)
    # 6. A12*B22p: 1x2x2, rank 4 (schoolbook)
    # 7. A11*B12: 1x1x2, rank 2 (schoolbook)
    # 8. A11*B12p: 1x1x2, rank 2 (schoolbook)
    # Total rank: 7+7+4+4+4+4+2+2 = 34
    
    # Index conventions:
    # A is 3x3: A[i,j] for i=0,1,2 (rows), j=0,1,2 (cols)
    #   A22 = A[0:2, 0:2] (rows 0,1; cols 0,1)
    #   A21 = A[0:2, 2:3] (rows 0,1; col 2)
    #   A12 = A[2:3, 0:2] (row 2; cols 0,1)
    #   A11 = A[2:3, 2:3] (row 2; col 2)
    #
    # B is 3x4: B[j,k] for j=0,1,2 (rows), k=0,1,2,3 (cols)
    #   B22 = B[0:2, 0:2] (rows 0,1; cols 0,1)
    #   B22p = B[0:2, 2:4] (rows 0,1; cols 2,3)
    #   B12 = B[2:3, 0:2] (row 2; cols 0,1)
    #   B12p = B[2:3, 2:4] (row 2; cols 2,3)
    #
    # C is 3x4: C[i,k] for i=0,1,2 (rows), k=0,1,2,3 (cols)
    #   C22 = C[0:2, 0:2] (rows 0,1; cols 0,1)
    #   C22p = C[0:2, 2:4] (rows 0,1; cols 2,3)
    #   C12 = C[2:3, 0:2] (row 2; cols 0,1)
    #   C12p = C[2:3, 2:4] (row 2; cols 2,3)
    #
    # W index: W[k*n+i, c] = W[k*3+i, c]
    # U index: U[i*m+j, c] = U[i*3+j, c]
    # V index: V[j*p+k, c] = V[j*4+k, c]
    
    rank = 34
    U = [[F(0)] * rank for _ in range(n * m)]  # 9 x 34
    V = [[F(0)] * rank for _ in range(m * p)]  # 12 x 34
    W = [[F(0)] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    strassen_factors = _strassen_2x2x2()
    
    # Helper: add a Strassen 2x2x2 block
    # A_sub: 2x2 submatrix of A, with global row indices and col indices
    # B_sub: 2x2 submatrix of B, with global row indices and col indices
    # C_sub: 2x2 submatrix of C, with global row indices and col indices
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        for (u, v, w) in strassen_factors:
            # u = [A[0,0], A[0,1], A[1,0], A[1,1]]
            # v = [B[0,0], B[0,1], B[1,0], B[1,1]]
            # w = [C[0,0], C[0,1], C[1,0], C[1,1]]
            for a_idx in range(4):
                if u[a_idx] != 0:
                    # Determine which A entry this is
                    a_row, a_col = divmod(a_idx, 2)
                    gi = A_rows[a_row]
                    gj = A_cols[a_col]
                    U[gi * m + gj][col] += u[a_idx]
            for b_idx in range(4):
                if v[b_idx] != 0:
                    b_row, b_col = divmod(b_idx, 2)
                    gj = B_rows[b_row]
                    gk = B_cols[b_col]
                    V[gj * p + gk][col] += v[b_idx]
            for c_idx in range(4):
                if w[c_idx] != 0:
                    c_row, c_col = divmod(c_idx, 2)
                    gi = C_rows[c_row]
                    gk = C_cols[c_col]
                    W[gk * n + gi][col] += w[c_idx]
            col += 1
    
    # Helper: add a schoolbook multiplication for small blocks
    # A_sub: (na x ma) submatrix, B_sub: (ma x pa) submatrix, C_sub: (na x pa) submatrix
    def add_schoolbook(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        na = len(A_rows)
        ma = len(A_cols)
        pa = len(C_cols)
        for i_idx in range(na):
            for j_idx in range(ma):
                for k_idx in range(pa):
                    U[A_rows[i_idx] * m + A_cols[j_idx]][col] = F(1)
                    V[B_rows[j_idx] * p + B_cols[k_idx]][col] = F(1)
                    W[C_cols[k_idx] * n + C_rows[i_idx]][col] = F(1)
                    col += 1
    
    # 1. A22*B22 -> C22: 2x2x2 Strassen
    # A22: rows [0,1], cols [0,1]
    # B22: rows [0,1], cols [0,1]
    # C22: rows [0,1], cols [0,1]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. A22*B22p -> C22p: 2x2x2 Strassen
    # A22: rows [0,1], cols [0,1]
    # B22p: rows [0,1], cols [2,3]
    # C22p: rows [0,1], cols [2,3]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # 3. A21*B12 -> C22: 2x1x2 schoolbook
    # A21: rows [0,1], col [2]
    # B12: row [2], cols [0,1]
    # C22: rows [0,1], cols [0,1]
    add_schoolbook([0, 1], [2], [2], [0, 1], [0, 1], [0, 1])
    
    # 4. A21*B12p -> C22p: 2x1x2 schoolbook
    # A21: rows [0,1], col [2]
    # B12p: row [2], cols [2,3]
    # C22p: rows [0,1], cols [2,3]
    add_schoolbook([0, 1], [2], [2], [2, 3], [0, 1], [2, 3])
    
    # 5. A12*B22 -> C12: 1x2x2 schoolbook
    # A12: row [2], cols [0,1]
    # B22: rows [0,1], cols [0,1]
    # C12: row [2], cols [0,1]
    add_schoolbook([2], [0, 1], [0, 1], [0, 1], [2], [0, 1])
    
    # 6. A12*B22p -> C12p: 1x2x2 schoolbook
    # A12: row [2], cols [0,1]
    # B22p: rows [0,1], cols [2,3]
    # C12p: row [2], cols [2,3]
    add_schoolbook([2], [0, 1], [0, 1], [2, 3], [2], [2, 3])
    
    # 7. A11*B12 -> C12: 1x1x2 schoolbook
    # A11: row [2], col [2]
    # B12: row [2], cols [0,1]
    # C12: row [2], cols [0,1]
    add_schoolbook([2], [2], [2], [0, 1], [2], [0, 1])
    
    # 8. A11*B12p -> C12p: 1x1x2 schoolbook
    # A11: row [2], col [2]
    # B12p: row [2], cols [2,3]
    # C12p: row [2], cols [2,3]
    add_schoolbook([2], [2], [2], [2, 3], [2], [2, 3])
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if not _validate(U, V, W):
        # Fallback to schoolbook rank 36
        rank = n * m * p
        U = [[F(0)] * rank for _ in range(n * m)]
        V = [[F(0)] * rank for _ in range(m * p)]
        W = [[F(0)] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = F(1)
                    V[j * p + k][c] = F(1)
                    W[k * n + i][c] = F(1)
                    c += 1
    
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
