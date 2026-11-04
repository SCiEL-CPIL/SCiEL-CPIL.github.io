import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook_2x3x6():
    """
    Schoolbook matrix multiplication tensor decomposition for 2x3x6.
    Rank = n * m * p = 36.
    
    U shape: (n*m, rank) = (6, 36)
    V shape: (m*p, rank) = (18, 36)
    W shape: (p*n, rank) = (12, 36)
    
    Index conventions:
      U[i*m + j, r]
      V[j*p + k, r]
      W[k*n + i, r]
    
    The tensor decomposition must satisfy:
      sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    for all i, k, j, l.
    """
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2_block(U_base, V_base, W_base, i0, k0, rank_offset, n, m, p):
    """
    Add a Strassen 2x2x2 block to the decomposition.
    i0: row offset in A (2 rows: i0, i0+1)
    k0: col offset in B (2 cols: k0, k0+1)
    
    Strassen products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
    Output:
      C11 = M1 + M4 - M5 - M7
      C12 = M3 + M5
      C21 = M2 + M7
      C22 = M1 - M2 + M3 + M4
    """
    # U factors for A submatrix (2x2): rows i0, i0+1; cols 0,1
    # V factors for B submatrix (2x2): rows 0,1; cols k0, k0+1
    # W factors for C submatrix (2x2): rows k0, k0+1; cols i0, i0+1
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11 + A22 -> row i0: 1, row i0+1: 1
    # V: B11 + B22 -> row 0: 1, row 1: 1
    # W: C11 + C22 -> (k0, i0): 1, (k0+1, i0+1): 1
    U_base[i0 * m + 0][rank_offset] = 1
    U_base[(i0 + 1) * m + 1][rank_offset] = 1
    V_base[0 * p + k0][rank_offset] = 1
    V_base[1 * p + (k0 + 1)][rank_offset] = 1
    W_base[k0 * n + i0][rank_offset] = 1
    W_base[(k0 + 1) * n + (i0 + 1)][rank_offset] = 1
    
    # M2 = (A21+A22)B11
    # U: A21 + A22 -> row i0+1: col 1: 1, col 2: 1
    # V: B11 -> row 0, col k0
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 1] = 1
    U_base[(i0 + 1) * m + 2][rank_offset + 1] = 1
    V_base[0 * p + k0][rank_offset + 1] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 1] = 1
    
    # M3 = A11(B12-B22)
    # U: A11 -> row i0, col 0
    # V: B12 - B22 -> row 0, col k0+1: 1; row 1, col k0+1: -1
    # W: C12 -> (k0+1, i0)
    U_base[i0 * m + 0][rank_offset + 2] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 2] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 2] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 2] = 1
    
    # M4 = A22(B21-B11)
    # U: A22 -> row i0+1, col 1
    # V: B21 - B11 -> row 1, col k0: 1; row 0, col k0: -1
    # W: C11 -> (k0, i0)
    U_base[(i0 + 1) * m + 1][rank_offset + 3] = 1
    V_base[1 * p + k0][rank_offset + 3] = 1
    V_base[0 * p + k0][rank_offset + 3] = -1
    W_base[k0 * n + i0][rank_offset + 3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11 + A12 -> row i0: col 0: 1, col 1: 1
    # V: B22 -> row 1, col k0+1
    # W: C11 -> (k0, i0) with -1; C12 -> (k0+1, i0) with 1
    U_base[i0 * m + 0][rank_offset + 4] = 1
    U_base[i0 * m + 1][rank_offset + 4] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 4] = 1
    W_base[k0 * n + i0][rank_offset + 4] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21 - A11 -> row i0+1, col 1: 1; row i0, col 0: -1
    # V: B11 + B12 -> row 0, col k0: 1; row 0, col k0+1: 1
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 5] = 1
    U_base[i0 * m + 0][rank_offset + 5] = -1
    V_base[0 * p + k0][rank_offset + 5] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 5] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12 - A22 -> row i0, col 1: 1; row i0+1, col 1: -1
    # V: B21 + B22 -> row 1, col k0: 1; row 1, col k0+1: 1
    # W: C21 -> (k0, i0+1) with 1; C11 -> (k0, i0) with -1
    U_base[i0 * m + 1][rank_offset + 6] = 1
    U_base[(i0 + 1) * m + 1][rank_offset + 6] = -1
    V_base[1 * p + k0][rank_offset + 6] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + i0][rank_offset + 6] = -1


def _assemble_strassen():
    """
    Assemble rank-33 decomposition for 2x3x6:
    - Split columns into three 2-column blocks: cols 0-1, 2-3, 4-5
    - For each block, split rows: first 2 rows use Strassen (rank 7), last row uses schoolbook (rank 2)
    - Total: 3 * (7 + 2) = 27... wait, that's not right.
    
    Actually:
    A is 2x3, B is 3x6.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Split B into three 3x2 blocks: B_0 (cols 0-1), B_1 (cols 2-3), B_2 (cols 4-5)
    C = A * B_0 + A * B_1 + A * B_2
    
    For each A * B_t (where B_t is 3x2):
    Split A into A_top (2x3) and A_bottom (1x3)
    A * B_t = A_top * B_t + A_bottom * B_t
    
    A_top * B_t: 2x3x2. Split B_t rows into two: B_t_top (2x2), B_t_bottom (1x2)
    A_top * B_t_top: 2x2x2 -> Strassen rank 7
    A_top * B_t_bottom: 2x1x2 -> schoolbook rank 4
    A_bottom * B_t_top: 1x2x2 -> schoolbook rank 4
    A_bottom * B_t_bottom: 1x1x2 -> schoolbook rank 2
    Total per block: 7 + 4 + 4 + 2 = 17
    Total: 3 * 17 = 51. That's worse.
    
    Let me rethink. The previous analysis suggested rank 33 using:
    - 2x2x6 part: split into three 2x2x2 Strassen blocks = 3*7 = 21
    - 2x1x6 part: schoolbook = 12
    Total = 33.
    
    But this requires splitting A into a 2x3 part and a 1x3 part? No.
    A is 2x3. We can't split A into 2x2 and 2x1 in a way that works for all B columns.
    
    Actually, we can split the columns of B into groups. For each group of 2 columns:
    C[:, k0:k0+2] = A * B[:, k0:k0+2]
    where B[:, k0:k0+2] is 3x2.
    
    For this 2x3x2 subproblem:
    Split A rows? No, A is 2x3.
    Split B rows: B[:, k0:k0+2] = [B_top (2x2); B_bottom (1x2)]
    C[:, k0:k0+2] = A * B_top + A * B_bottom
    
    A * B_top: 2x3x2. Split A columns: A = [A_left (2x2), A_right (2x1)]
    A_left * B_top: 2x2x2 -> Strassen rank 7
    A_right * B_top: 2x1x2 -> schoolbook rank 4
    A * B_bottom: 2x1x2 -> schoolbook rank 4
    Total per 2-col block: 7 + 4 + 4 = 15
    Total: 3 * 15 = 45. Still worse.
    
    The rank 33 claim from before was based on:
    "Rank of 2x2x6: Split into three 2x2x2 blocks. Each 2x2x2 has rank 7 (Strassen). Total rank 3x7=21."
    "And the 2x1x6 part using schoolbook. Total Rank 33."
    
    But 2x2x6 + 2x1x6 doesn't make sense for 2x3x6.
    
    Let me just implement the schoolbook rank 36 solution which is guaranteed correct.
    Any improvement needs careful verification.
    
    Given the complexity and risk of errors in more complex decompositions, I'll return schoolbook.
    """
    return _schoolbook_2x3x6()


def _verify_full(U, V, W):
    """
    Verify that the tensor decomposition is correct.
    sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    """
    from fractions import Fraction
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[l * p + k][r]) * Fraction(W[k * n + i][r])
                    expected = Fraction(1) if j == l else Fraction(0)
                    if total != expected:
                        return False
    return True


def construct():
    deadline = time.time() + 110.0
    try:
        result = _assemble_strassen()
        if time.time() < deadline:
            if _verify_full(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    return _schoolbook_2x3x6()

# EVOLVE-BLOCK-END
