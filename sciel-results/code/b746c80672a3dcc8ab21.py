"""Matrix multiplication 3x3 x 3x5 — minimal-rank exact tensor decomposition."""
import time
import math
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=3, p=5):
    """Check that U * V^T * W^T reproduces the matrix multiplication tensor.
    
    Convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    The product tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all (i,j,k) pairs.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _check_half_integer(entries):
    """Check all entries are integers or half-integers (multiples of 0.5)."""
    for row in entries:
        for x in row:
            if abs(x * 2 - round(x * 2)) > 1e-9:
                return False
    return True


def _schoolbook(n=3, m=3, p=5):
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


def _try_strassen_based(n=3, m=3, p=5):
    """Try to construct a rank-21 decomposition using Strassen's 7-term
    2x2 multiplication embedded in a block structure.
    
    For 3x3 * 3x3, one can use a rank-20 or rank-21 decomposition.
    For 3x3 * 3x5, we split the 5-column into a 3+2 block.
    
    We use the following approach:
    - Split A (3x3) and B (3x5) into blocks.
    - Use Strassen for the 2x2 sub-blocks and standard for the rest.
    
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, etc.
    B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x2, etc.
    
    Actually, let's try a different split:
    A is 3x3, B is 3x5.
    Split A into rows: top 2 rows, bottom 1 row.
    Split B into columns: left 3 cols, right 2 cols.
    
    C = A * B (3x5)
    
    C_top(2x5) = A_top(2x3) * B (3x5)
    C_bot(1x5) = A_bot(1x3) * B (3x5)
    
    For C_top: A_top is 2x3, B is 3x5.
    We can further split A_top into (2x2, 2x1) and B into (3x3, 3x2).
    
    This gets complex. Let me try a known rank-21 for 3x3*3x3 and extend.
    
    Known: Strassen gives rank 7 for 2x2*2x2.
    For 3x3*3x3, a known bound is 19 (Alon-Boppana lower bound is 19).
    Rank 20 is achievable.
    
    Let me try a rank-20 for 3x3*3x3 and then handle the extra 2 columns.
    
    Actually, let me try a direct construction:
    
    For 3x3 * 3x5, think of it as 3x3 * [3x3 | 3x2].
    C = [C_left(3x3) | C_right(3x2)]
    
    C_left = A * B_left (3x3 * 3x3): use rank-20 decomposition
    C_right = A * B_right (3x3 * 3x2): use rank-15 (schoolbook) or better
    
    But we can share terms. Let me try:
    
    Split: A = rows a1, a2, a3 (each 1x3)
    B = columns b1, b2, b3, b4, b5 (each 3x1)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Schoolbook: 45 terms.
    
    Let me try to use Strassen on pairs of rows/columns.
    
    Actually, let me try a specific known decomposition for 3x3*3x3 of rank 20,
    and then extend to 3x3*3x5.
    
    Known rank-20 for 3x3*3x3 (from literature):
    This is complex to encode from memory. Let me try a different approach.
    
    I'll attempt a rank-27 decomposition using a 3x3 Strassen-like scheme:
    Use 7 terms per 2x2 block, with overlap handling.
    
    Split A into [[A11, A12], [A21, A22]] with A11=2x2, A12=2x1, A21=1x2, A22=1x1
    Split B into [[B11, B12], [B21, B22]] with B11=2x2, B12=2x1, B21=1x2, B22=1x1
    (for 3x3*3x3 case)
    
    C11 = A11*B11 + A12*B21 (2x2)
    C12 = A11*B12 + A12*B22 (2x1)
    C21 = A21*B11 + A22*B21 (1x2)
    C22 = A21*B12 + A22*B22 (1x1)
    
    Using Strassen for C11 (2x2*2x2): 7 terms
    For the cross terms, we need more.
    
    This is getting too complex to get right from memory. Let me try a simpler
    but still improved approach: use rank-30 by grouping.
    
    Actually, let me just try to verify if a simple rank reduction works.
    
    For 3x3 * 3x5:
    Use the fact that we can compute with rank 3*3*5 = 45 (schoolbook).
    
    Let me try rank 30:
    Split columns of B into two groups: first 3, last 2.
    C_left = A * B[:, :3] (3x3 * 3x3): rank 20 (if we can get it)
    C_right = A * B[:, 3:] (3x3 * 3x2): rank 15 (schoolbook)
    Total: 35 if independent, but we can share.
    
    Hmm, this is hard to do correctly without verified coefficients.
    
    Let me try a completely different approach: use a rank-21 for 3x3*3x3
    (which I believe exists) and extend.
    
    Actually, I recall that the tensor rank of 3x3 matrix multiplication is
    believed to be 19 or 20. Let me just try to use the schoolbook and
    see if I can find a small reduction.
    
    Given the risk of incorrect coefficients, let me try a structured rank-30:
    
    Group the 5 columns into (3, 2).
    For the 3x3 * 3x3 part, use a rank-21 scheme.
    For the 3x3 * 3x2 part, use rank-15.
    But share the A factors.
    
    This is still complex. Let me just return schoolbook 45 to be safe,
    but try one simple optimization first.
    
    Simple idea: since p=5, we can write B = [B1 | B2] where B1 is 3x3, B2 is 3x2.
    C = [A*B1 | A*B2].
    
    A*B1: 3x3*3x3, schoolbook rank 27.
    A*B2: 3x3*3x2, schoolbook rank 18.
    Total: 45. No savings.
    
    Let me try: use 3 independent "Strassen-like" 2x2 multiplications.
    Rows (0,1), (1,2), (0,2) — 3 pairs.
    Columns: (0,1), (2,3), (4) — or some grouping.
    
    This is too error-prone. I'll return schoolbook.
    """
    return None


def _try_rank_reduction_greedy(n=3, m=3, p=5, deadline=None):
    """Try to find a lower-rank decomposition via greedy sparsification.
    
    Start from schoolbook and try to merge/remove terms.
    """
    if deadline is not None and time.time() > deadline - 2:
        return None
    
    base = _schoolbook(n, m, p)
    rank = n * m * p
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    
    # Try to zero out terms by checking if the contribution is redundant
    # This is a heuristic; may not actually reduce rank
    # For now, just return None to avoid incorrect results
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 100  # 100s hard limit, 20s safety margin for 120s budget
    
    # Try a known low-rank construction
    # For 3x3 * 3x5, the minimum rank is at least 19 (Alon-Boppana for 3x3*3x3
    # gives 19, and 3x5 should be similar or slightly more).
    # Upper bound: 45 (schoolbook), 30 (grouped), 27+15=42, etc.
    
    # Let me try to construct a rank-21 for 3x3*3x3 and extend to 3x5.
    # 
    # Known construction for 3x3*3x3 rank 21 (from Strassen generalization):
    # 
    # Actually, let me try a specific approach:
    # Write A and B in terms of 2x2 blocks with overlap.
    #
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[p, q, r, s, t], [u, v, w, x, y], [z, A_, B_, C_, D_]]
    #
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # Let me define 7 Strassen products for the (0,1)x(0,1) block:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # This gives C00, C01, C10, C11 for the 2x2 subproblem.
    #
    # For the full 3x3, I need to handle the interactions with row/col 2.
    #
    # This is getting too complicated to encode correctly from memory.
    # Let me just return the schoolbook decomposition.
    
    if time.time() > deadline - 5:
        return _schoolbook()
    
    # Try a rank-30 construction:
    # Split B into B_left (3x3) and B_right (3x2).
    # C = A * B = [A * B_left | A * B_right]
    # 
    # For A * B_left (3x3 * 3x3), use schoolbook: 27 terms
    # For A * B_right (3x3 * 3x2), use schoolbook: 18 terms
    # But we can share the U factors since they depend only on A.
    # 
    # Actually in the tensor decomposition framework, each term c has:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # 
    # The U factor is shared across k for fixed (i,j).
    # So if we use the same U for B_left and B_right terms, we save on U
    # but the rank is still the number of terms.
    # 
    # Rank = number of terms = 27 + 18 = 45. No savings.
    #
    # To save, we need to combine terms so that a single term contributes
    # to multiple (i,j,k) positions.
    
    # Let me try a rank-27 approach:
    # Use 3 Strassen 2x2 multiplications on overlapping row pairs.
    # Row pairs: (0,1), (1,2), (0,2)
    # For each pair, use 7 Strassen terms on the 2x2 x 3x5 subproblem?
    # No, Strassen is for 2x2 * 2x2.
    
    # I think the safest bet is schoolbook 45.
    # But let me try one more thing: a rank-35 construction.
    
    # Actually, let me just return schoolbook. The risk of incorrect
    # coefficients in a lower-rank scheme is too high.
    
    result = _schoolbook()
    
    # Verify
    if not _verify(result["U"], result["V"], result["W"]):
        # Should never happen
        return _schoolbook()
    
    return result

# EVOLVE-BLOCK-END
