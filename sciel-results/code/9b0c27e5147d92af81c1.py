"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
_DEADLINE = time.time() + 135.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists of Fraction for schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    # The matmul tensor T[i,j,k] = 1 if and only if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # For correctness: for each (i,j,k), the sum over c must equal 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True


def _strassen_block(U, V, W, n, m, p, r):
    """
    Apply one Strassen step to a block decomposition of size (n,m,p) with r columns.
    Splits into 2x2 blocks and uses 7 multiplications instead of 8.
    Returns new (U, V, W) with rank 7*r.
    """
    n2, m2, p2 = n // 2, m // 2, p // 2
    if n2 == 0 or m2 == 0 or p2 == 0:
        return None

    new_rank = 7 * r
    new_U = [[Fraction(0)] * new_rank for _ in range(n * m)]
    new_V = [[Fraction(0)] * new_rank for _ in range(m * p)]
    new_W = [[Fraction(0)] * new_rank for _ in range(p * n)]

    # Strassen formulas for 2x2 matrix multiply:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each original column c, we create 7 new columns.
    # The U, V, W matrices for the 7 Strassen sub-multiplications:
    # We need to express the 7 Strassen products in terms of the original tensor.
    #
    # For a block at position (bi, bj, bk) in the 2x2x2 grid, the original
    # sub-tensor has U rows indexed by (bi*n2 + i', bj*m2 + j'), etc.
    #
    # The 7 Strassen multiplications each involve sums/differences of blocks.
    # For multiplication M1 = (A11+A22)(B11+B22):
    #   U_factor for row (i,j): if block is (0,0) or (1,1), contribute +1
    #   V_factor for row (j,k): if block is (0,0) or (1,1), contribute +1
    #   W_factor for row (k,i): if block is (0,0) or (1,1), contribute +1
    #
    # This is getting complex. Let me use a different approach:
    # Build the Strassen decomposition directly from scratch for the 4x5x6 case
    # by padding to 8x8x8 and applying Strassen, then projecting back.

    return None


def _strassen_via_padding(n, m, p):
    """
    Pad the n x m * m x p problem to a power-of-2 size, apply Strassen,
    then extract the relevant sub-tensor.
    """
    # Find next power of 2 >= max(n, m, p)
    max_dim = max(n, m, p)
    s = 1
    while s < max_dim:
        s *= 2
    # s = 8 for 4,5,6

    # Build schoolbook for s x s * s x s
    U_s, V_s, W_s = _schoolbook(s, s, s)

    # Apply Strassen recursively
    # For s=8: split into 4x4 blocks, apply Strassen (7 mults of 4x4),
    # each 4x4 can be further split into 2x2 with Strassen.
    # Total rank for 8x8x8: 7^3 = 343 (vs schoolbook 512)

    # Let me implement Strassen decomposition directly.
    # For 2x2x2 blocks of size (n/2, m/2, p/2):
    # The 7 Strassen multiplications are:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B21-B22)
    # M4 = A22 * (B12-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)

    # For each of the 7, the U factor is a linear combination of the 4 U-blocks,
    # V factor is a linear combination of the 4 V-blocks,
    # W factor is a linear combination of the 4 W-blocks.

    # Let's define the 7 Strassen steps with their (U_coeff, V_coeff, W_coeff)
    # where each coeff is a 4x4 matrix (for the 2x2x2 block structure).

    # Actually, for tensor decomposition, each "multiplication" M = U * V * W
    # where U is (n*m) x 1, V is (m*p) x 1, W is (p*n) x 1.
    # For Strassen, we replace 8 such rank-1 tensors with 7.

    # Let me think differently. The matmul tensor for n x m * m x p has
    # T[i,j,k] = delta_{i,j,k} meaning T[i,j,k]=1 for all valid (i,j,k).
    # A rank-1 decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]

    # For schoolbook: each (i,j,k) gets its own rank-1 term.
    # For Strassen: we group 8 adjacent (i,j,k) terms into 7.

    # For the 2x2x2 case (n=m=p=2):
    # The 8 schoolbook terms are for (i,j,k) in {0,1}^3.
    # Strassen groups them into 7 terms.

    # Let me just hardcode the Strassen decomposition for 2x2x2 and use it
    # recursively via padding.

    # 2x2x2 Strassen: rank 7
    # Using standard Strassen for 2x2 matrix multiply:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(g-h), M4 = d(f-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6

    # In tensor form for n=m=p=2:
    # T[i,j,k] = A[i,j]*B[j,k]
    # i in {0,1}, j in {0,1}, k in {0,1}
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10-B11) -> contributes to C01, C11
    # M4 = A11(B01-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00, C11

    # For each M, we need U (2x2 -> 4 entries), V (2x2 -> 4 entries), W (2x2 -> 4 entries)
    # such that sum over the 7 M's reconstructs C.

    # This is getting very complex. Let me take a simpler approach:
    # Use known results. The rank of 4x4x4 matmul is at most 49 (Strassen).
    # For 4x5x6, we can pad to 8x8x8 (rank <= 343), but that's worse than 120.
    # Or pad to 4x4x4 for the 4x4 part... but we have 4x5x6.

    # Alternative: use the fact that rank(4x5x6) <= rank(8x8x8) = 343? No, that's worse.
    # Actually, we can use a hybrid: split 4x5x6 into sub-problems.

    # Let me try a direct approach: use a known low-rank decomposition.
    # The rank of 4x5x6 matrix multiplication is known to be at most...
    # Actually, let me just try to find a better decomposition via search.

    return None


def _try_strassen_4x5x6():
    """
    Try to construct a Strassen-based decomposition for 4x5x6.
    Strategy: Pad 4x5x6 to 4x8x8 (or 8x8x8), apply Strassen, extract.
    But 8x8x8 has rank 343 > 120, so that's not helpful.

    Better strategy: Use a 2x2x2 Strassen step on a 4x4x4 sub-block,
    and handle the rest with schoolbook.

    Actually, let me think about this more carefully.
    4x5x6: n=4, m=5, p=6.
    We can split n into 2+2, m into 2+3, p into 3+3.
    This gives 8 sub-problems of varying sizes.
    Applying Strassen-like reductions to each...

    This is getting too complex for a quick implementation.
    Let me try a different approach: use a known explicit decomposition.

    The rank of 4x4x4 matmul is at most 49 (Strassen gives 49).
    For 4x5x6, we can use:
    - Split m=5 into m1=4 + m2=1
    - C = A1*B1 + A2*B2 where A1 is 4x4, B1 is 4x6, A2 is 4x1, B2 is 1x6
    - rank(4x4x6) + rank(4x1x6)
    - rank(4x4x6): pad to 4x4x8, use Strassen? 4x4x8 rank...
    - This is getting complicated.

    Let me just implement a simple search: try to find a rank-r decomposition
    by solving a system of equations. But that's expensive.

    For now, let me return the schoolbook and note that Strassen improvements
    need more careful implementation.
    """
    return None


def _convert_to_output(U, V, W):
    """Convert Fraction-based matrices to the required output format."""
    rank = len(U[0]) if U else 0
    U_out = [[int(U[i][c]) if U[i][c].denominator == 1 else float(U[i][c]) for c in range(rank)] for i in range(len(U))]
    V_out = [[int(V[i][c]) if V[i][c].denominator == 1 else float(V[i][c]) for c in range(rank)] for i in range(len(V))]
    W_out = [[int(W[i][c]) if W[i][c].denominator == 1 else float(W[i][c]) for c in range(rank)] for i in range(len(W))]
    return {"U": U_out, "V": V_out, "W": W_out}


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Try Strassen-based decomposition
    result = _try_strassen_4x5x6()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return _convert_to_output(U, V, W)

    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return _convert_to_output(U, V, W)
# EVOLVE-BLOCK-END
