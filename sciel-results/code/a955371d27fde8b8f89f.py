"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Block decomposition: M_{3x3x4} = M_{3x3x3} ⊕ M_{3x3x1}
    # U has n*m = 9 rows, V has m*p = 12 rows, W has p*n = 12 rows
    # First 19 columns: M_{3x3x3} part (i in 0..2, j in 0..2, k in 0..2)
    # Last 9 columns:  M_{3x3x1} part (i in 0..2, j in 0..2, k=3 only)

    rank = 28
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 9 x 28
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 x 28
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 12 x 28

    # Rank-19 decomposition of M_{3x3x3} (integer coefficients)
    # Tuples: (i, j, k, ui, vj, wk)
    rank19_terms = [
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 1, 0),
        (0, 0, 2, 1, 1, 0),
        (0, 1, 0, 1, 1, 0),
        (0, 1, 1, 1, 1, 1),
        (0, 1, 2, 1, 1, 0),
        (0, 2, 0, 1, 1, 0),
        (0, 2, 1, 1, 1, 0),
        (0, 2, 2, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (1, 0, 1, 1, 1, 0),
        (1, 0, 2, 1, 1, 1),
        (1, 1, 0, 1, 1, 0),
        (1, 1, 1, 1, 1, 1),
        (1, 1, 2, 1, 1, 0),
        (1, 2, 0, 1, 1, 0),
        (1, 2, 1, 1, 1, 1),
        (1, 2, 2, 1, 1, 0),
        (2, 0, 0, 1, 1, 0),
        (2, 0, 1, 1, 1, 1),
        (2, 0, 2, 1, 1, 0),
        (2, 1, 0, 1, 1, 1),
        (2, 1, 1, 1, 1, 0),
        (2, 1, 2, 1, 1, 0),
        (2, 2, 0, 1, 1, 0),
        (2, 2, 1, 1, 1, 0),
        (2, 2, 2, 1, 1, 1),
    ]

    # Verify: for each (i,j,k) in {0,1,2}^3, sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c] should be 1
    # For the above, all terms have u=v=w=1, so contribution = number of terms with that (i,j,k).
    # That's just schoolbook (rank 27). Need actual rank-19 with cancellations.
    # Let me use the known rank-19 scheme.

    # Actually, I cannot reliably recall the exact rank-19 coefficients from memory.
    # Let me use a safe approach: rank 36 (schoolbook) to ensure validity.
    # But I want to try rank 28.

    # Fallback: use schoolbook for the 3x3x3 block too (rank 27+9=36)
    # This is safe but not an improvement.

    # Alternative: use rank-19 if I can verify it. Since I can't verify, let me use schoolbook.
    rank = n * m * p  # 36
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
