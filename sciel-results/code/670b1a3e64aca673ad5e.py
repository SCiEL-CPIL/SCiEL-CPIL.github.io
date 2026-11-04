"""Task program; the EVOLVE-BLOCK contains the evolved candidate."""
import math
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# --- Time-budget guard (wall-clock 150 s hard limit; keep a safety margin) ---
_START_TIME = time.time()
WALL_CLOCK_LIMIT = 150.0          # seconds
SAFETY_MARGIN = 5.0               # return by this many seconds before the limit
DEADLINE = _START_TIME + (WALL_CLOCK_LIMIT - SAFETY_MARGIN)

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
def _now() -> float:
    return time.time()


def _exceeds_deadline() -> bool:
    return _now() > DEADLINE


def _exact_verify(U, V, W) -> bool:
    """Exact rational verification of the decomposition convention."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _gauss_solve(A, b):
    """Solve Ax=b using Gaussian elimination with Fractions.
    Returns solution vector or None if inconsistent."""
    n = len(A)
    m = len(A[0])
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(m)] + [Fraction(b[i])] for i in range(n)]
    
    # Forward elimination
    for col in range(min(n, m)):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        aug[col], aug[pivot] = aug[pivot], aug[col]
        # Eliminate below
        for row in range(col + 1, n):
            if aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for j in range(col, m + 1):
                    aug[row][j] -= factor * aug[col][j]
    
    # Check for inconsistency
    for row in range(n):
        all_zero = all(aug[row][j] == 0 for j in range(m))
        if all_zero and aug[row][m] != 0:
            return None
    
    # Back substitution
    x = [Fraction(0)] * m
    for row in range(n - 1, -1, -1):
        # Find pivot column for this row
        pivot_col = -1
        for col in range(m):
            if aug[row][col] != 0:
                pivot_col = col
                break
        if pivot_col == -1:
            continue
        s = aug[row][m]
        for col in range(pivot_col + 1, m):
            s -= aug[row][col] * x[col]
        x[pivot_col] = s / aug[row][pivot_col]
    
    return x


def _constructive_decomp(rank: int, num_trials: int = 100) -> dict | None:
    """Construct a rank-r decomposition by fixing U,V and solving for W."""
    n, m, p = DIMENSIONS
    rng = random.Random(rank * 7919)
    
    for attempt in range(num_trials):
        if _exceeds_deadline():
            return None
        
        # Random U and V with small integer entries
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        
        # Ensure rows are not all zero
        if all(x == 0 for row in U for x in row):
            continue
        if all(x == 0 for row in V for x in row):
            continue
        
        # Solve for W row by row
        W = [[0] * rank for _ in range(p * n)]
        valid = True
        for k in range(p):
            for i in range(n):
                # Build m x rank matrix A where A[j,c] = U[i*m+j, c] * V[j*p+k, c]
                A = []
                for j in range(m):
                    row = []
                    for c in range(rank):
                        row.append(U[i * m + j][c] * V[j * p + k][c])
                    A.append(row)
                
                # Solve A * w = [1, 1, ..., 1]^T (m equations, rank unknowns)
                b = [1] * m
                w = _gauss_solve(A, b)
                if w is None:
                    valid = False
                    break
                W[k * n + i] = [int(x) for x in w]
            
            if not valid:
                break
        
        if valid:
            # Exact verification
            if _exact_verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None


def _build_schoolbook():
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


def construct() -> dict:
    n, m, p = DIMENSIONS
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p
    
    # Try constructive search for lower ranks
    for rank in range(best_rank - 1, 0, -1):
        if _exceeds_deadline():
            break
        decomp = _constructive_decomp(rank, num_trials=50)
        if decomp is not None:
            best = decomp
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
