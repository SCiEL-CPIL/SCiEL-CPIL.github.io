"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _solve_linear_exact(A, b):
    """Solve A x = b exactly using fraction arithmetic via Gaussian elimination.
    A is list of lists (m_eq x n_var), b is list of numbers (m_eq).
    Returns x (list of Fraction, length n_var) or None if inconsistent.
    If underdetermined, sets free variables to 0."""
    m_eq = len(A)
    if m_eq == 0:
        n_var = len(A[0]) if A else 0
        return [Fraction(0)] * n_var
    n_var = len(A[0])
    
    # Build augmented matrix with Fractions
    aug = []
    for i in range(m_eq):
        row = [Fraction(A[i][j]) for j in range(n_var)] + [Fraction(b[i])]
        aug.append(row)
    
    rank = 0
    pivot_cols = []
    for col in range(n_var):
        # Find pivot
        pivot_row = None
        for row in range(rank, m_eq):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        pivot_cols.append(col)
        # Eliminate all other rows
        pivot_val = aug[rank][col]
        for row in range(m_eq):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col] / pivot_val
                for j in range(col, n_var + 1):
                    aug[row][j] -= factor * aug[rank][j]
        rank += 1
        if rank == n_var:
            break
    
    # Check consistency
    for row in range(rank, m_eq):
        if aug[row][n_var] != 0:
            return None
    
    # Extract solution: free variables = 0
    x = [Fraction(0)] * n_var
    for i, pc in enumerate(pivot_cols):
        x[pc] = aug[i][n_var] / aug[i][pc]
    
    # Verify
    for i in range(m_eq):
        s = sum(A[i][j] * x[j] for j in range(n_var))
        if s != b[i]:
            return None
    
    return x


def _try_slice_method(r, U, V, n, m, p, deadline):
    """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r).
    For each (k, i), solve: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j.
    Returns W or None."""
    W = [[Fraction(0)] * r for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            # Build system: for each j, sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            A = []
            for j in range(m):
                row = []
                for c in range(r):
                    row.append(U[i * m + j][c] * V[j * p + k][c])
                A.append(row)
            b = [1] * m
            
            sol = _solve_linear_exact(A, b)
            if sol is None:
                return None
            W[k * n + i] = sol
        
        if time.time() > deadline:
            return None
    
    return W


def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is integer)."""
    x2 = x * 2
    return x2.denominator == 1


def _try_rank(r, deadline, n, m, p):
    """Try to find a rank-r decomposition. Returns dict or None."""
    import random
    rng = random.Random(42)
    
    max_attempts = 8000 if r <= 18 else 4000
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 2:
            return None
        
        # Generate random U: 9 x r with entries in {-1, 0, 1}
        U = []
        for row_idx in range(n * m):
            row = [0] * r
            num_nonzero = rng.choice([1, 2, 2, 3, 3, 4])
            if num_nonzero > r:
                num_nonzero = r
            cols = rng.sample(range(r), num_nonzero)
            for c in cols:
                row[c] = rng.choice([-1, 1])
            U.append(row)
        
        # Generate random V: 15 x r with entries in {-1, 0, 1}
        V = []
        for row_idx in range(m * p):
            row = [0] * r
            num_nonzero = rng.choice([1, 2, 2, 3, 3, 4])
            if num_nonzero > r:
                num_nonzero = r
            cols = rng.sample(range(r), num_nonzero)
            for c in cols:
                row[c] = rng.choice([-1, 1])
            V.append(row)
        
        # Try to solve for W
        W = _try_slice_method(r, U, V, n, m, p, deadline)
        if W is None:
            continue
        
        # Check if W has half-integer entries
        valid = True
        for row in W:
            for val in row:
                if not _is_half_integer(val):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to int or half-int representation
            U_out = [[x for x in row] for row in U]
            V_out = [[x for x in row] for row in V]
            W_out = []
            for row in W:
                wrow = []
                for val in row:
                    if val.denominator == 1:
                        wrow.append(int(val))
                    else:
                        wrow.append(float(val))
                W_out.append(wrow)
            
            # Verify
            if _verify(U_out, V_out, W_out, n, m, p):
                return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, m*p, n*m) = 15
    # Schoolbook: 45
    
    # Try rank 15 first with many attempts, then 16, 17, etc.
    # If no success by rank 25, fall back to schoolbook
    
    for r in range(15, 26):
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, deadline, n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
