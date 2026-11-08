"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

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
    random.seed(42)
    
    # For efficiency, we'll try a few structured patterns first, then random
    # Pattern 1: Identity-like U and V
    # U[i*m+j, c] = 1 if c == i*m+j (for c < n*m), else 0
    # V[j*p+k, c] = 1 if c == j*p+k (for c < m*p), else 0
    # This only works if n*m == m*p, which is not the case here (9 != 15)
    
    # Pattern 2: Try U and V with limited support
    # For each row of U, have exactly 1 or 2 non-zero entries
    # For each row of V, have exactly 1 or 2 non-zero entries
    
    attempts = 0
    max_attempts = 500
    
    while attempts < max_attempts:
        if time.time() > deadline:
            return None
        
        attempts += 1
        
        # Generate random U: 9 x r, each row has 1-3 ones
        U = []
        for row_idx in range(n * m):
            row = [0] * r
            num_ones = random.choice([1, 1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
            U.append(row)
        
        # Generate random V: 15 x r, each row has 1-3 ones
        V = []
        for row_idx in range(m * p):
            row = [0] * r
            num_ones = random.choice([1, 1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
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
            # Convert to integers/half-integers
            U_int = [[int(u) if u.denominator == 1 else int(u * 2) / 2 for u in row] for row in U]
            V_int = [[int(v) if v.denominator == 1 else int(v * 2) / 2 for v in row] for row in V]
            W_int = [[int(w) if w.denominator == 1 else int(w * 2) / 2 for w in row] for row in W]
            
            # Verify
            if _verify(U_int, V_int, W_int, n, m, p):
                return {"U": U_int, "V": V_int, "W": W_int}
    
    return None


def _try_groebner_approach(r, deadline, n, m, p):
    """Try to find a rank-r decomposition using a structured approach inspired by Gröbner bases.
    
    The idea: work over GF(2) first to find a sparse combinatorial pattern,
    then lift to integer coefficients.
    """
    import random
    random.seed(123)
    
    # For GF(2) approach, we look for U, V, W with entries in {0, 1}
    # such that the bilinear system is satisfied mod 2
    # Then lift to integers
    
    # Strategy: try sparse binary U and V, solve for W over Q, check if W is binary
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Generate sparse binary U: 9 x r
        U = []
        for row_idx in range(n * m):
            row = [0] * r
            num_ones = random.choice([1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
            U.append(row)
        
        # Generate sparse binary V: 15 x r
        V = []
        for row_idx in range(m * p):
            row = [0] * r
            num_ones = random.choice([1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
            V.append(row)
        
        # Solve for W
        W = _try_slice_method(r, U, V, n, m, p, deadline)
        if W is None:
            continue
        
        # Check if W is binary (entries in {0, 1})
        valid = True
        for row in W:
            for val in row:
                if val not in [Fraction(0), Fraction(1)]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify over integers
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up to schoolbook-1
    # Lower bound: max(n*p, m*p, n*m) = max(15, 15, 9) = 15
    # Schoolbook: 45
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try structured approaches for each rank
    for r in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try Gröbner-inspired approach
        result = _try_groebner_approach(r, deadline, n, m, p)
        if result is not None:
            best_result = result
            best_rank = r
            break
        
        # Try slice method with random sparse matrices
        result = _try_rank(r, deadline, n, m, p)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
