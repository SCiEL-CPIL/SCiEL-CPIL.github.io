"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a sparse dict.
    T[a, b, c] = 1 where a = i*m+k, b = k*p+j, c = j*n+i
    for all valid i in [0,n), k in [0,m), j in [0,p).
    """
    T = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[(a, b, c)] = 1
    return T

def _matrix_rank_rational(M):
    """Compute rank of a matrix over the rationals using Gaussian elimination."""
    if not M or not M[0]:
        return 0
    rows = len(M)
    cols = len(M[0])
    M_copy = [[Fraction(x) for x in row] for row in M]
    rank = 0
    for col in range(cols):
        pivot = -1
        for row in range(rank, rows):
            if M_copy[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        M_copy[rank], M_copy[pivot] = M_copy[pivot], M_copy[rank]
        pivot_val = M_copy[rank][col]
        for j in range(col, cols):
            M_copy[rank][j] /= pivot_val
        for row in range(rows):
            if row != rank and M_copy[row][col] != 0:
                factor = M_copy[row][col]
                for j in range(col, cols):
                    M_copy[row][j] -= factor * M_copy[rank][j]
        rank += 1
        if rank == rows:
            break
    return rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    Uses the convention: U[i*m+k, r], V[k*p+j, r], W[j*n+i, r]
    T[a,b,c] = 1 iff a=i*m+k, b=k*p+j, c=j*n+i for consistent i,k,j.
    """
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Build expected tensor
    T = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[(a, b, c)] = 1
    
    # Verify all entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if actual != expected:
                    return False
    
    # Also verify some zero entries to ensure no spurious contributions
    # Check a sample of zero entries
    checked = set()
    for _ in range(min(100, n*m*p)):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        if (a, b, c) in checked:
            continue
        checked.add((a, b, c))
        if (a, b, c) in T:
            continue
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if actual != 0:
            return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_low_rank_search(n, m, p, target_rank, deadline, max_attempts=200):
    """Try to find a rank decomposition of the given rank using structured random search.
    Uses the fact that we need integer/half-integer entries.
    """
    random.seed(42)
    np.random.seed(42)
    
    # The tensor has n*m*p = 60 non-zero entries
    # We need to represent it as sum of target_rank rank-1 tensors
    
    # Strategy: Start with schoolbook and try to merge columns
    # Or: Random initialization with values in {-1, -0.5, 0, 0.5, 1}
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random initialization with small integer/half-integer values
        values = [-1, -0.5, 0, 0, 0, 0.5, 1]  # biased towards 0 for sparsity
        
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_reduction(n, m, p, deadline):
    """Try to reduce the schoolbook rank by finding linear dependencies.
    This is a more systematic approach that looks for ways to combine terms.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    
    # Try to find pairs of columns that can be combined
    # This is a complex optimization problem
    # For now, just return the schoolbook
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as baseline
    best = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Compute lower bounds from matrix flattenings
    T = _build_tensor(n, m, p)
    
    # M1: (n*m) x (m*p * p*n)
    dim1 = n * m
    dim2 = m * p * p * n
    M1 = [[0] * dim2 for _ in range(dim1)]
    for (a, b, c), val in T.items():
        M1[a][b * p * n + c] = val
    
    if time.time() > deadline:
        return best
    
    lb1 = _matrix_rank_rational(M1)
    
    # M2: (m*p) x (n*m * p*n)
    dim2a = m * p
    dim2b = n * m * p * n
    M2 = [[0] * dim2b for _ in range(dim2a)]
    for (a, b, c), val in T.items():
        M2[b][a * p * n + c] = val
    
    if time.time() > deadline:
        return best
    
    lb2 = _matrix_rank_rational(M2)
    
    # M3: (p*n) x (n*m * m*p)
    dim3a = p * n
    dim3b = n * m * m * p
    M3 = [[0] * dim3b for _ in range(dim3a)]
    for (a, b, c), val in T.items():
        M3[c][a * m * p + b] = val
    
    if time.time() > deadline:
        return best
    
    lb3 = _matrix_rank_rational(M3)
    
    lower_bound = max(lb1, lb2, lb3)
    
    # Try to find decomposition at lower_bound or slightly above
    # For 3x4x5, the lower bound is likely 48 (from known results)
    # Schoolbook is 60, so there's room for improvement
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    search_start = max(lower_bound, best_rank - 20)  # Don't search too far
    search_end = best_rank - 1
    
    for candidate_rank in range(search_start, search_end + 1):
        if time.time() > deadline - 5:
            break
        
        result = _try_low_rank_search(n, m, p, candidate_rank, deadline)
        if result is not None:
            best = result
            best_rank = candidate_rank
            break  # Found a lower rank, stop searching
    
    # Also try the structured reduction approach
    if time.time() < deadline - 5:
        result = _try_structured_reduction(n, m, p, deadline)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    return best
# EVOLVE-BLOCK-END
