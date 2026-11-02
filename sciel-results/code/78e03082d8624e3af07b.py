"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Uses structured random search with fast tensor evaluation, exploiting the
algebraic structure of matmul. Falls back to schoolbook rank-36 if no
better solution is found within the time budget.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# Time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s deadline, 10s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
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


def _evaluate(U, V, W, n, m, p):
    """Compute the tensor from factors and return max error from 1.0.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal 1.0 for all valid i,j,k.
    """
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _random_half_int(nrows, rank, scale=2):
    """Random half-integer matrix."""
    mat = []
    for _ in range(nrows):
        row = []
        for _ in range(rank):
            val = random.randint(-scale, scale) * 0.5
            row.append(val)
        mat.append(row)
    return mat


def _try_structured_rank(n, m, p, rank, deadline, num_restarts=50):
    """Try to find a rank-`rank` decomposition.
    
    Strategy: random initialization with structural constraints,
    then local search on the half-integer lattice.
    """
    best_err = float('inf')
    best_factors = None
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random half-integers, but ensure some structure
        U = _random_half_int(n * m, rank, scale=1)
        V = _random_half_int(m * p, rank, scale=1)
        W = _random_half_int(p * n, rank, scale=1)
        
        # Ensure each rank dimension has at least one non-zero in each factor
        for r in range(rank):
            U[random.randint(0, n * m - 1)][r] = random.choice([0.5, 1.0, -0.5, -1.0])
            V[random.randint(0, m * p - 1)][r] = random.choice([0.5, 1.0, -0.5, -1.0])
            W[random.randint(0, p * n - 1)][r] = random.choice([0.5, 1.0, -0.5, -1.0])
        
        cur_err = _evaluate(U, V, W, n, m, p)
        
        if cur_err < best_err:
            best_err = cur_err
            best_factors = (U[:], V[:], W[:])
        
        if cur_err < 1e-9:
            return U, V, W
        
        # Local search: coordinate descent on half-integer lattice
        max_steps = 2000
        step = 0
        while step < max_steps and time.time() < deadline:
            step += 1
            
            # Pick random factor, row, col, and step direction
            factor = random.randint(0, 2)
            if factor == 0:
                mat, nrows = U, n * m
            elif factor == 1:
                mat, nrows = V, m * p
            else:
                mat, nrows = W, p * n
            
            row = random.randint(0, nrows - 1)
            col = random.randint(0, rank - 1)
            delta = random.choice([-0.5, 0.5, -1.0, 1.0])
            
            old_val = mat[row][col]
            mat[row][col] = old_val + delta
            
            new_err = _evaluate(U, V, W, n, m, p)
            
            if new_err < cur_err:
                cur_err = new_err
                if cur_err < best_err:
                    best_err = cur_err
                    best_factors = (U[:], V[:], W[:])
                if cur_err < 1e-9:
                    return U, V, W
            else:
                mat[row][col] = old_val
    
    if best_factors is not None and best_err < 1e-9:
        return best_factors
    return None


def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from the theoretical lower bound."""
    # Theoretical lower bounds:
    # - max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # - Strassen-type bounds suggest possibly lower
    # Try ranks from 12 upward
    
    for rank in range(12, 36):
        if time.time() > deadline:
            break
        
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=30)
        if result is not None:
            return result
    
    return None


def _try_pattern_based(n, m, p, deadline):
    """Try specific structured patterns that might yield low rank.
    
    For matmul, there are known algebraic structures. Try:
    1. Kronecker product patterns
    2. Sparse patterns with specific symmetries
    """
    if time.time() > deadline:
        return None
    
    # Try a few specific structured patterns
    patterns = []
    
    # Pattern 1: Based on the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    # Try to factor the summation index j
    
    # For now, just try a few random structured initializations
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        rank = 15 + attempt  # Try various ranks
        if rank >= 36:
            break
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Create a structured pattern: each rank dimension handles a subset
        # of the (i,j,k) tuples
        used = set()
        for r in range(rank):
            # Assign this rank dimension to handle certain (i,j,k) combinations
            # Simple approach: random assignment with some structure
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            # Set values to try to cover this (i,j,k)
            U[i * m + j][r] = 1.0
            V[j * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
        
        err = _evaluate(U, V, W, n, m, p)
        if err < 1e-9:
            return U, V, W
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.
    
    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)
    
    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS
    
    # Try structured search for low-rank solutions
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try pattern-based search
    result = _try_pattern_based(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-36
    return _schoolbook()
