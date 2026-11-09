"""Tensor decomposition search for 3x5 * 5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

import time
import random
import itertools

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] == 1 for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using list operations."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            urow = U[ui]
            for k in range(p):
                vrow = V[j * p + k]
                wrow = W[k * n + i]
                s = 0
                for c in range(r):
                    s += urow[c] * vrow[c] * wrow[c]
                if s != 1:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Return the trivial (schoolbook) rank-n*m*p decomposition."""
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


def _total_error(U, V, W, n, m, p):
    """Compute sum of squared errors over all (i,j,k)."""
    r = len(U[0])
    err = 0
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            urow = U[ui]
            for k in range(p):
                vrow = V[j * p + k]
                wrow = W[k * n + i]
                s = 0
                for c in range(r):
                    s += urow[c] * vrow[c] * wrow[c]
                err += (s - 1) * (s - 1)
    return err


def _local_search(U, V, W, n, m, p, time_budget=5.0, max_steps=10000):
    """Coordinate descent local search. Returns True if error < 1e-6."""
    r = len(U[0])
    start_time = time.time()
    current_err = _total_error(U, V, W, n, m, p)
    if current_err < 1e-6:
        return True
    
    # Precompute row indices for speed
    all_U_rows = list(range(n * m))
    all_V_rows = list(range(m * p))
    all_W_rows = list(range(p * n))
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Pick random matrix, row, column
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            row = random.choice(all_U_rows)
            col = random.randint(0, r - 1)
            target = U
            nrows = n * m
        elif mat_idx == 1:
            row = random.choice(all_V_rows)
            col = random.randint(0, r - 1)
            target = V
            nrows = m * p
        else:
            row = random.choice(all_W_rows)
            col = random.randint(0, r - 1)
            target = W
            nrows = p * n
        
        old_val = target[row][col]
        # Try small perturbations
        for delta in [0.5, -0.5, 0.25, -0.25, 1.0, -1.0]:
            new_val = old_val + delta
            target[row][col] = new_val
            new_err = _total_error(U, V, W, n, m, p)
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-6:
                    return True
                break
        else:
            target[row][col] = old_val
    
    return current_err < 1e-6


def _try_rank(r, n, m, p, deadline, max_attempts=8, time_per_attempt=4.0):
    """Try to find a valid rank-r decomposition."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random integer initialization in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        # Quick check
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Local search
        if _local_search(U, V, W, n, m, p, time_budget=time_per_attempt, max_steps=5000):
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank_continuous(r, n, m, p, deadline, max_attempts=5, time_per_attempt=4.0):
    """Try with continuous values and snap to half-integers."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Random continuous initialization
        U = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(n * m)]
        V = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(m * p)]
        W = [[random.uniform(-2, 2) for _ in range(r)] for _ in range(p * n)]
        
        if _local_search(U, V, W, n, m, p, time_budget=time_per_attempt, max_steps=5000):
            # Snap to half-integers
            def _snap(mat):
                return [[round(v * 2) / 2 for v in row] for row in mat]
            U2 = _snap(U)
            V2 = _snap(V)
            W2 = _snap(W)
            if _verify(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None


def _structured_search(n, m, p, deadline):
    """
    Structured search: try specific rank values with multiple strategies.
    For 3x5x6, try ranks from 24 to 60.
    """
    # Candidate ranks to try, in order of preference
    candidate_ranks = list(range(24, 61))
    
    # Try each rank
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Strategy 1: Integer-valued search
        result = _try_rank(r, n, m, p, deadline, max_attempts=6, time_per_attempt=3.0)
        if result:
            return result
        
        # Strategy 2: Continuous search with snapping
        if time.time() > deadline:
            break
        result = _try_rank_continuous(r, n, m, p, deadline, max_attempts=4, time_per_attempt=3.0)
        if result:
            return result
    
    return None


def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Fallback: schoolbook
    fallback = _make_schoolbook(n, m, p)
    
    # Try structured search for better rank
    result = _structured_search(n, m, p, deadline)
    if result:
        return result
    
    # If time remains, try a few more targeted attempts
    if time.time() < deadline - 10:
        # Try rank 30 with more attempts
        result = _try_rank(30, n, m, p, deadline, max_attempts=10, time_per_attempt=5.0)
        if result:
            return result
        
        result = _try_rank_continuous(30, n, m, p, deadline, max_attempts=8, time_per_attempt=5.0)
        if result:
            return result
    
    # Return schoolbook as fallback
    return fallback

# EVOLVE-BLOCK-END
