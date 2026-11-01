"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(R):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _is_half_integer_array(arr, tol=1e-6):
    """Check if all entries are close to half-integers."""
    arr = np.array(arr, dtype=float)
    for val in arr.flatten():
        nearest = round(val * 2) / 2.0
        if abs(val - nearest) > tol:
            return False
    return True

def _round_to_half(arr):
    """Round array to nearest half-integers."""
    arr = np.array(arr, dtype=float)
    return np.round(arr * 2) / 2.0

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-(n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_with_fixed_UV(n, m, p, target_rank, deadline, seed=42):
    """Search: generate random sparse U, V with half-integer entries, solve for W, check."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for trial in range(2000):
        if time.time() > deadline:
            return None
        
        # Generate U with limited non-zeros per column
        U = np.zeros((nm, target_rank))
        for r in range(target_rank):
            nnz = rng.randint(1, min(4, nm) + 1)
            pos = rng.choice(nm, size=nnz, replace=False)
            for idx in pos:
                U[idx, r] = rng.choice(values)
        
        # Generate V with limited non-zeros per column
        V = np.zeros((mp, target_rank))
        for r in range(target_rank):
            nnz = rng.randint(1, min(4, mp) + 1)
            pos = rng.choice(mp, size=nnz, replace=False)
            for idx in pos:
                V[idx, r] = rng.choice(values)
        
        # Build linear system for W
        # For each (i,k): sum_r W[k*n+i, r] * A_ik[r] = 1
        # where A_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        A = np.zeros((n * p, target_rank))
        for i in range(n):
            for k in range(p):
                idx = k * n + i
                for r in range(target_rank):
                    A[idx, r] = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
        
        # Check: each row of A must have at least one non-zero
        if np.any(np.all(np.abs(A) < 1e-12, axis=1)):
            continue
        
        # Solve: for each row, find W row such that W_row · A_row = 1
        # Use the row with largest norm for direct solve, others via least squares
        # Actually, each equation is independent: W[idx, :] · A[idx, :] = 1
        
        W = np.zeros((pn, target_rank))
        valid = True
        for idx in range(n * p):
            a_row = A[idx, :]
            if np.linalg.norm(a_row) < 1e-12:
                valid = False
                break
            # Find a solution: minimize ||W_row|| subject to W_row · a_row = 1
            # Solution: W_row = a_row / (a_row · a_row)
            dot = np.dot(a_row, a_row)
            if abs(dot) < 1e-12:
                valid = False
                break
            W[idx, :] = a_row / dot
        
        if not valid:
            continue
        
        # Check if W is close to half-integers
        if _is_half_integer_array(W, tol=1e-4):
            W_r = _round_to_half(W)
            if _verify_decomposition(U, V, W_r, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_r.tolist()}
        
        # Also try: use a different solution - pick one free variable
        # For each row, set all but one to 0
        W2 = np.zeros((pn, target_rank))
        valid2 = True
        for idx in range(n * p):
            a_row = A[idx, :]
            nonzero = [r for r in range(target_rank) if abs(a_row[r]) > 1e-12]
            if not nonzero:
                valid2 = False
                break
            r0 = nonzero[rng.randint(len(nonzero))]
            W2[idx, r0] = 1.0 / a_row[r0]
        
        if valid2 and _is_half_integer_array(W2, tol=1e-4):
            W2_r = _round_to_half(W2)
            if _verify_decomposition(U, V, W2_r, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W2_r.tolist()}
    
    return None

def _search_structured_patterns(n, m, p, target_rank, deadline, seed=0):
    """Try structured patterns for U and V."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    values = [-1, 0, 1]
    
    # Pattern 1: U is block-diagonal (each i-block uses separate columns)
    for trial in range(500):
        if time.time() > deadline:
            return None
        
        # U: for row (i,j), only columns in range [i*target_rank//n, (i+1)*target_rank//n] can be nonzero
        U = np.zeros((nm, target_rank))
        for i in range(n):
            col_start = i * target_rank // n
            col_end = (i + 1) * target_rank // n
            for j in range(m):
                for r in range(col_start, col_end):
                    if rng.random() < 0.5:
                        U[i * m + j, r] = rng.choice(values)
        
        V = np.zeros((mp, target_rank))
        for j in range(m):
            for k in range(p):
                for r in range(target_rank):
                    if rng.random() < 0.3:
                        V[j * p + k, r] = rng.choice(values)
        
        # Build A
        A = np.zeros((n * p, target_rank))
        for i in range(n):
            for k in range(p):
                idx = k * n + i
                for r in range(target_rank):
                    A[idx, r] = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
        
        if np.any(np.all(np.abs(A) < 1e-12, axis=1)):
            continue
        
        W = np.zeros((pn, target_rank))
        valid = True
        for idx in range(n * p):
            a_row = A[idx, :]
            dot = np.dot(a_row, a_row)
            if abs(dot) < 1e-12:
                valid = False
                break
            W[idx, :] = a_row / dot
        
        if valid and _is_half_integer_array(W, tol=1e-4):
            W_r = _round_to_half(W)
            if _verify_decomposition(U, V, W_r, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_r.tolist()}
    
    return None

def _try_integer_sparse_search(n, m, p, target_rank, deadline, seed=0):
    """Search with integer-only entries for U and V."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(3000):
        if time.time() > deadline:
            return None
        
        # U with integer entries {-1, 0, 1}, sparse
        U = np.zeros((nm, target_rank))
        for r in range(target_rank):
            nnz = rng.randint(1, min(3, nm) + 1)
            pos = rng.choice(nm, size=nnz, replace=False)
            for idx in pos:
                U[idx, r] = rng.choice([-1, 1])
        
        V = np.zeros((mp, target_rank))
        for r in range(target_rank):
            nnz = rng.randint(1, min(3, mp) + 1)
            pos = rng.choice(mp, size=nnz, replace=False)
            for idx in pos:
                V[idx, r] = rng.choice([-1, 1])
        
        # Build A
        A = np.zeros((n * p, target_rank))
        for i in range(n):
            for k in range(p):
                idx = k * n + i
                for r in range(target_rank):
                    A[idx, r] = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
        
        if np.any(np.all(np.abs(A) < 1e-12, axis=1)):
            continue
        
        # Solve for W
        W = np.zeros((pn, target_rank))
        valid = True
        for idx in range(n * p):
            a_row = A[idx, :]
            dot = np.dot(a_row, a_row)
            if abs(dot) < 1e-12:
                valid = False
                break
            W[idx, :] = a_row / dot
        
        if not valid:
            continue
        
        # Check half-integer
        if _is_half_integer_array(W, tol=1e-4):
            W_r = _round_to_half(W)
            if _verify_decomposition(U, V, W_r, n, m, p, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_r.tolist()}
    
    return None

def _direct_construction_attempt(n, m, p, target_rank):
    """Try a direct structured construction for specific ranks."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try: each (i,j) pair in U gets a dedicated set of columns
    # For rank R, try to assign columns to (i,j) pairs
    if target_rank >= n * m:
        # Each (i,j) gets at least one column
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Assign first n*m columns, one per (i,j)
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                # V[j*p+k, col] needs to be set for all k
                for k in range(p):
                    V[j * p + k, col] = 1
                # W[k*n+i, col] needs to be set for all k
                for k in range(p):
                    W[k * n + i, col] = 1
                col += 1
        
        if col <= target_rank:
            # Fill remaining columns with zeros (they don't contribute)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First try direct construction for higher ranks (quick check)
    for rank in range(n * m * p, 17, -1):
        if time.time() > deadline - 5:
            break
        result = _direct_construction_attempt(n, m, p, rank)
        if result is not None:
            return result
    
    # Try searching for lower ranks
    for target_rank in range(18, n * m * p + 1):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Split time among different search strategies
        time_per_strategy = remaining / 3.0
        d1 = time.time() + time_per_strategy * 0.8
        d2 = time.time() + time_per_strategy * 1.6
        d3 = time.time() + time_per_strategy * 2.4
        
        if d1 < deadline:
            result = _search_with_fixed_UV(n, m, p, target_rank, d1, seed=target_rank * 100)
            if result is not None:
                return result
        
        if d2 < deadline:
            result = _search_structured_patterns(n, m, p, target_rank, d2, seed=target_rank * 200)
            if result is not None:
                return result
        
        if d3 < deadline:
            result = _try_integer_sparse_search(n, m, p, target_rank, d3, seed=target_rank * 300)
            if result is not None:
                return result
    
    # Fallback: trivial rank-24 decomposition
    return _trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
