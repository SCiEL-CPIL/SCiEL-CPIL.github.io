"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
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
    for val in arr.flatten():
        nearest = round(val * 2) / 2
        if abs(val - nearest) > tol:
            return False
    return True

def _round_to_half(arr):
    """Round array to nearest half-integers."""
    return np.round(arr * 2) / 2

def _search_rank(n, m, p, target_rank, deadline, seed=42):
    """Search for a rank-target_rank decomposition.
    
    Strategy: Enumerate sparse patterns for U and V, solve for W, check integrality.
    """
    rng = np.random.RandomState(seed)
    values = [-1, -0.5, 0, 0.5, 1]
    
    # We'll try: fix U to be sparse, fix V to be sparse, solve for W
    # U has shape (n*m, R) = (6, R)
    # V has shape (m*p, R) = (12, R)  
    # W has shape (p*n, R) = (8, R)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(500):
        if time.time() > deadline:
            return None
        
        # Generate sparse U: each column has at most 2 non-zero entries
        U = np.zeros((nm, target_rank))
        for r in range(target_rank):
            # Pick 1-2 random positions
            nnz = rng.randint(1, 3)
            pos = rng.choice(nm, size=nnz, replace=False)
            for idx in pos:
                U[idx, r] = rng.choice(values)
        
        # Generate sparse V: each column has at most 2 non-zero entries
        V = np.zeros((mp, target_rank))
        for r in range(target_rank):
            nnz = rng.randint(1, 3)
            pos = rng.choice(mp, size=nnz, replace=False)
            for idx in pos:
                V[idx, r] = rng.choice(values)
        
        # For each (i,k), we need: sum_r W[k*n+i, r] * A_ik_r = 1
        # where A_ik_r = sum_j U[i*m+j, r] * V[j*p+k, r]
        # This gives us n*p = 8 independent equations, each with R unknowns.
        # We can solve each equation by picking R-1 W values freely and solving for the last.
        
        W = np.zeros((pn, target_rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Compute A_ik_r for r = 0..R-1
                A = np.zeros(target_rank)
                for r in range(target_rank):
                    A[r] = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
                
                # Need: sum_r W[k*n+i, r] * A[r] = 1
                # If all A[r] are zero, this is impossible
                if np.allclose(A, 0, atol=1e-10):
                    valid = False
                    break
                
                # Pick a random solution: set all but one W to 0, solve for the last
                # Or use a structured approach: set W[k*n+i, r] = A[r] / sum(A^2) for one r
                # Actually, simplest: pick r0 where A[r0] != 0, set W[k*n+i, r0] = 1/A[r0], rest = 0
                nonzero_idx = [r for r in range(target_rank) if abs(A[r]) > 1e-10]
                if not nonzero_idx:
                    valid = False
                    break
                r0 = rng.choice(nonzero_idx)
                W[k * n + i, r0] = 1.0 / A[r0]
                # Set others to 0 (already 0)
        
        if not valid:
            continue
        
        # Check if W entries are half-integers
        if _is_half_integer_array(W, tol=1e-6):
            # Round and verify
            W_r = _round_to_half(W)
            if _verify_decomposition(U, V, W_r, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_r.tolist()}
        
        # Also try: use least squares to find W, then check if close to half-integers
        # Build the full system
        M = np.zeros((n * p, n * p * target_rank))
        b = np.ones(n * p)
        for i in range(n):
            for k in range(p):
                row = k * n + i
                for r in range(target_rank):
                    A_ik_r = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
                    M[row, row * target_rank + r] = A_ik_r
        
        try:
            w_sol, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
        except:
            continue
        
        W_ls = w_sol.reshape(pn, target_rank)
        
        # Check if close to half-integers
        if _is_half_integer_array(W_ls, tol=1e-4):
            W_r = _round_to_half(W_ls)
            if _verify_decomposition(U, V, W_r, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_r.tolist()}
    
    return None

def _search_rank_structured(n, m, p, target_rank, deadline, seed=123):
    """Structured search: try specific patterns known to work for matrix multiplication."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try: U has identity-like structure, V and W are optimized
    # For 2x3 x 3x4, try to use the fact that we can combine columns
    
    # Approach: Use 24 basis terms and try to find linear combinations
    # that reduce the rank while maintaining the decomposition.
    
    # Let's try a different approach: assume U has a specific form.
    # For example, U[i*m+j, r] = u_i * v_j for some factorization.
    
    # Or: try random U with values in {-1, 0, 1}, solve for V and W.
    
    for trial in range(300):
        if time.time() > deadline:
            return None
        
        # Random U with values in {-1, 0, 1}
        U = rng.randint(-1, 2, size=(nm, target_rank))
        
        # For each (i,k), compute B_ik_r = sum_j U[i*m+j, r] * V[j*p+k, r]
        # But V is unknown. Let's fix U and solve for V and W jointly.
        
        # Actually, let's fix U, then for each (i,j,k) we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is bilinear in V and W. Hard to solve directly.
        
        # Alternative: fix U and V, solve for W (linear).
        # So let's also fix V randomly.
        
        V = rng.randint(-1, 2, size=(mp, target_rank))
        
        # Solve for W: for each (i,k): sum_r W[k*n+i, r] * A_ik_r = 1
        # where A_ik_r = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        M = np.zeros((n * p, n * p * target_rank))
        b = np.ones(n * p)
        for i in range(n):
            for k in range(p):
                row = k * n + i
                for r in range(target_rank):
                    A_ik_r = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
                    M[row, row * target_rank + r] = A_ik_r
        
        try:
            w_sol, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
        except:
            continue
        
        W = w_sol.reshape(pn, target_rank)
        
        # Check if all of U, V, W are close to half-integers
        if _is_half_integer_array(U, tol=1e-6) and \
           _is_half_integer_array(V, tol=1e-6) and \
           _is_half_integer_array(W, tol=1e-4):
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-(n*m*p) decomposition."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition, starting from lower ranks
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        result = _search_rank(n, m, p, target_rank, deadline - 5, seed=42)
        if result is not None:
            return result
        
        result = _search_rank_structured(n, m, p, target_rank, deadline - 5, seed=123)
        if result is not None:
            return result
    
    # Fallback: trivial rank-24 decomposition
    return _trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
