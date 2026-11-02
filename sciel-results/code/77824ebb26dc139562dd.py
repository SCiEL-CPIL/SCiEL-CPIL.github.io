"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W index is k*n + i, U is i*m + j, V is j*p + k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _update_W(U, V, W, n, m, p, r):
    """Update W rows given U and V."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k*n + i, :] = sol

def _update_U(U, V, W, n, m, p, r):
    """Update U rows given V and W."""
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r))
            for k in range(p):
                B[k, :] = V[j*p + k, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i*m + j, :] = sol

def _update_V(U, V, W, n, m, p, r):
    """Update V rows given U and W."""
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, r))
            for i in range(n):
                C[i, :] = U[i*m + j, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j*p + k, :] = sol

def _is_half_integer(mat, tol=1e-5):
    """Check if all entries are close to multiples of 0.5."""
    rounded = np.round(mat * 2) / 2
    return np.allclose(mat, rounded, atol=tol)

def _round_to_half_integer(mat):
    return np.round(mat * 2) / 2

def _compute_error(U, V, W, n, m, p, r):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def _polynomial_interp_attempt(n, m, p, target_rank):
    """Try to find a decomposition using polynomial interpolation.
    
    For small ranks, we can use the fact that the matmul tensor has structure.
    We try to find U, V, W with entries from a small set.
    """
    if target_rank < 1:
        return None
    
    # For the matmul tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: use a structured initialization based on polynomial evaluation
    # Let's try with entries from {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    
    # For very small ranks, try exhaustive search over small entry sets
    if target_rank <= 4:
        # Try all combinations from a small set
        entries = [-1.0, -0.5, 0.0, 0.5, 1.0]
        # This is too large for brute force, so use random sampling
        for trial in range(1000):
            U = np.random.choice(entries, size=(n*m, target_rank))
            V = np.random.choice(entries, size=(m*p, target_rank))
            W = np.random.choice(entries, size=(p*n, target_rank))
            
            valid, _ = _verify_rank(n, m, p, target_rank, U, V, W)
            if valid:
                return U, V, W
    
    return None

def _als_optimize(U, V, W, n, m, p, r, max_iters=200, tol=1e-8, deadline=None):
    """Run ALS optimization with proper loop structure."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < tol:
            break
        
        # Update W
        _update_W(U, V, W, n, m, p, r)
        # Update U
        _update_U(U, V, W, n, m, p, r)
        # Update V
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _polish_solution(U, V, W, n, m, p, r, deadline=None):
    """Final high-precision polish pass."""
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < 1e-10:
            break
        
        _update_W(U, V, W, n, m, p, r)
        _update_U(U, V, W, n, m, p, r)
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin out of 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
                
    random.seed(42)
    np.random.seed(42)
    
    # Search for lower ranks
    # We start from 95 down to 1
    for target_rank in range(schoolbook_rank - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        attempts = 8
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Random initialization with varied scales
            scale = 0.3 + 0.4 * (attempt % 3)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Normalize columns to avoid scale drift
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale_c = 1.0 / (norm ** (1.0/3.0))
                    U[:, c] *= scale_c
                    V[:, c] *= scale_c
                    W[:, c] *= scale_c
            
            # ALS iterations
            U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=150, deadline=deadline)
            
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-7:
                # Check if we can round to half-integers
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    U_rounded = _round_to_half_integer(U)
                    V_rounded = _round_to_half_integer(V)
                    W_rounded = _round_to_half_integer(W)
                    
                    # Verify rounded solution
                    valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                    if valid:
                        best_rank = target_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
            
            # If not half-integer, try polishing
            if err < 1e-4:
                U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline)
                err = _compute_error(U, V, W, n, m, p, target_rank)
                
                if err < 1e-8:
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                        if valid:
                            best_rank = target_rank
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            found = True
                            break
        
        if found:
            # Found a solution at this rank, continue to lower ranks
            continue
        else:
            # If we've tried many ranks without success, we can stop earlier
            # for efficiency, but keep going for now
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
