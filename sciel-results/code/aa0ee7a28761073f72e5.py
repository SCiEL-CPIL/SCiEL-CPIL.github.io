"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition: rank = n*m*p."""
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid.
    
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _check_residual(U, V, W, n, m, p):
    """Compute max absolute residual."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                max_res = max(max_res, abs(s - 1.0))
    return max_res


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline, seed=0):
    """Try to reduce rank from current to target_rank by zeroing columns and solving.
    
    Strategy: Pick target_rank columns to keep, zero out the rest, then try to 
    adjust the kept columns to satisfy all equations. This is a nonlinear system,
    but for small reductions it might be solvable via local search.
    """
    import random
    rng = random.Random(seed)
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    # Try multiple random selections of which columns to keep
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Select which columns to keep
        keep_idx = sorted(rng.sample(range(current_rank), target_rank))
        
        U_trial = U[:, keep_idx].copy()
        V_trial = V[:, keep_idx].copy()
        W_trial = W[:, keep_idx].copy()
        
        # Local search: gradient-free perturbation
        best_res = _check_residual(U_trial, V_trial, W_trial, n, m, p)
        
        for step in range(200):
            if time.time() > deadline - 5.0:
                return None
            
            # Random perturbation
            scale = 0.1 / (1 + step * 0.01)
            U_pert = U_trial + rng.gauss(0, scale, U_trial.shape)
            V_pert = V_trial + rng.gauss(0, scale, V_trial.shape)
            W_pert = W_trial + rng.gauss(0, scale, W_trial.shape)
            
            res = _check_residual(U_pert, V_pert, W_pert, n, m, p)
            if res < best_res:
                best_res = res
                U_trial = U_pert
                V_trial = V_pert
                W_trial = W_pert
                if best_res < 1e-10:
                    break
        
        if best_res < 1e-10:
            # Convert back to list format with rational-looking entries
            U_list = [[round(float(x), 6) for x in row] for row in U_trial.T]
            V_list = [[round(float(x), 6) for x in row] for row in V_trial.T]
            W_list = [[round(float(x), 6) for x in row] for row in W_trial.T]
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_block_structure(n, m, p, deadline):
    """Try to exploit block structure.
    
    For 3x4 * 4x5, we can think of the 4x4 middle as having some structure.
    Try to find a decomposition where some terms are shared across blocks.
    """
    # One approach: try to find a rank-R decomposition by solving the system
    # where we parameterize U, V, W with fewer free parameters.
    
    # For now, try a direct approach: for each target rank from lower bound up,
    # try random initialization and local search.
    
    lower_bound = max(n * m, n * p, m * p)  # = max(12, 15, 20) = 20
    upper_bound = n * m * p  # 60
    
    # We'll search from a reasonable starting point
    # Known: for 2x2x2, rank 7 (vs 8 schoolbook)
    # For larger matrices, improvements are harder to find
    
    # Try rank 59, 58, ... down to lower_bound + some margin
    for target_rank in range(upper_bound - 1, max(lower_bound, upper_bound - 10), -1):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization for U, V, W
        np.random.seed(target_rank)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Local search
        best_res = _check_residual(U, V, W, n, m, p)
        
        for step in range(500):
            if time.time() > deadline - 10.0:
                break
            
            scale = 0.5 / (1 + step * 0.005)
            U_pert = U + np.random.randn(*U.shape) * scale
            V_pert = V + np.random.randn(*V.shape) * scale
            W_pert = W + np.random.randn(*W.shape) * scale
            
            res = _check_residual(U_pert, V_pert, W_pert, n, m, p)
            if res < best_res:
                best_res = res
                U = U_pert
                V = V_pert
                W = W_pert
                if best_res < 1e-12:
                    break
        
        if best_res < 1e-10:
            U_list = [[round(float(x), 6) for x in row] for row in U.T]
            V_list = [[round(float(x), 6) for x in row] for row in V.T]
            W_list = [[round(float(x), 6) for x in row] for row in W.T]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find a better solution by reducing rank
    # First, try the block structure approach
    result = _try_block_structure(n, m, p, deadline)
    if result is not None:
        best = result
    
    # Verify the result
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
    
    return best
