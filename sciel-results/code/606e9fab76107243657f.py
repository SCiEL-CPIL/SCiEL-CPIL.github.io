"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

We seek a rank-minimal exact decomposition of the multiplication tensor.
The multiplication tensor T is of size (n*m) x (m*p) x (p*n) with the
convention that the output index is k*n + i (i.e., row i of result,
where result = A * B, A is n x m, B is m x p).

The tensor entry T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k,
and 0 otherwise.

The rank of this tensor is at most n*m*p (schoolbook). For 2x3 * 3x10,
schoolbook rank is 60.

Key insight: The multiplication tensor for (n,m,p) can sometimes be
decomposed at lower rank than n*m*p. For instance, if we can find a
set of rank-1 tensors that sum to the multiplication tensor.

We'll implement a search that tries to find such a decomposition.
"""

import numpy as np
import random
import time
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Compute the contracted tensor from U, V, W
    # T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can do this efficiently:
    # M1[a, b] = sum_r U[a,r] * V[b,r]  (shape: n*m x m*p)
    # T_reconstructed[a, b, c] = sum_r M1[a,b] ... wait, W depends on r too.
    # T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # This might be slow for large rank, but rank is <= 60.
    # Dimensions: 6 x 30 x 20. Total elements 3600.
    # Einsum is fast.
    T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Check difference
    if not np.allclose(T_reconstructed, T_target, atol=1e-9):
        return False
        
    return True

def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _search_rank(n, m, p, target_rank, time_limit=10.0):
    """
    Try to find a decomposition of rank target_rank using optimization.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def objective(x):
        # x is flattened U, V, W
        size_u = dim_u * target_rank
        size_v = dim_v * target_rank
        size_w = dim_w * target_rank
        
        U_flat = x[:size_u]
        V_flat = x[size_u:size_u + size_v]
        W_flat = x[size_u + size_v:size_u + size_v + size_w]
        
        U = U_flat.reshape(dim_u, target_rank)
        V = V_flat.reshape(dim_v, target_rank)
        W = W_flat.reshape(dim_w, target_rank)
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_rec - T_target
        return np.sum(diff * diff)

    # Initialize with schoolbook (padded/truncated)
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    U_s = np.array(U_school)
    V_s = np.array(V_school)
    W_s = np.array(W_school)
    
    # If target_rank < 60, we need a different init.
    # Random init might get stuck in local minima.
    # Let's try a few random inits and one schoolbook-based init (if possible).
    
    start_time = time.time()
    best_solution = None
    best_error = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            # Try to use schoolbook structure if target_rank >= 60, else random
            if target_rank >= 60:
                U_init = U_s[:, :target_rank]
                V_init = V_s[:, :target_rank]
                W_init = W_s[:, :target_rank]
                # Add small noise to break symmetry
                U_init = U_init + np.random.randn(*U_init.shape) * 0.1
                V_init = V_init + np.random.randn(*V_init.shape) * 0.1
                W_init = W_init + np.random.randn(*W_init.shape) * 0.1
            else:
                # Random init
                U_init = np.random.randn(dim_u, target_rank) * 0.1
                V_init = np.random.randn(dim_v, target_rank) * 0.1
                W_init = np.random.randn(dim_w, target_rank) * 0.1
        else:
            U_init = np.random.randn(dim_u, target_rank) * 0.5
            V_init = np.random.randn(dim_v, target_rank) * 0.5
            W_init = np.random.randn(dim_w, target_rank) * 0.5

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        except Exception:
            continue
            
        if res.fun < best_error:
            best_error = res.fun
            if best_error < 1e-9:
                # Extract solution
                size_u = dim_u * target_rank
                size_v = dim_v * target_rank
                size_w = dim_w * target_rank
                
                U_flat = res.x[:size_u]
                V_flat = res.x[size_u:size_u + size_v]
                W_flat = res.x[size_u + size_v:size_u + size_v + size_w]
                
                U_sol = U_flat.reshape(dim_u, target_rank)
                V_sol = V_flat.reshape(dim_v, target_rank)
                W_sol = W_flat.reshape(dim_w, target_rank)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2.0
                V_snap = np.round(V_sol * 2) / 2.0
                W_snap = np.round(W_sol * 2) / 2.0
                
                if _verify_fast(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave margin for the evaluator
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = len(U_school[0])
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower rank solutions
    # Based on inspiration, rank 55 is possible.
    # Try ranks from 55 down to 50, then maybe higher if time permits?
    # Actually, start from low ranks and go up? Or start from schoolbook and go down?
    # Going down is better because we know 60 works.
    
    ranks_to_try = [55, 54, 53, 52, 51, 50, 59, 58, 57, 56]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        if r >= best_rank:
            continue
            
        print(f"Trying rank {r}...") # Debug
        result = _search_rank(n, m, p, r, time_limit=15.0)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Double check
            if _verify_fast(U_sol, V_sol, W_sol, n, m, p) and _is_valid_entries(U_sol, V_sol, W_sol):
                best_rank = r
                best_solution = (U_sol, V_sol, W_sol)
                # If we find 55, we might stop or try lower
                if r <= 50:
                    break

    U_best, V_best, W_best = best_solution
    return {"U": U_best, "V": V_best, "W": W_best}
