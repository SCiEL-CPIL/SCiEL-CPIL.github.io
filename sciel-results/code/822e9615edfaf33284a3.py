"""
Matrix multiplication tensor rank minimization for 2x4 * 4x7.
Uses random search with gradient-free refinement to find a lower-rank
decomposition than the schoolbook rank of 56.
"""
import time
import random
import math
import numpy as np

DIMENSIONS = [2, 4, 7]


def _verify(U, V, W):
    """Check whether the decomposition exactly reproduces the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Compute all triple products
    # For each (i, j, k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build the tensor implicitly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _to_half_int_matrix(M, tol=1e-9):
    """Check if all entries are within tol of half-integers; return rounded version."""
    M = np.array(M, dtype=float)
    scaled = M * 2.0
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) > tol:
        return None
    return (rounded / 2.0)


def _rand_init(n, m, p, rank, seed=None):
    """Random initialization of U, V, W."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W


def _objective(U, V, W, n, m, p):
    """Sum of squared errors over all (i,j,k) pairs."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s = float(np.dot(u_row, v_row * w_row))
                err += (s - 1.0) ** 2
    return err


def _refine(U, V, W, n, m, p, lr=0.001, iters=200, rng_seed=0):
    """Simple coordinate-descent refinement."""
    rng = np.random.RandomState(rng_seed)
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    best_err = _objective(U, V, W, n, m, p)
    
    for it in range(iters):
        # Pick random element in one of the three matrices
        mat = rng.randint(3)
        if mat == 0:
            rows, cols = n * m, rank
        elif mat == 1:
            rows, cols = m * p, rank
        else:
            rows, cols = p * n, rank
        
        r = rng.randint(rows)
        c = rng.randint(cols)
        
        if mat == 0:
            old = U[r, c]
        elif mat == 1:
            old = V[r, c]
        else:
            old = W[r, c]
        
        # Try small perturbation
        step = lr * (0.5 + rng.random())
        for sign in [1, -1]:
            new_val = old + sign * step
            if mat == 0:
                U[r, c] = new_val
                err = _objective(U, V, W, n, m, p)
                U[r, c] = old
            elif mat == 1:
                V[r, c] = new_val
                err = _objective(V, W, U, n, m, p) if False else _objective(U, V, W, n, m, p)
                V[r, c] = old
            else:
                W[r, c] = new_val
                err = _objective(U, V, W, n, m, p)
                W[r, c] = old
            
            if err < best_err - 1e-12:
                if mat == 0:
                    U[r, c] = new_val
                elif mat == 1:
                    V[r, c] = new_val
                else:
                    W[r, c] = new_val
                best_err = err
                break
    
    return U, V, W, best_err


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56
    deadline = time.time() + 140  # safety margin
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try random search for lower rank
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Also rank >= n*m*p / min(...) is not standard. Try from 28 upward.
    
    # Try various ranks below schoolbook
    for rank in range(28, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 7919 + 13
            U, V, W = _rand_init(n, m, p, rank, seed=seed)
            
            # Quick check: scale to make magnitudes reasonable
            # Each (i,j,k) needs dot product = 1
            # With random init, typical |dot| ~ sqrt(rank) * (1/sqrt(rank))^3 = 1/rank
            # Scale to get order-1 products
            scale = rank ** 0.5
            U = U / scale
            V = V / scale
            W = W / scale
            
            err = _objective(U, V, W, n, m, p)
            
            # Refine
            U, V, W, err = _refine(U, V, W, n, m, p, lr=0.05, iters=50, rng_seed=seed)
            
            if err < 1e-6:
                # Check for exact solution
                # Try rounding to half-integers
                Uh = _to_half_int_matrix(U, tol=1e-4)
                Vh = _to_half_int_matrix(V, tol=1e-4)
                Wh = _to_half_int_matrix(W, tol=1e-4)
                if Uh is not None and Vh is not None and Wh is not None:
                    if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist()):
                        found = True
                        best_rank = rank
                        best_U = Uh.tolist()
                        best_V = Vh.tolist()
                        best_W = Wh.tolist()
                        break
                # Also check raw
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    found = True
                    best_rank = rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Convert best to list format
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}


# EVOLVE-BLOCK-END
