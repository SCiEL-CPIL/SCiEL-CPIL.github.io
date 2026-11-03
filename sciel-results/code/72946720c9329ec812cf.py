import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 6]

def _target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = _target_tensor(n, m, p)
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _snap_to_half_integers(arr, tol=1e-6):
    # Snap to nearest half-integer
    snapped = np.round(arr * 2) / 2
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    return None

def _try_rank(n, m, p, r, max_iters=100, time_limit=5):
    start = time.time()
    T = _target_tensor(n, m, p)
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    for _ in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Random U, V with small integers
        U = np.random.randint(-1, 2, (n_rows, r))
        V = np.random.randint(-1, 2, (m_rows, r))
        
        # Solve for W
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # For each c, we have a system for W[c, :]
        # But W[c, r] is shared for all a, b.
        # Let's form the linear system A W_flat = T_flat
        # Unknowns: W[c, r] for c in 0..p_rows-1, r in 0..r-1
        # Total unknowns: p_rows * r
        # Equations: T[a, b, c] for all a, b, c
        # A is a matrix of size (n_rows * m_rows * p_rows) x (p_rows * r)
        # A[(a * m_rows + b) * p_rows + c, c * r + r_idx] = U[a, r_idx] * V[b, r_idx]
        
        A = np.zeros((n_rows * m_rows * p_rows, p_rows * r))
        for a in range(n_rows):
            for b in range(m_rows):
                for c in range(p_rows):
                    for r_idx in range(r):
                        A[(a * m_rows + b) * p_rows + c, c * r + r_idx] = U[a, r_idx] * V[b, r_idx]
        
        T_flat = T.flatten()
        
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            W = W_flat.reshape(p_rows, r)
            
            # Check if the solution is exact
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_dec, T, atol=1e-6):
                # Try to snap U, V, W to half-integers
                U_snap = _snap_to_half_integers(U)
                V_snap = _snap_to_half_integers(V)
                W_snap = _snap_to_half_integers(W)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap
        except np.linalg.LinAlgError:
            continue
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 18 to best_rank - 1
    for r in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(n, m, p, r, max_iters=20, time_limit=2)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found the lowest rank, stop
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
