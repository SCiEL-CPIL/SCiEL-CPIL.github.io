"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication
    using ALS optimization with half-integer rounding.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp * pn // p  # n*m*p = 50
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Build the target tensor T[i,j,k] = 1 if we want U[i,j]*V[j,k]*W[k,i]
    # The matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Precompute the structure
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But we know rank >= max(n*m, m*p, p*n) for bilinear complexity
    # Actually for matrix mult, rank >= n*m, m*p, p*n individually
    # For 2x5*5x5: rank >= max(10, 25, 10) = 25
    
    # Known lower bounds are higher. Let's try from reasonable values.
    # Start from rank 35 and go up to schoolbook
    for target_rank in range(35, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 5
        if time.time() > deadline - 60:
            max_restarts = 2
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            result = _try_als_decomposition(n, m, p, target_rank, restart)
            if result is not None:
                rank_found = result["rank"]
                if rank_found < best_rank:
                    best_rank = rank_found
                    best_result = result
                # Found a valid decomposition, try to go lower
                break  # Break restart loop, try next (lower) rank
    
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
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


def _try_als_decomposition(n, m, p, target_rank, seed_offset):
    """
    Try to find a rank-target_rank decomposition using ALS.
    Returns dict with U, V, W if successful, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    rng = random.Random(seed_offset * 1000 + target_rank)
    
    # Random initialization
    U = np.random.randn(nm, target_rank) * 0.5
    V = np.random.randn(mp, target_rank) * 0.5
    W = np.random.randn(pn, target_rank) * 0.5
    
    # Scale to make the product closer to 1
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
    V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
    U *= target_rank ** 0.5
    V *= target_rank ** 0.5
    W *= target_rank ** 0.5
    
    # ALS iterations
    max_iter = 200
    tol = 1e-8
    
    prev_loss = float('inf')
    
    for it in range(max_iter):
        # Update U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # For fixed (i,j), the "effective" target is 1 for all k
        # U[i*m+j, :] should satisfy: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
        
        # Build the effective RHS for U
        # For each (i,j): solve U[i*m+j,:] * M_ij = ones(p)
        # where M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                # M_ij has shape (p, target_rank)
                M_ij = np.zeros((p, target_rank))
                for k in range(p):
                    M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve: U[i*m+j, :] @ M_ij.T = ones(p)
                # i.e., M_ij @ U[i*m+j, :].T = ones(p)
                rhs = np.ones(p)
                try:
                    # Least squares: M_ij @ x = rhs
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_ij, rhs, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), V[j*p+k, :] should satisfy
        # sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        for j in range(m):
            for k in range(p):
                M_jk = np.zeros((n, target_rank))
                for i in range(n):
                    M_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                rhs = np.ones(n)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_jk, rhs, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), W[k*n+i, :] should satisfy
        # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        for k in range(p):
            for i in range(n):
                M_ki = np.zeros((m, target_rank))
                for j in range(m):
                    M_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                rhs = np.ones(m)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M_ki, rhs, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute loss
        loss = _compute_loss(n, m, p, U, V, W)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    # Try to snap to half-integers and verify
    snapped = _snap_and_validate(n, m, p, U, V, W)
    if snapped is not None:
        return snapped
    
    return None


def _compute_loss(n, m, p, U, V, W):
    """Compute the loss: sum over all (i,j,k) of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2"""
    nm = n * m
    mp = m * p
    pn = p * n
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                loss += (val - 1.0) ** 2
    return loss


def _snap_and_validate(n, m, p, U, V, W):
    """
    Try to snap U, V, W to half-integers and verify exactness.
    Returns dict if valid, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Try different rounding strategies
    for strategy in ['round', 'floor', 'ceil']:
        U_snap = _snap_matrix(U, strategy)
        V_snap = _snap_matrix(V, strategy)
        W_snap = _snap_matrix(W, strategy)
        
        if _is_valid(n, m, p, U_snap, V_snap, W_snap):
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist(),
                "rank": rank
            }
    
    # Try mixed strategies
    for u_strat in ['round']:
        for v_strat in ['round']:
            for w_strat in ['round']:
                U_snap = _snap_matrix(U, u_strat)
                V_snap = _snap_matrix(V, v_strat)
                W_snap = _snap_matrix(W, w_strat)
                
                if _is_valid(n, m, p, U_snap, V_snap, W_snap):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist(),
                        "rank": rank
                    }
    
    return None


def _snap_matrix(M, strategy):
    """Snap matrix entries to half-integers."""
    if strategy == 'round':
        return np.round(M * 2) / 2
    elif strategy == 'floor':
        return np.floor(M * 2) / 2
    elif strategy == 'ceil':
        return np.ceil(M * 2) / 2
    else:
        return np.round(M * 2) / 2


def _is_valid(n, m, p, U, V, W):
    """
    Check if the decomposition is exact.
    For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                # Check if total is very close to 1
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
