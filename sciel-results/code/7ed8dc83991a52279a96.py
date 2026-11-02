"""Rewritten program: search for rank-minimal exact tensor decomposition of 4x4*4x5."""
import time
import math
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_baseline():
    """Schoolbook rank-80 decomposition (baseline)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _build_strassen_block():
    """
    Block-wise Strassen structure for 4x4 * 4x5.
    Treat as 2x2 blocks of size 2x2 (left) and 2x2.5 (right) -- approximate.
    Actually, use 2x2 block Strassen on the 4x4 part, and handle the 5th column separately.
    This gives a rank-64 baseline using Strassen on 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    # For simplicity, start from schoolbook; Strassen adaptation is complex for 4x4*4x5.
    # We'll just return schoolbook as the starting point for refinement.
    return _build_baseline()


def _round_to_halfint(A):
    """Round each entry to the nearest half-integer."""
    return np.round(2.0 * A) / 2.0


def _verify(U, V, W, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} (1 if i==k? No.)
    Actually, the tensor T[i,j,k] = 1 if the (i,k) entry of A*B is produced... 
    Wait: the matrix multiplication tensor has T[i, j, k] = 1 if i==j? No.
    
    The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] * B[j, k].
    The tensor T has indices (i, j, k) and T[i, j, k] = 1 if this term contributes, i.e., always 1.
    So T[i, j, k] = 1 for all valid i, j, k.
    
    The decomposition: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor as a 3D array
    T = np.ones((n, m, p))
    # Compute the reconstructed tensor
    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
    err = np.max(np.abs(recon - T))
    return err < tol, err


def _als_step(U, V, W, lr=0.01, sparsity_lambda=0.0):
    """
    One ALS step: update U, V, W in turn.
    Minimize ||T - sum_c U_c * V_c * W_c||^2 + lambda * (soft sparsity penalty).
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.ones((n, m, p))
    
    # Precompute V*W products
    # For updating U: fix V, W. For each i, j, U[i*m+j, :] should satisfy
    # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k.
    # This is a linear system: U[i*m+j, :] * M = 1 where M[k, c] = V[j*p+k, c] * W[k*n+i, c].
    
    # Update U
    U_new = np.copy(U)
    for i in range(n):
        for j in range(m):
            # Build M: (p, rank)
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V[j * p + k, :] * W[k * n + i, :]
            # Solve U[i*m+j, :] * M = 1 (vector of ones)
            # This is underdetermined if rank > p. Use least squares.
            try:
                sol, residuals, rankM, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                U_new[i * m + j, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    # Update V
    V_new = np.copy(V)
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                sol, residuals, rankM, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                V_new[j * p + k, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    # Update W
    W_new = np.copy(W)
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                sol, residuals, rankM, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                W_new[k * n + i, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    return U_new, V_new, W_new


def _soft_sparsity_penalty(A, lam=0.01):
    """Log-sum-exp based soft sparsity penalty."""
    return lam * np.sum(np.logaddexp(0, np.abs(A) * 10))


def construct():
    deadline = time.time() + 110  # 110s safety margin
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 80
    
    # Start from schoolbook
    U, V, W = _build_baseline()
    best_rank = base_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify baseline
    ok, err = _verify(U, V, W)
    if not ok:
        # Should not happen, but return baseline anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try ALS refinement with decreasing rank
    # Start with rank = base_rank, try to reduce
    current_rank = base_rank
    U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
    
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try to reduce rank by 1
        new_rank = current_rank - 1
        if new_rank < 1:
            break
        
        # Initialize with current factors, drop the last column
        U_try = U_curr[:, :new_rank].copy()
        V_try = V_curr[:, :new_rank].copy()
        W_try = W_curr[:, :new_rank].copy()
        
        # Run ALS for a few iterations
        converged = False
        for it in range(50):
            if time.time() > deadline - 5:
                break
            U_try, V_try, W_try = _als_step(U_try, V_try, W_try)
            
            # Round to half-integers and check
            U_r = _round_to_halfint(U_try)
            V_r = _round_to_halfint(V_try)
            W_r = _round_to_halfint(W_try)
            
            ok, err = _verify(U_r, V_r, W_r)
            if ok:
                # Found a valid decomposition with lower rank!
                current_rank = new_rank
                U_curr, V_curr, W_curr = U_r.copy(), V_r.copy(), W_r.copy()
                best_rank = current_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                converged = True
                break
        
        if not converged:
            # Try a few more iterations with the current rank
            for it in range(20):
                if time.time() > deadline - 5:
                    break
                U_try, V_try, W_try = _als_step(U_try, V_try, W_try)
                U_r = _round_to_halfint(U_try)
                V_r = _round_to_halfint(V_try)
                W_r = _round_to_halfint(W_try)
                ok, err = _verify(U_r, V_r, W_r)
                if ok:
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
        
        if time.time() > deadline - 5:
            break
    
    # Final verification
    ok, err = _verify(best_U, best_V, best_W)
    if not ok:
        # Fall back to schoolbook
        U, V, W = _build_baseline()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
