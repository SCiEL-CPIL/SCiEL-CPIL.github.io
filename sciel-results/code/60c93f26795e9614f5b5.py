"""Matrix multiplication 3,4,4 rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _tensor_ijk(n, m, p):
    """Build the matrix-multiplication tensor T[i,j,k] = 1 if the (i,k) output
    depends on A[i,j]*B[j,k], else 0. Shape (n, m, p)."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _to_mat(T, mode):
    """Reshape tensor to a matrix with `mode` as the row index (flattened)."""
    n, m, p = T.shape
    if mode == 0:  # i rows, (j,k) cols
        return T.reshape(n, m * p)
    elif mode == 1:  # j rows, (i,k) cols
        return T.transpose(1, 0, 2).reshape(m, n * p)
    elif mode == 2:  # k rows, (i,j) cols
        return T.transpose(2, 0, 1).reshape(p, n * m)

def _reconstruct(U, V, W, n, m, p):
    """Reconstruct tensor from factors. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Returns T[i,j,k]."""
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # T[i,j,k] += uc[i,j]*vc[j,k]*wc[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return T

def _round_to_half(x):
    """Round to nearest multiple of 0.5."""
    return np.round(x * 2.0) / 2.0

def _is_half_integer(x, tol=1e-9):
    return np.allclose(x * 2, np.round(x * 2), atol=tol)

def _verify_exact(U, V, W, T_target, n, m, p, tol=1e-6):
    """Check if factors exactly reconstruct the target tensor."""
    T_recon = _reconstruct(U, V, W, n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _als_step(U, V, W, T, n, m, p, mode):
    """One ALS iteration: update factor `mode` while keeping others fixed.
    mode: 0=U, 1=V, 2=W."""
    r = U.shape[1]
    if mode == 0:
        # Update U: U = T_mat0 * (V ⊙ W)^T * (V⊙W)^T(V⊙W) ... 
        # Actually U[i*m+j, c] such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
        # For fixed V,W: U[:,c] = argmin sum_{i,j,k} (T[i,j,k] - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        # This is: U[:,c] = (T_mat0 * (V_c ⊙ W_c)) / (V_c ⊙ W_c)^T (V_c ⊙ W_c)  ... not quite
        # Let's do it properly with the outer product structure.
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed V,W, U is found by: for each c, U[:,c] = M_c / ||d_c||^2
        # where d_c[i*m+j] = sum_{k} V[j*p+k,c]*W[k*n+i,c]  ... no, this is per-element.
        
        # Better: use the standard ALS for 3-way tensors.
        # U = T_(0) * (V ⊙ W) * (V ⊙ W)^T (V ⊙ W)^{-1}
        # where ⊙ is the Khatri-Rao product.
        # V ⊙ W: for column c, (V⊙W)[:,c] = V[:,c] ⊗ W[:,c] but reshaped appropriately.
        # Actually for our specific structure:
        # The "flattened" Khatri-Rao: KR = V_{(m*p, r)} ⊙ W_{(p*n, r)} 
        # The product V[j*p+k,c]*W[k*n+i,c] for fixed c gives a matrix A_c[i*m+j, k] ... 
        # This is getting complex. Let me just do the direct update.
        
        # For each column c of U, we need to minimize over u = U[:,c]:
        # sum_{i,j,k} (T[i,j,k] - u[i*m+j]*V[j*p+k,c]*W[k*n+i,c])^2
        # Let a_{i,j,k} = V[j*p+k,c]*W[k*n+i,c]
        # Then u[i*m+j] = sum_{i,j,k: fixed i,j} T[i,j,k]*a_{i,j,k} / sum_{i,j,k: fixed i,j} a_{i,j,k}^2
        # But a depends on i,j,k through V and W indices.
        
        # Actually, let me just use a simpler approach: treat it as a linear system.
        # T_flat = A_c * u_c where A_c is the design matrix for column c.
        # But T is shared across all c. So we solve jointly.
        
        # Let me just do gradient-based or use the fact that this is a standard tensor ALS.
        # For mode-0 unfolding: U = T_(0) * (V ⊙ W) * ((V ⊙ W)^T (V ⊙ W))^{-1}
        # where (V ⊙ W) has columns (V[:,c] ⊗ W[:,c]) but we need the right shape.
        
        # For our specific tensor structure where T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]:
        # The Khatri-Rao product matrix K has K[i*m+j, l*m*p+k... ] no.
        
        # Let me just do it numerically with a loop for small dimensions.
        U_new = np.zeros_like(U)
        for c in range(r):
            # For this column, the contribution is outer product structure
            # We want to find u = U[:,c] to best fit T - (contributions from other columns)
            residual = T.copy()
            for c2 in range(r):
                if c2 == c:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
            # Now minimize sum_{i,j,k} (residual[i,j,k] - u[i*m+j]*V[j*p+k,c]*W[k*n+i,c])^2
            # Group by (i,j): for each (i,j), the coefficient is sum_k V[j*p+k,c]*W[k*n+i,c] * ... 
            # Actually it's: for each (i,j), u[i*m+j] = sum_k residual[i,j,k]*V[j*p+k,c]*W[k*n+i,c] / sum_k (V[j*p+k,c]*W[k*n+i,c])^2
            for i in range(n):
                for j in range(m):
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        a = V[j*p+k, c] * W[k*n+i, c]
                        num += residual[i, j, k] * a
                        den += a * a
                    if den > 1e-12:
                        U_new[i*m+j, c] = num / den
                    else:
                        U_new[i*m+j, c] = U[i*m+j, c]
        return U_new, V, W
    elif mode == 1:
        V_new = np.zeros_like(V)
        for c in range(r):
            residual = T.copy()
            for c2 in range(r):
                if c2 == c:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        a = U[i*m+j, c] * W[k*n+i, c]
                        num += residual[i, j, k] * a
                        den += a * a
                    if den > 1e-12:
                        V_new[j*p+k, c] = num / den
                    else:
                        V_new[j*p+k, c] = V[j*p+k, c]
        return U, V_new, W
    elif mode == 2:
        W_new = np.zeros_like(W)
        for c in range(r):
            residual = T.copy()
            for c2 in range(r):
                if c2 == c:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i, j, k] -= U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
            for k in range(p):
                for i in range(n):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        a = U[i*m+j, c] * V[j*p+k, c]
                        num += residual[i, j, k] * a
                        den += a * a
                    if den > 1e-12:
                        W_new[k*n+i, c] = num / den
                    else:
                        W_new[k*n+i, c] = W[k*n+i, c]
        return U, V, W_new

def _als_decompose(T, n, m, p, rank, max_iter=100, tol=1e-8, seed=None):
    """ALS decomposition of tensor T into rank-r factors."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    prev_err = np.inf
    for it in range(max_iter):
        U, V, W = _als_step(U, V, W, T, n, m, p, 0)
        U, V, W = _als_step(U, V, W, T, n, m, p, 1)
        U, V, W = _als_step(U, V, W, T, n, m, p, 2)
        
        err = np.linalg.norm(_reconstruct(U, V, W, n, m, p) - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _try_round_and_verify(U, V, W, T, n, m, p):
    """Try rounding factors to half-integers and verify exactness."""
    U_r = _round_to_half(U)
    V_r = _round_to_half(V)
    W_r = _round_to_half(W)
    if _verify_exact(U_r, V_r, W_r, T, n, m, p):
        return U_r, V_r, W_r
    return None

def _try_int_round(U, V, W, T, n, m, p):
    """Try rounding to integers."""
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    if _verify_exact(U_r, V_r, W_r, T, n, m, p):
        return U_r, V_r, W_r
    return None

def _refine_with_rounding(U, V, W, T, n, m, p):
    """Try various rounding strategies."""
    # Half-integer rounding
    result = _try_round_and_verify(U, V, W, T, n, m, p)
    if result is not None:
        return result
    # Integer rounding
    result = _try_int_round(U, V, W, T, n, m, p)
    if result is not None:
        return result
    # Try rounding with small perturbations
    rng = np.random
    for _ in range(50):
        U_p = U + rng.randn(*U.shape) * 0.01
        V_p = V + rng.randn(*V.shape) * 0.01
        W_p = W + rng.randn(*W.shape) * 0.01
        result = _try_round_and_verify(U_p, V_p, W_p, T, n, m, p)
        if result is not None:
            return result
        result = _try_int_round(U_p, V_p, W_p, T, n, m, p)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _tensor_ijk(n, m, p)
    T_target = T
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Known: for 3x4x4, rank is likely around 18-25
    best_result = None
    
    for rank in range(16, 49):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _als_decompose(T, n, m, p, rank, max_iter=50, seed=restart)
            
            # Check if we got a good fit
            recon = _reconstruct(U, V, W, n, m, p)
            err = np.linalg.norm(recon - T_target)
            
            if err < 0.1:
                # Try to round and verify
                result = _refine_with_rounding(U, V, W, T_target, n, m, p)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_result = (rank, U_r, V_r, W_r)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
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
