"""Matrix multiplication 3x3x5 rank minimization via ALS with sparsity."""
DIMENSIONS = [3, 3, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
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


def _tensor_expected(n, m, p):
    """Build the expected matmul tensor T[i,j,k] = 1 if j==k else 0."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for j in range(min(m, p)):
        T[:, j, j] = 1.0
    return T


def _reconstruction_error(U, V, W, T, n, m, p):
    """Compute Frobenius norm of reconstruction error."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T_recon[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)
    T_recon = np.einsum('ijkc,kic->ijk', UV, W3)
    return np.linalg.norm(T_recon - T)


def _round_to_halfint(x, tol=0.15):
    """Round values to nearest half-integer if close enough, else keep."""
    result = np.round(x * 2) / 2.0
    # Check if rounding is close
    if np.max(np.abs(x - result)) < tol:
        return result
    return x


def _als_step(U, V, W, T, n, m, p, l1_reg, max_iter=50, tol=1e-8):
    """
    Alternating Least Squares to minimize reconstruction error.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    T: (n, m, p) expected tensor
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = _reconstruction_error(U, V, W, T, n, m, p)
    
    for it in range(max_iter):
        # Update U: for each column c, minimize ||T - sum_c' U_c' ⊙ V_c' ⊙ W_c'||
        # U[i*m+j, c] = argmin sum_{i,j,k} (T[i,j,k] - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        # This is linear in U for fixed V, W
        
        for c in range(rank):
            # Residual without column c
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # Compute contribution of column c
            # T_recon[i,j,k] = sum_c' U3[i,j,c']*V3[j,k,c']*W3[k,i,c']
            # We want to find U3[:,:,c] to minimize ||T - (T_recon - U3[:,:,c]*V3[:,:,c]*W3[:,:,c])||^2
            # = ||T - T_recon_other - U3[:,:,c] * (V3[:,:,c] * W3[:,:,c])||^2
            
            # Compute product P[i,j,k] = V3[j,k,c] * W3[k,i,c]
            P = np.einsum('jkc,kic->ijk', V3[:, :, c], W3[:, :, c])
            
            # T_recon_other[i,j,k] = sum_{c' != c} U3[i,j,c']*V3[j,k,c']*W3[k,i,c']
            # = T_recon[i,j,k] - U3[i,j,c]*P[i,j,k]
            
            # Full T_recon
            UV_full = np.einsum('ijc,jkc->ijkc', U3, V3)
            T_recon = np.einsum('ijkc,kic->ijk', UV_full, W3)
            T_other = T_recon - U3[:, :, c] * P
            
            # Now minimize ||T - T_other - U3[:,:,c] * P||^2 over U3[:,:,c]
            # This is a linear system: for each (i,j), U3[i,j,c] * sum_k P[i,j,k]^2 = sum_k (T[i,j,k] - T_other[i,j,k]) * P[i,j,k]
            
            # Flatten: U_vec = U3[:,:,c].reshape(-1) of length n*m
            # A[i,j] = sum_k P[i,j,k]^2
            # b[i,j] = sum_k (T[i,j,k] - T_other[i,j,k]) * P[i,j,k]
            
            A = np.sum(P ** 2, axis=2)  # (n, m)
            b = np.sum((T - T_other) * P, axis=2)  # (n, m)
            
            # Solve A * U_vec = b element-wise
            U_new = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    if A[i, j] > 1e-12:
                        U_new[i, j] = b[i, j] / A[i, j]
                    else:
                        U_new[i, j] = U3[i, j, c]
            
            # L1 regularization: soft threshold
            U_new = np.sign(U_new) * np.maximum(np.abs(U_new) - l1_reg, 0)
            
            U3[:, :, c] = U_new
            U = U3.reshape(n * m, rank)
        
        # Update V similarly
        for c in range(rank):
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            P = np.einsum('ijc,kic->ijk', U3[:, :, c], W3[:, :, c])  # P[i,j,k] = U3[i,j,c]*W3[k,i,c]
            
            UV_full = np.einsum('ijc,jkc->ijkc', U3, V3)
            T_recon = np.einsum('ijkc,kic->ijk', UV_full, W3)
            T_other = T_recon - V3[:, :, c] * P
            
            A = np.sum(P ** 2, axis=0)  # (p,) -> for each k, sum_{i,j} P[i,j,k]^2
            # Wait, P is (n,m,p), V3[:,:,c] is (m,p). 
            # V3[j,k,c] multiplies P[i,j,k]. So for each (j,k):
            # A[j,k] = sum_i P[i,j,k]^2
            # b[j,k] = sum_i (T[i,j,k] - T_other[i,j,k]) * P[i,j,k]
            
            A = np.sum(P ** 2, axis=0)  # (m, p)
            b = np.sum((T - T_other) * P, axis=0)  # (m, p)
            
            V_new = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    if A[j, k] > 1e-12:
                        V_new[j, k] = b[j, k] / A[j, k]
                    else:
                        V_new[j, k] = V3[j, k, c]
            
            V_new = np.sign(V_new) * np.maximum(np.abs(V_new) - l1_reg, 0)
            V3[:, :, c] = V_new
            V = V3.reshape(m * p, rank)
        
        # Update W similarly
        for c in range(rank):
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # W3[k,i,c] multiplies U3[i,j,c]*V3[j,k,c]. For each (k,i):
            # P[k,i,j] = U3[i,j,c] * V3[j,k,c]
            P = np.einsum('ijc,jkc->kij', U3[:, :, c], V3[:, :, c])  # (p, n, m)
            
            UV_full = np.einsum('ijc,jkc->ijkc', U3, V3)
            T_recon = np.einsum('ijkc,kic->ijk', UV_full, W3)
            T_other = T_recon - W3[:, :, c] * np.einsum('kij,ijk->kij', P, np.ones((n,m,p)))
            # Actually: T_other[i,j,k] = T_recon[i,j,k] - W3[k,i,c] * P[k,i,j]
            T_other = T_recon - np.einsum('kic,kij->ijk', W3[:, :, c], P)
            
            A = np.sum(P ** 2, axis=2)  # (p, n)
            b = np.sum((T - T_other) * np.transpose(P, (2, 0, 1)), axis=2)  # (p, n)
            # b[k,i] = sum_j (T[i,j,k] - T_other[i,j,k]) * P[k,i,j]
            
            W_new = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    if A[k, i] > 1e-12:
                        W_new[k, i] = b[k, i] / A[k, i]
                    else:
                        W_new[k, i] = W3[k, i, c]
            
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - l1_reg, 0)
            W3[:, :, c] = W_new
            W = W3.reshape(p * n, rank)
        
        err = _reconstruction_error(U, V, W, T, n, m, p)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err


def _try_rank(U_init, V_init, W_init, T, n, m, p, l1_reg, als_iters):
    """Try to refine initialization to get exact decomposition."""
    U, V, W, err = _als_step(U_init, V_init, W_init, T, n, m, p, l1_reg, als_iters)
    
    # Round to half-integers
    U_r = _round_to_halfint(U)
    V_r = _round_to_halfint(V)
    W_r = _round_to_halfint(W)
    
    # Validate
    U3 = U_r.reshape(n, m, -1)
    V3 = V_r.reshape(m, p, -1)
    W3 = W_r.reshape(p, n, -1)
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)
    T_recon = np.einsum('ijkc,kic->ijk', UV, W3)
    
    if np.allclose(T_recon, T, atol=1e-9):
        return U_r, V_r, W_r, True
    
    # Try without rounding
    if np.allclose(T_recon, T, atol=1e-9):
        return U, V, W, True
    
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    T = _tensor_expected(n, m, p)
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = np.random.RandomState(12345)
    
    # Try ranks from 43 downward (known lower bound is 43)
    for target_rank in range(43, 44, -1):
        if time.time() > deadline:
            break
        
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.choice([-1, 0, 1], size=(n * m, target_rank)).astype(np.float64)
            V = rng.choice([-1, 0, 1], size=(m * p, target_rank)).astype(np.float64)
            W = rng.choice([-1, 0, 1], size=(p * n, target_rank)).astype(np.float64)
            
            # Also try Gaussian init
            if restart % 2 == 1:
                U = rng.randn(n * m, target_rank) * 0.5
                V = rng.randn(m * p, target_rank) * 0.5
                W = rng.randn(p * n, target_rank) * 0.5
            
            # Run ALS with varying L1 regularization
            for l1_reg in [0.0, 0.05, 0.1, 0.2]:
                if time.time() > deadline:
                    break
                
                U_out, V_out, W_out, success = _try_rank(
                    U.copy(), V.copy(), W.copy(), T, n, m, p, l1_reg, 30
                )
                
                if success:
                    best = {
                        "U": U_out.tolist(),
                        "V": V_out.tolist(),
                        "W": W_out.tolist()
                    }
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best
    else:
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
