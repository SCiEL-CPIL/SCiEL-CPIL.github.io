"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if output (i,k) depends on input (i,j) and (j,k)."""
    T = np.ones((n, m, p))
    return T

def _compute_reconstruction(U, V, W, n, m, p, rank):
    """Compute reconstruction T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return recon

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose tensor T. Returns error value."""
    n, m, p = T.shape
    rank = U.shape[1]
    recon = _compute_reconstruction(U, V, W, n, m, p, rank)
    return np.max(np.abs(recon - T))

def _als_refine(U, V, W, T, n, m, p, target_rank, max_inner=200, tol=1e-12, deadline=None):
    """Refine U, V, W using alternating least squares."""
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for inner_iter in range(max_inner):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ A where A[r,k] = V[j*p+k,r]*W[k*n+i,r]
                A = np.zeros((target_rank, p))
                for r in range(target_rank):
                    for k in range(p):
                        A[r, k] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U[i * m + j, :], _, _, _ = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)
                except:
                    pass
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ B where B[r,i] = U[i*m+j,r]*W[k*n+i,r]
                B = np.zeros((target_rank, n))
                for r in range(target_rank):
                    for i in range(n):
                        B[r, i] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V[j * p + k, :], _, _, _ = np.linalg.lstsq(B.T, T[:, j, k], rcond=None)
                except:
                    pass
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ C where C[r,j] = U[i*m+j,r]*V[j*p+k,r]
                C = np.zeros((target_rank, m))
                for r in range(target_rank):
                    for j in range(m):
                        C[r, j] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W[k * n + i, :], _, _, _ = np.linalg.lstsq(C.T, T[i, :, k], rcond=None)
                except:
                    pass
        
        # Check error
        curr_error = _verify_decomposition(U, V, W, T, tol=1e-10)
        
        if curr_error < best_error:
            best_error = curr_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if curr_error < tol:
            break
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def _try_rank_with_init(n, m, p, target_rank, U_init, V_init, W_init, max_inner=300, tol=1e-12, deadline=None):
    """Try to refine a given initialization to find a valid decomposition."""
    T = _build_tensor(n, m, p)
    result = _als_refine(U_init, V_init, W_init, T, n, m, p, target_rank, max_inner, tol, deadline)
    if result is not None:
        return result
    return None

def _generate_structured_init(n, m, p, target_rank, rng):
    """Generate structured initializations that exploit matrix multiplication structure."""
    inits = []
    
    # Strategy 1: Block-diagonal-like structure
    U1 = np.zeros((n * m, target_rank))
    V1 = np.zeros((m * p, target_rank))
    W1 = np.zeros((p * n, target_rank))
    for r in range(target_rank):
        i_idx = r % n
        j_idx = (r // n) % m
        k_idx = (r // (n * m)) % p
        U1[i_idx * m + j_idx, r] = 1.0
        V1[j_idx * p + k_idx, r] = 1.0
        W1[k_idx * n + i_idx, r] = 1.0
    # Add small random perturbation to escape local minima
    U1 += rng.randn(n * m, target_rank) * 0.01
    V1 += rng.randn(m * p, target_rank) * 0.01
    W1 += rng.randn(p * n, target_rank) * 0.01
    inits.append((U1, V1, W1))
    
    # Strategy 2: Random sparse initialization
    U2 = np.zeros((n * m, target_rank))
    V2 = np.zeros((m * p, target_rank))
    W2 = np.zeros((p * n, target_rank))
    for r in range(target_rank):
        # Each rank term touches a random subset of indices
        u_sparsity = max(1, min(n * m, rng.randint(1, 3)))
        v_sparsity = max(1, min(m * p, rng.randint(1, 3)))
        w_sparsity = max(1, min(p * n, rng.randint(1, 3)))
        
        u_idx = rng.choice(n * m, size=u_sparsity, replace=False)
        v_idx = rng.choice(m * p, size=v_sparsity, replace=False)
        w_idx = rng.choice(p * n, size=w_sparsity, replace=False)
        
        for idx in u_idx:
            U2[idx, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for idx in v_idx:
            V2[idx, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for idx in w_idx:
            W2[idx, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    inits.append((U2, V2, W2))
    
    # Strategy 3: Kronecker-like structure for 2x5x5
    # Use the fact that n=2 is small
    U3 = np.zeros((n * m, target_rank))
    V3 = np.zeros((m * p, target_rank))
    W3 = np.zeros((p * n, target_rank))
    
    # Try to use pairs of columns that interact
    for r in range(target_rank):
        # Assign to random (i,j) for U, (j,k) for V, (k,i) for W
        i_val = rng.randint(0, n - 1)
        j_val = rng.randint(0, m - 1)
        k_val = rng.randint(0, p - 1)
        
        U3[i_val * m + j_val, r] = rng.uniform(0.5, 1.5)
        V3[j_val * p + k_val, r] = rng.uniform(0.5, 1.5)
        W3[k_val * n + i_val, r] = rng.uniform(0.5, 1.5)
        
        # Add a second component for some ranks
        if r % 3 == 0 and target_rank > 3:
            i_val2 = rng.randint(0, n - 1)
            j_val2 = rng.randint(0, m - 1)
            k_val2 = rng.randint(0, p - 1)
            U3[i_val2 * m + j_val2, r] += rng.uniform(0.3, 0.8)
            V3[j_val2 * p + k_val2, r] += rng.uniform(0.3, 0.8)
            W3[k_val2 * n + i_val2, r] += rng.uniform(0.3, 0.8)
    
    inits.append((U3, V3, W3))
    
    # Strategy 4: Pure random
    U4 = rng.randn(n * m, target_rank) * 0.3
    V4 = rng.randn(m * p, target_rank) * 0.3
    W4 = rng.randn(p * n, target_rank) * 0.3
    inits.append((U4, V4, W4))
    
    return inits

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Based on literature, rank for 2x5x5 is likely around 35-45
    # Try ranks from 49 down to 30
    for target_rank in range(n * m * p - 1, max(n * m * p - 25, 25), -1):
        if time.time() > deadline:
            break
        
        found = False
        rng = np.random.RandomState(target_rank * 7919)
        
        # Try multiple structured initializations
        inits = _generate_structured_init(n, m, p, target_rank, rng)
        
        for init_idx, (U_init, V_init, W_init) in enumerate(inits):
            if time.time() > deadline:
                break
            
            result = _try_rank_with_init(n, m, p, target_rank, U_init, V_init, W_init, 
                                          max_inner=200, tol=1e-12, deadline=deadline)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
