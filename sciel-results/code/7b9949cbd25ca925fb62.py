"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import random
import time

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
    
    # Use einsum for efficient computation
    recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return recon

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose tensor T. Returns error value."""
    n, m, p = T.shape
    rank = U.shape[1]
    recon = _compute_reconstruction(U, V, W, n, m, p, rank)
    return np.max(np.abs(recon - T))

def _try_rank_structured(n, m, p, target_rank, max_iterations=200, seed=42, deadline=None):
    """Try to find a decomposition using structured initialization and gradient descent."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    # Initialize with structured pattern: use Kronecker products for block structure
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Try multiple initialization strategies
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init_strategy in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        if init_strategy == 0:
            # Random initialization
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
        elif init_strategy == 1:
            # Block-structured initialization
            for r in range(target_rank):
                # Assign random blocks
                i_block = r % n
                j_block = (r // n) % m
                k_block = (r // (n * m)) % p
                U[i_block * m + j_block, r] = 1.0
                V[j_block * p + k_block, r] = 1.0
                W[k_block * n + i_block, r] = 1.0
        elif init_strategy == 2:
            # Sparse random initialization
            U = rng.randint(0, 2, size=(n * m, target_rank)).astype(float) - 0.5
            V = rng.randint(0, 2, size=(m * p, target_rank)).astype(float) - 0.5
            W = rng.randint(0, 2, size=(p * n, target_rank)).astype(float) - 0.5
        elif init_strategy == 3:
            # Half-integer initialization
            U = rng.randint(0, 4, size=(n * m, target_rank)).astype(float) * 0.5 - 1.0
            V = rng.randint(0, 4, size=(m * p, target_rank)).astype(float) * 0.5 - 1.0
            W = rng.randint(0, 4, size=(p * n, target_rank)).astype(float) * 0.5 - 1.0
        else:
            # Small random perturbation from identity-like structure
            U = rng.randn(n * m, target_rank) * 0.05
            V = rng.randn(m * p, target_rank) * 0.05
            W = rng.randn(p * n, target_rank) * 0.05
            # Add some structure
            for r in range(min(target_rank, n * m)):
                idx = r % (n * m)
                U[idx, r] += 1.0
    
        # Gradient descent with momentum
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        lr = 0.01
        momentum_coeff = 0.9
        
        for iteration in range(max_iterations):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute reconstruction
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            recon = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
            
            # Compute error and gradients
            error = np.max(np.abs(recon - T))
            if error < 1e-12:
                return U, V, W
            
            # Gradients: dE/dU[i*m+j, r] = sum_k (recon[i,j,k] - T[i,j,k]) * V[j*p+k, r] * W[k*n+i, r]
            # Use sign-based gradient for robustness
            diff = recon - T
            
            # Gradient for U
            grad_U = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i * m + j, :] += np.sign(diff[i, j, k]) * V[j * p + k, :] * W[k * n + i, :]
            
            # Gradient for V
            grad_V = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        grad_V[j * p + k, :] += np.sign(diff[i, j, k]) * U[i * m + j, :] * W[k * n + i, :]
            
            # Gradient for W
            grad_W = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        grad_W[k * n + i, :] += np.sign(diff[i, j, k]) * U[i * m + j, :] * V[j * p + k, :]
            
            # Update with momentum
            momentum_U = momentum_coeff * momentum_U + (1 - momentum_coeff) * grad_U
            momentum_V = momentum_coeff * momentum_V + (1 - momentum_coeff) * grad_V
            momentum_W = momentum_coeff * momentum_W + (1 - momentum_coeff) * grad_W
            
            U -= lr * momentum_U
            V -= lr * momentum_V
            W -= lr * momentum_W
            
            # Check convergence
            curr_error = _verify_decomposition(U, V, W, T, tol=1e-10)
            if curr_error < 1e-12:
                return U, V, W
            
            if curr_error < best_error:
                best_error = curr_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def _try_rank(n, m, p, target_rank, max_iterations=100, seed=42, deadline=None):
    """Try to find a decomposition of rank target_rank using alternating least squares."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize randomly with small values to avoid numerical issues
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        for inner_iter in range(50):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)[0]
                    except:
                        pass
            
            # Check error
            curr_error = _verify_decomposition(U, V, W, T, tol=1e-10)
            
            if curr_error < 1e-10:
                return U, V, W
            
            if curr_error < best_error:
                best_error = curr_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_error < 1e-8:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 130  # 20s safety margin before 150s limit
    
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
    # For 2x5x5, schoolbook is 50. Try ranks from 49 down to 25
    for target_rank in range(n * m * p - 1, max(n * m * p - 20, 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try structured approach first
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            
            result = _try_rank_structured(n, m, p, target_rank, max_iterations=100, seed=seed, deadline=deadline)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
        
        # If not found, try ALS approach
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, target_rank, max_iterations=30, seed=seed, deadline=deadline)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
