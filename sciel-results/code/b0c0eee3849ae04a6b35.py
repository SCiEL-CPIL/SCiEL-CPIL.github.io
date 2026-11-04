"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor exactly."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    # Build target tensor (all ones)
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Compute reconstruction
    recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    
    # Check if all entries are approximately 1
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition."""
    rank = len(U[0]) if U else 0
    recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    return np.linalg.norm(recon - np.ones((n, m, p)))

def _als_refine(U, V, W, n, m, p, max_iter=200, tol=1e-10, time_limit=5.0):
    """Alternating least squares refinement."""
    start_time = time.time()
    rank = len(U[0]) if U else 0
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_error = _compute_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let A[r,k] = V[j*p+k,r] * W[k*n+i,r]
                # Then T[i,j,:] = U[i*m+j,:] @ A
                A = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        A[r, k] = V[j * p + k, r] * W[k * n + i, r]
                target = np.ones(p)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A.T, target, rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, n))
                for r in range(rank):
                    for i in range(n):
                        A[r, i] = U[i * m + j, r] * W[k * n + i, r]
                target = np.ones(n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A.T, target, rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                A = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        A[r, j] = U[i * m + j, r] * V[j * p + k, r]
                target = np.ones(m)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A.T, target, rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        error = _compute_error(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_error < tol:
            break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_error

def _try_rank_structured(n, m, p, target_rank, seed=42, time_limit=8.0):
    """Try to find a rank-`target_rank` decomposition using structured initialization."""
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    
    # Try multiple initialization strategies
    best_result = None
    best_error = float('inf')
    
    for init_strategy in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize U, V, W
        if init_strategy == 0:
            # Random small values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif init_strategy == 1:
            # Kronecker-like structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            # Try to exploit the all-ones structure
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] = 1.0 / (n * m) + np.random.randn() * 0.1
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r] = 1.0 / (m * p) + np.random.randn() * 0.1
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, r] = 1.0 / (p * n) + np.random.randn() * 0.1
        elif init_strategy == 2:
            # Block diagonal-like
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, r] = np.random.choice([-1, 1]) * 0.5
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, r] = np.random.choice([-1, 1]) * 0.5
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, r] = np.random.choice([-1, 1]) * 0.5
        elif init_strategy == 3:
            # Sparse initialization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for r in range(target_rank):
                # Random sparse patterns
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.5:
                            U[i * m + j, r] = random.uniform(-1, 1)
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j * p + k, r] = random.uniform(-1, 1)
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.5:
                            W[k * n + i, r] = random.uniform(-1, 1)
        else:
            # Gradient-like random
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
        
        # Refine using ALS
        U_ref, V_ref, W_ref, error = _als_refine(
            U.tolist(), V.tolist(), W.tolist(), 
            n, m, p, max_iter=300, tol=1e-10, time_limit=2.0
        )
        
        if error < best_error:
            best_error = error
            best_result = (U_ref, V_ref, W_ref)
        
        if best_error < 1e-8:
            break
    
    if best_result is not None and best_error < 1e-8:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    return None

def construct():
    n, m, p = 2, 5, 5
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition as fallback (rank 50)
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fallback
    
    # Try lower ranks from known lower bounds upward
    # Lower bound for 2x5x5 is at least max(2,5,5) = 5, but realistically around 15-20
    for target_rank in range(15, rank_fallback):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            result = _try_rank_structured(n, m, p, target_rank, seed=seed, time_limit=5.0)
            if result is not None:
                # Verify the result
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-7):
                    best_rank = target_rank
                    best_result = result
                    found = True
                    break
        
        if found:
            break  # Found a valid decomposition at this rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
