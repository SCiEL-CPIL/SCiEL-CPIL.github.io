"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _is_half_int(x):
    """Check if x is a half-integer (multiple of 0.5)."""
    x2 = x * 2
    return abs(x2 - round(x2)) < 1e-10

def _make_schoolbook():
    """
    Construct the schoolbook decomposition of rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
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

def _try_rank_structured(r, deadline, rng):
    """
    Try to find a rank-r decomposition using structured random initialization.
    Uses the fact that for matrix multiplication, there are known constructions
    that can reduce rank below n*m*p.
    
    Strategy: Start with schoolbook and try to merge columns or use
    Strassen-like block decompositions where applicable.
    """
    n, m, p = DIMENSIONS
    # For 3x3 x 3x5, we can think of it as 3x3 x 3x3 + 3x3 x 3x2
    # The 3x3 x 3x3 part has known rank 23 (Strassen) or even lower
    # But for exact half-integer, Strassen uses integers
    
    # Let's try a direct optimization approach
    # Initialize with random half-integer values
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    total_size = size_U + size_V + size_W
    
    # Random initialization with small values
    U_init = rng.uniform(-1, 1, size_U)
    V_init = rng.uniform(-1, 1, size_V)
    W_init = rng.uniform(-1, 1, size_W)
    
    # Try to solve using iterative refinement
    # This is a non-convex problem, so we use multiple restarts
    best_loss = float('inf')
    best_params = None
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create initial guess
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Reshape
        U_mat = U.reshape(n * m, r)
        V_mat = V.reshape(m * p, r)
        W_mat = W.reshape(p * n, r)
        
        # Compute current loss
        def compute_loss():
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(r):
                            total += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                        loss += (total - 1) ** 2
            return loss
        
        # Simple gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients numerically
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            # Sample a few (i,j,k) tuples for gradient estimation
            samples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            for idx, (i, j, k) in enumerate(samples):
                if idx % 3 == 0:  # Sample 1/3 for speed
                    for c in range(r):
                        # Gradient w.r.t. U[i*m+j, c]
                        dU = V_mat[j * p + k, c] * W_mat[k * n + i, c]
                        U_grad[i * m * r + j * r + c] += 2 * (U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c] - 1) * dU / len(samples)
                        
                        # Gradient w.r.t. V[j*p+k, c]
                        dV = U_mat[i * m + j, c] * W_mat[k * n + i, c]
                        V_grad[j * p * r + k * r + c] += 2 * (U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c] - 1) * dV / len(samples)
                        
                        # Gradient w.r.t. W[k*n+i, c]
                        dW = U_mat[i * m + j, c] * V_mat[j * p + k, c]
                        W_grad[k * n * r + i * r + c] += 2 * (U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c] - 1) * dW / len(samples)
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            U_mat = U.reshape(n * m, r)
            V_mat = V.reshape(m * p, r)
            W_mat = W.reshape(p * n, r)
            
            # Check loss
            current_loss = compute_loss()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = (U.copy(), V.copy(), W.copy())
            
            if current_loss < 1e-6:
                break
        
        # Reshuffle for next restart
        U_init = rng.uniform(-1, 1, size_U)
        V_init = rng.uniform(-1, 1, size_V)
        W_init = rng.uniform(-1, 1, size_W)
    
    # If we found a good solution, round to half-integers and verify
    if best_params is not None and best_loss < 0.1:
        U_best, V_best, W_best = best_params
        U_rounded = np.round(U_best * 2) / 2
        V_rounded = np.round(V_best * 2) / 2
        W_rounded = np.round(W_best * 2) / 2
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin (150s limit)
    
    # Get schoolbook solution as fallback
    result = _make_schoolbook()
    assert _verify(result["U"], result["V"], result["W"]), "Schoolbook verification failed"
    
    # Try to find lower rank decompositions
    rng = np.random.default_rng(12345)
    
    # Try ranks from 44 down, but limit attempts to stay within time budget
    # For 3x3 x 3x5, known results suggest rank around 20-30 might be possible
    # but exact half-integer solutions are hard to find
    
    # Try a few specific ranks that are more likely to have solutions
    ranks_to_try = [40, 35, 30, 25, 23]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        candidate = _try_rank_structured(r, deadline, rng)
        if candidate is not None:
            result = candidate
            break
    
    return result
# EVOLVE-BLOCK-END
