import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensors: U[i,j,c], V[j,k,c], W[k,i,c]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _try_random_search(n, m, p, target_rank, num_trials, deadline, best_rank):
    """Try random initializations and local refinement to find valid decompositions."""
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize to keep values reasonable
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Quick check: compute the tensor and see how close to all-ones we are
        U_tensor = U.reshape(n, m, target_rank)
        V_tensor = V.reshape(m, p, target_rank)
        W_tensor = W.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
        
        # If we're close to the target, try local refinement
        error = np.max(np.abs(T - 1.0))
        if error < 0.5:  # Reasonably close
            # Try a few steps of gradient descent
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients (approximate)
                # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
                # This is expensive, so use a simple heuristic instead
                
                # Add small random perturbations
                U += 0.01 * np.random.randn(n * m, target_rank)
                V += 0.01 * np.random.randn(m * p, target_rank)
                W += 0.01 * np.random.randn(p * n, target_rank)
                
                # Re-normalize
                U /= np.sqrt(target_rank)
                V /= np.sqrt(target_rank)
                W /= np.sqrt(target_rank)
                
                # Check error
                U_tensor = U.reshape(n, m, target_rank)
                V_tensor = V.reshape(m, p, target_rank)
                W_tensor = W.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
                error = np.max(np.abs(T - 1.0))
                
                if error < 1e-9:
                    return U, V, W
    
    return None

def _try_structured_patterns(n, m, p, deadline, best_rank):
    """Try various structured patterns that might give lower rank."""
    # Pattern 1: Try rank = n*m + m*p - n (for square case, this is n^2 + n^2 - n = 2n^2 - n)
    # For 4x5x5: 20 + 25 - 4 = 41 (but this formula is for square matrices)
    
    # Pattern 2: Try rank = n*p + m*p - m
    # For 4x5x5: 20 + 25 - 5 = 40
    
    # Pattern 3: Try rank = n*m + n*p - n
    # For 4x5x5: 20 + 20 - 4 = 36
    
    # These are heuristics based on known results for square matrices
    # For non-square, we need to be more careful
    
    candidate_ranks = [99, 98, 97, 96, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random search for this rank
        result = _try_random_search(n, m, p, target_rank, 20, deadline, best_rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for safety
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions
    result = _try_structured_patterns(n, m, p, deadline, best_rank)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = U_try.shape[1]
    
    # If time remains, try a few more random searches with different seeds
    if time.time() < deadline - 20:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            result = _try_random_search(n, m, p, best_rank - 1, 10, deadline, best_rank)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = U_try.shape[1]
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
