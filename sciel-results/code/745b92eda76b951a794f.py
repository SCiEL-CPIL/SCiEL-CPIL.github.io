"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _strassen_decomposition():
    """Return the base Strassen rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    for b_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        for s_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            col = b_idx * 7 + s_idx
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                j = a_col_off + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                j = b_row_off + li
                r = b_col_off + lj
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                r = c_col_off + lj
                g_idx = r * 2 + i
                W[g_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # should equal 1 if i corresponds to (a_row, a_col), j to (b_row, b_col), 
    # k to (c_row, c_col) with a_col == b_row and a_row == c_row, else 0.
    
    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for a_row in range(n):
        for b_col in range(p):
            for a_col in range(m):
                # T[(a_row, a_col), (a_col, b_col), (b_col, a_row)] = 1
                i = a_row * m + a_col
                j = a_col * p + b_col
                k = b_col * n + a_row
                T_target[i, j, k] = 1
    
    # Compute the tensor from decomposition
    T_decomp = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    diff = np.max(np.abs(T_decomp - T_target))
    return diff < tol

def _try_reduce_rank(base_result, max_rank_to_try=27, time_limit=100.0):
    """Try to find a lower rank decomposition by perturbing and optimizing."""
    start_time = time.time()
    best_result = base_result
    best_rank = base_result["U"].__len__() if False else len(base_result["U"][0])
    
    n, m, p = DIMENSIONS
    U0 = np.array(base_result["U"], dtype=float)
    V0 = np.array(base_result["V"], dtype=float)
    W0 = np.array(base_result["W"], dtype=float)
    
    # Try ranks from best_rank-1 down to max_rank_to_try
    for target_rank in range(best_rank - 1, max_rank_to_try - 1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Strategy: Try to merge pairs of Strassen terms that share structure
        # The four blocks each use 7 terms. If we can find linear dependencies
        # across blocks, we can reduce rank.
        
        # Approach: Form the tensor and try to find a lower-rank decomposition
        # by solving a system of equations.
        
        # Build the tensor T as a 3D array
        T = np.zeros((n * m, m * p, p * n))
        for a_row in range(n):
            for b_col in range(p):
                for a_col in range(m):
                    i = a_row * m + a_col
                    j = a_col * p + b_col
                    k = b_col * n + a_row
                    T[i, j, k] = 1.0
        
        # Try to find U, V, W with target_rank columns such that
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        
        # This is a non-convex problem. Use a simple approach:
        # Fix W to be the identity-like structure and solve for U, V.
        # Or use the fact that for fixed W, the problem becomes linear in U and V.
        
        # Simpler approach: Try random initializations and use gradient descent
        # on the tensor reconstruction error.
        
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to match the norm of the target tensor
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute T_decomp
                T_decomp = np.zeros((n * m, m * p, p * n))
                for r in range(target_rank):
                    T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Compute error
                error = T_decomp - T
                
                # Check convergence
                max_err = np.max(np.abs(error))
                if max_err < 1e-8:
                    # Found a valid decomposition
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Compute gradients
                # d/dU[i,r] = sum_{j,k} error[i,j,k] * V[j,r] * W[k,r]
                # d/dV[j,r] = sum_{i,k} error[i,j,k] * U[i,r] * W[k,r]
                # d/dW[k,r] = sum_{i,j} error[i,j,k] * U[i,r] * V[j,r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(target_rank):
                    # dU[:, r] = sum_{j,k} error[:, j, k] * V[j, r] * W[k, r]
                    # = (error @ (V[:, r] * W[:, r])) but need to be careful with dimensions
                    # error is (n*m, m*p, p*n), V[:, r] is (m*p,), W[:, r] is (p*n,)
                    # We need: for each i, dU[i, r] = sum_j sum_k error[i,j,k] * V[j,r] * W[k,r]
                    # = sum_j error[i,j,:] dot (W[:,r] * V[j,r]) ... this is expensive
                    
                    # Vectorized: dU[:, r] = np.einsum('ijk,j,k->i', error, V[:, r], W[:, r])
                    dU[:, r] = np.einsum('ijk,j,k->i', error, V[:, r], W[:, r])
                    dV[:, r] = np.einsum('ijk,i,k->j', error, U[:, r], W[:, r])
                    dW[:, r] = np.einsum('ijk,i,j->k', error, U[:, r], V[:, r])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # After optimization, check if we converged
            T_decomp = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            if np.max(np.abs(T_decomp - T)) < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() - start_time > time_limit:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin
    
    # Start with the known rank-28 Strassen decomposition
    base_result = _strassen_decomposition()
    
    # Verify it's correct
    if not _verify_decomposition(base_result["U"], base_result["V"], base_result["W"]):
        # Should never happen, but handle gracefully
        return base_result
    
    # Try to find a lower rank decomposition
    improved = _try_reduce_rank(base_result, max_rank_to_try=24, time_limit=deadline - (time.time() - start_time) - 5.0)
    
    if time.time() > deadline:
        return base_result
    
    # Verify the improved result if different
    if improved is not base_result:
        if _verify_decomposition(improved["U"], improved["V"], improved["W"]):
            return improved
    
    return base_result
# EVOLVE-BLOCK-END
