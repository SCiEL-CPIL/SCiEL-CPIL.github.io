"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _is_valid(U, V, W, n, m, p, tol=1e-6):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to tensors
    # U: (n*m, rank) -> (n, m, rank)
    Ut = Ua.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    Vt = Va.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    Wt = Wa.reshape(p, n, rank)
    
    # Compute tensor: T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum
    T = np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)
    
    # Expected tensor: E[i, j, k] = 1 if the indices correspond to a valid multiplication
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T[i, j, k] should be 1 for all valid (i, j, k) in the multiplication
    # Actually, the tensor for matrix multiplication has T[i, j, k] = 1 for all i, j, k
    # because each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # The correct tensor for matrix multiplication is all ones
    E = np.ones((n, m, p))
    
    return np.allclose(T, E, atol=tol, rtol=tol)

def _try_rank(rank, n, m, p, max_iter=100, seed=None):
    """Try to find a rank-`rank` decomposition using iterative refinement."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Reshape
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    
    # Target tensor
    T_target = np.ones((n, m, p))
    
    for it in range(max_iter):
        # Compute current tensor
        T_cur = np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)
        error = T_cur - T_target
        
        if np.max(np.abs(error)) < 1e-8:
            # Found a solution
            return U, V, W
        
        # Gradient-based update
        # dU[i,j,r] = sum_{k} V[j,k,r] * W[k,i,r] * error[i,j,k]
        # dV[j,k,r] = sum_{i} U[i,j,r] * W[k,i,r] * error[i,j,k]
        # dW[k,i,r] = sum_{j} U[i,j,r] * V[j,k,r] * error[i,j,k]
        
        dU = np.einsum('jkr,kir,ijk->ijr', Vt, Wt, error)
        dV = np.einsum('ijr,kir,ijk->jkr', Ut, Wt, error)
        dW = np.einsum('ijr,jkr,ijk->kir', Ut, Vt, error)
        
        # Simple gradient descent
        lr = 0.01
        U -= lr * dU.reshape(n * m, rank)
        V -= lr * dV.reshape(m * p, rank)
        W -= lr * dW.reshape(p * n, rank)
        
        Ut = U.reshape(n, m, rank)
        Vt = V.reshape(m, p, rank)
        Wt = W.reshape(p, n, rank)
    
    # Check if we converged
    T_final = np.einsum('ijr,jkr,kir->ijk', Ut, Vt, Wt)
    if np.max(np.abs(T_final - T_target)) < 1e-6:
        return U, V, W
    
    return None

def _round_to_half_int(arr):
    """Round array values to nearest half-integer."""
    return np.round(arr * 2) / 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with the known schoolbook decomposition
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Known lower bounds and bounds for 4x4x9
    # The trivial lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But we can do better with clever decompositions
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For 4x4x9, let's try from 36 upward
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few ranks below schoolbook
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = None
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank(rank, n, m, p, max_iter=50, seed=rank * 100 + attempt)
            if result is not None:
                U, V, W = result
                # Round to half-integers
                U_rounded = _round_to_half_int(U)
                V_rounded = _round_to_half_int(V)
                W_rounded = _round_to_half_int(W)
                
                # Check if rounded version is still valid
                if _is_valid(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    found = (U_rounded, V_rounded, W_rounded)
                    break
                else:
                    # Try unrounded
                    if _is_valid(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        found = (U, V, W)
                        break
        
        if found is not None:
            U, V, W = found
            best_result = {
                "U": U.tolist() if hasattr(U, 'tolist') else U,
                "V": V.tolist() if hasattr(V, 'tolist') else V,
                "W": W.tolist() if hasattr(W, 'tolist') else W
            }
            best_rank = rank
            break  # Found the lowest rank, stop
    
    return best_result

# EVOLVE-BLOCK-END
