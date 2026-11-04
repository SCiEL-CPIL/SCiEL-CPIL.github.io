"""Improved tensor decomposition search for 3x4 * 4x7 matrix multiplication."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # The tensor T[i, j, k] = delta (1 if A[i,j]*B[j,k] contributes to C[i,k])
    # For matrix multiplication: T[i, j, k] = 1 for all i,j,k (it's the identity tensor in the sense that
    # the multiplication tensor has T[i,j,k] = 1)
    # Actually, the matrix multiplication tensor T is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k in the standard basis decomposition
    # A rank-r decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k]
    
    # Known bounds: rank(3,4,7) is between some lower bound and 84 (schoolbook)
    # Lower bound: max(3*4 + 4*7 - 7, 3*4 + 3*7 - 3, 4*7 + 3*7 - 4) 
    # = max(12+28-7, 12+21-3, 28+21-4) = max(33, 30, 45) = 45
    # Actually the Strassen lower bound / border rank considerations give various bounds.
    # Let's try to find a good decomposition.
    
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 84
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try random search for lower rank decompositions
    # Use a heuristic: try various ranks and use randomized optimization
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact (up to tolerance)."""
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-6):
        """Fast verification using tensor contraction."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Build the tensor T from U, V, W
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum for speed
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(T, 1.0, atol=tol)
    
    # Random search with local optimization
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from a lower bound upward
    # Lower bound is at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also the "sum of dimensions" lower bound: n+m+p - something
    # Let's try ranks from 35 to 83
    
    ranks_to_try = list(range(83, 34, -1))  # Try from high to low, stop early if found
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Gradient-free optimization: use a simple iterative approach
            # Minimize ||T(U,V,W) - T_target||^2 where T_target[i,j,k] = 1
            
            T_target = np.ones((n, m, p))
            
            # Simple iterative refinement
            best_err = np.inf
            for iteration in range(50):
                if time.time() > deadline - 20:
                    break
                    
                T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = np.sum((T - T_target)**2)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-12:
                        break
                
                if err < 1e-12:
                    break
                
                # Compute gradients
                # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] * (T[i,j,k] - 1)
                # dT/dV[j,k,c] = U[i,j,c] * W[k,i,c] * (T[i,j,k] - 1)
                # dT/dW[k,i,c] = U[i,j,c] * V[j,k,c] * (T[i,j,k] - 1)
                
                diff = T - T_target
                
                dU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
                dV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
                dW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
                
                # Adaptive step size
                lr = 0.01 / (1 + iteration * 0.1)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we found a valid decomposition
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = np.sum((T - T_target)**2)
            
            if err < 1e-10:
                # Try to round to nearest half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_r = np.einsum('ijc,jkc,kic->ijk', U_rounded, V_rounded, W_rounded)
                err_r = np.sum((T_r - T_target)**2)
                
                if err_r < 1e-9:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    break
                else:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
            elif err < 0.01:
                # Close but not exact - try local polish
                for polish_iter in range(100):
                    if time.time() > deadline - 20:
                        break
                    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                    err = np.sum((T - T_target)**2)
                    if err < 1e-12:
                        break
                    diff = T - T_target
                    dU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
                    dV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
                    dW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
                    lr = 0.001
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = np.sum((T - T_target)**2)
                if err < 1e-10:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    T_r = np.einsum('ijc,jkc,kic->ijk', U_rounded, V_rounded, W_rounded)
                    err_r = np.sum((T_r - T_target)**2)
                    if err_r < 1e-9:
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
    
    # Also try some structured decompositions
    if time.time() < deadline - 30 and best_rank == rank_school:
        # Try Strassen-like block decomposition ideas
        # For 3x4 * 4x7, we can try to exploit structure
        pass
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Clean up: convert to int or keep as float
    def clean_matrix(M):
        result = []
        for row in M:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    clean_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    clean_row.append(round(val * 2) / 2)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    U_list = clean_matrix(U_list)
    V_list = clean_matrix(V_list)
    W_list = clean_matrix(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
