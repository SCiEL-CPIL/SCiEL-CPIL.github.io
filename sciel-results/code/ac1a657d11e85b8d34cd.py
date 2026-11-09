"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Deadline guard: evaluator kills after 150s, use 120s as safety margin
_DEADLINE = time.time() + 115.0

def _remaining_time():
    return _DEADLINE - time.time()

def _is_valid_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Fast vectorized check for valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Reconstructed tensor
    # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Efficient computation:
    # Temp1[a,b] = sum_r U[a,r] V[b,r] -> (n*m, m*p)
    # T_rec[a,b,c] = sum_r Temp1[a,b] * W[c,r] ... wait, this is not quite right.
    # T_rec = sum_r U[:,r] * V[:,r] * W[:,r]
    # Let M1 = U * V.T -> (n*m, m*p) where M1[a,b] = sum_r U[a,r]V[b,r]
    # Then T_rec[a,b,c] = sum_r (U[a,r]V[b,r]) W[c,r]
    # This is equivalent to: for each r, add outer(M1_r, W[:,r])? No.
    
    # Correct efficient way:
    # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = T_rec - T_target
    return np.max(np.abs(diff)) < tol

def _is_integer_or_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    arr = np.array(arr, dtype=float)
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _round_to_half_integer(arr, tol=1e-6):
    """Round entries to nearest half-integer."""
    arr = np.array(arr, dtype=float)
    return np.round(2.0 * arr) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Precompute target tensor for fast loss calculation
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
                
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    best_loss = np.max(np.abs(np.einsum('ar,br,cr->abc', best_U, best_V, best_W) - T_target))
    
    # Try to find lower rank decompositions
    # We focus on rank 23, the known upper bound.
    # We will use multiple random restarts and gradient descent with momentum.
    
    target_ranks = [23, 24, 25, 26]
    
    for target_rank in target_ranks:
        if _remaining_time() < 15:
            break
            
        # Number of restarts based on remaining time
        # Each restart takes ~10-50ms for 100 steps.
        # Let's allocate time budget.
        max_restarts = 50
        steps_per_restart = 200
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        found_valid = False
        
        for restart in range(max_restarts):
            if _remaining_time() < 5:
                break
            
            # Initialize with random values
            # Scale initialization to help convergence
            scale = 0.5
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Momentum buffers
            mU = np.zeros_like(U)
            mV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            current_loss = np.inf
            best_local_loss = np.inf
            best_local_U = U.copy()
            best_local_V = V.copy()
            best_local_W = W.copy()
            
            for step in range(steps_per_restart):
                if _remaining_time() < 5:
                    break
                    
                # Compute loss and gradients
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_rec - T_target
                loss = np.sum(diff**2)
                
                if loss < best_local_loss:
                    best_local_loss = loss
                    best_local_U = U.copy()
                    best_local_V = V.copy()
                    best_local_W = W.copy()
                    
                    if best_local_loss < 1e-10:
                        break
                        
                # Gradients
                dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Adam update
                t = step + 1
                mU = beta1 * mU + (1 - beta1) * dU
                mV = beta1 * mV + (1 - beta1) * dV
                mW = beta1 * mW + (1 - beta1) * dW
                
                vU = beta2 * vU + (1 - beta2) * (dU**2)
                vV = beta2 * vV + (1 - beta2) * (dV**2)
                vW = beta2 * vW + (1 - beta2) * (dW**2)
                
                mU_hat = mU / (1 - beta1**t)
                mV_hat = mV / (1 - beta1**t)
                mW_hat = mW / (1 - beta1**t)
                
                vU_hat = vU / (1 - beta2**t)
                vV_hat = vV / (1 - beta2**t)
                vW_hat = vW / (1 - beta2**t)
                
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                current_loss = loss
            
            # Check best local solution
            if best_local_loss < 1e-8:
                # Try to round to half-integers
                U_rounded = _round_to_half_integer(best_local_U)
                V_rounded = _round_to_half_integer(best_local_V)
                W_rounded = _round_to_half_integer(best_local_W)
                
                if _is_valid_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found_valid = True
                    break
                else:
                    # Check if unrounded is valid (should be, but check strictness)
                    if _is_valid_decomposition(best_local_U, best_local_V, best_local_W, n, m, p, tol=1e-9):
                         if _is_integer_or_half_integer(best_local_U) and _is_integer_or_half_integer(best_local_V) and _is_integer_or_half_integer(best_local_W):
                             best_U, best_V, best_W = _round_to_half_integer(best_local_U), _round_to_half_integer(best_local_V), _round_to_half_integer(best_local_W)
                             best_rank = target_rank
                             found_valid = True
                             break
                             
            if best_local_loss < best_loss:
                best_loss = best_local_loss
                # Don't update best_U/V/W unless valid, as we need exact decomposition
                # But we can track the best candidate for potential debugging or if we relax constraints later
                # For now, we only accept valid ones.
                
        if found_valid:
            break
            
    # If we found a better decomposition, use it
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
