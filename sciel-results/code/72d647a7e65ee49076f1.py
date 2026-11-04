import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank p*m*n."""
    rank = p * m * n
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
    return U, V, W, rank

def _compute_C(U, V, W, A, B, n, m, p):
    """Compute C = sum_t W[:,t] * (U[:,t]·A) * (V[:,t]·B) using vectorized ops."""
    # U shape: (n*m, r), A shape: (n, m)
    # u_vals[t] = sum_{i,j} U[i*m+j, t] * A[i,j]
    u_vals = U.T @ A.flatten()  # shape: (r,)
    v_vals = V.T @ B.flatten()  # shape: (r,)
    
    # C[i,k] = sum_t W[k*n+i, t] * u_vals[t] * v_vals[t]
    # W shape: (p*n, r), we need to reshape W to (p, n, r) then transpose
    W_reshaped = W.reshape(p, n, r) if False else W.reshape(p, n, -1)  # (p, n, r)
    # W_reshaped[k, i, t] = W[k*n+i, t]
    
    # C[i, k] = sum_t W_reshaped[k, i, t] * u_vals[t] * v_vals[t]
    # = sum_t (W_reshaped[k, i, :]) * (u_vals * v_vals)
    uv = u_vals * v_vals  # (r,)
    # C_reshaped[k, i] = W_reshaped[k, i, :] @ uv
    C_reshaped = np.einsum('kit,t->ki', W_reshaped, uv)
    return C_reshaped.T  # (n, p)

def _residual(U, V, W, n, m, p):
    """Compute sum of squared residuals using fixed test matrices."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    C_decomp = _compute_C(U, V, W, A, B, n, m, p)
    return np.sum((C_true - C_decomp) ** 2)

def _residual_multi(U, V, W, n, m, p, A_list, B_list):
    """Compute residual over multiple test matrices for robustness."""
    total_res = 0.0
    for A, B in zip(A_list, B_list):
        C_true = A @ B
        C_decomp = _compute_C(U, V, W, A, B, n, m, p)
        total_res += np.sum((C_true - C_decomp) ** 2)
    return total_res

def _gradients(U, V, W, A, B, C_true, n, m, p):
    """Compute analytical gradients of residual w.r.t. U, V, W."""
    r = U.shape[1]
    u_vals = U.T @ A.flatten()  # (r,)
    v_vals = V.T @ B.flatten()  # (r,)
    uv = u_vals * v_vals  # (r,)
    
    W_reshaped = W.reshape(p, n, r)  # (p, n, r)
    C_decomp = np.einsum('kit,t->ki', W_reshaped, uv).T  # (n, p)
    
    error = C_true - C_decomp  # (n, p)
    
    # Gradient w.r.t. W:
    # C[i,k] = sum_t W[k*n+i, t] * uv[t]
    # dC[i,k]/dW[k*n+i, t] = uv[t]
    # dL/dW[k*n+i, t] = -error[i,k] * uv[t]
    # In reshaped form: dL/dW_reshaped[k,i,t] = -error[i,k] * uv[t]
    dW_reshaped = -error.T[:, :, np.newaxis] * uv[np.newaxis, np.newaxis, :]  # (p, n, r)
    dW = dW_reshaped.reshape(p * n, r)
    
    # Gradient w.r.t. U:
    # C[i,k] = sum_t W[k*n+i, t] * u_vals[t] * v_vals[t]
    # u_vals[t] = sum_{i',j'} U[i'*m+j', t] * A[i',j']
    # dC[i,k]/dU[i'*m+j', t] = W[k*n+i, t] * v_vals[t] * A[i',j']
    # dL/dU[i'*m+j', t] = -sum_{i,k} error[i,k] * W[k*n+i, t] * v_vals[t] * A[i',j']
    # = -v_vals[t] * A[i',j'] * sum_{i,k} error[i,k] * W[k*n+i, t]
    
    # Let's compute: for each t, sum_{i,k} error[i,k] * W[k*n+i, t]
    # W_reshaped[k,i,t] = W[k*n+i, t]
    # sum_{i,k} error[i,k] * W_reshaped[k,i,t] = einsum('ik,kit->t', error, W_reshaped)
    err_W = np.einsum('ik,kit->t', error, W_reshaped)  # (r,)
    
    # dL/dU[i'*m+j', t] = -v_vals[t] * A[i',j'] * err_W[t]
    A_flat = A.flatten()  # (n*m,)
    dU = -np.outer(A_flat, v_vals * err_W)  # (n*m, r)
    
    # Gradient w.r.t. V:
    # Similarly: dL/dV[j'*p+k', t] = -u_vals[t] * B[j',k'] * err_W[t]
    B_flat = B.flatten()  # (m*p,)
    dV = -np.outer(B_flat, u_vals * err_W)  # (m*p, r)
    
    return dU, dV, dW

def _snap_to_rationals(arr, n, m, p, V=None, W=None, U=None, max_tries=20):
    """Try to snap values to small rational numbers (integers or half-integers)."""
    arr = np.array(arr, dtype=float)
    
    def check_residual(U_arr, V_arr, W_arr):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_decomp = _compute_C(U_arr, V_arr, W_arr, A, B, n, m, p)
        return np.sum((C_true - C_decomp) ** 2)
    
    # Try snapping to integers, half-integers, quarter-integers
    best_arr = arr
    best_res = check_residual(U, V, W)
    
    for scale in [1, 2, 4, 8]:
        snapped = np.round(arr * scale) / scale
        res = check_residual(snapped, V, W)
        if res < best_res:
            best_res = res
            best_arr = snapped
    
    # Also try rounding to nearest multiple of 0.1, 0.25, etc.
    for denom in [10, 20, 40]:
        snapped = np.round(arr * denom) / denom
        res = check_residual(snapped, V, W)
        if res < best_res:
            best_res = res
            best_arr = snapped
    
    return best_arr

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Start with schoolbook
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    # Generate test matrices for verification
    np.random.seed(12345)
    A_list = [np.random.randn(n, m) for _ in range(5)]
    B_list = [np.random.randn(m, p) for _ in range(5)]
    
    # Try to find lower rank decompositions
    # Known lower bounds for 3x4x7: at least 3*4*7 - something
    # Try ranks from schoolbook-1 down to a reasonable minimum
    min_rank = 40
    
    for target_rank in range(rank_school - 1, min_rank - 1, -1):
        if time.time() > deadline - 40:
            break
        
        # Try multiple random initializations
        for trial in range(8):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 10000 + target_rank * 100 + 42)
            
            # Strategy 1: Initialize from schoolbook subset + perturbation
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            # Select random subset of schoolbook columns
            cols = np.random.choice(rank_school, size=target_rank, replace=False)
            U_init[:] = U_school[:, cols]
            V_init[:] = V_school[:, cols]
            W_init[:] = W_school[:, cols]
            
            # Add perturbation
            U_init += 0.1 * np.random.randn(n * m, target_rank)
            V_init += 0.1 * np.random.randn(m * p, target_rank)
            W_init += 0.1 * np.random.randn(p * n, target_rank)
            
            # Run gradient descent with adaptive learning rate
            lr = 0.05
            best_res = _residual_multi(U_init, V_init, W_init, n, m, p, A_list, B_list)
            prev_res = best_res
            
            for step in range(300):
                if time.time() > deadline - 25:
                    break
                
                # Use first test matrix for gradient
                A = A_list[0]
                B = B_list[0]
                C_true = A @ B
                
                dU, dV, dW = _gradients(U_init, V_init, W_init, A, B, C_true, n, m, p)
                
                # Normalize gradients to prevent explosion
                gnorm_U = np.linalg.norm(dU)
                gnorm_V = np.linalg.norm(dV)
                gnorm_W = np.linalg.norm(dW)
                gnorm = max(gnorm_U, gnorm_V, gnorm_W, 1e-10)
                
                # Update
                U_init -= lr * dU / gnorm * np.sqrt(gnorm_U)
                V_init -= lr * dV / gnorm * np.sqrt(gnorm_V)
                W_init -= lr * dW / gnorm * np.sqrt(gnorm_W)
                
                # Compute new residual
                new_res = _residual_multi(U_init, V_init, W_init, n, m, p, A_list, B_list)
                
                if new_res < best_res:
                    best_res = new_res
                    if new_res < 1e-12:
                        break
                    # Adaptive lr: increase if improving
                    lr = min(lr * 1.05, 0.5)
                else:
                    # Revert and decrease lr
                    U_init += lr * dU / gnorm * np.sqrt(gnorm_U)
                    V_init += lr * dV / gnorm * np.sqrt(gnorm_V)
                    W_init += lr * dW / gnorm * np.sqrt(gnorm_W)
                    lr *= 0.8
                    if lr < 1e-6:
                        break
                
                # Early stop if converged
                if new_res < 1e-12:
                    break
            
            # Check if we found a valid decomposition
            if best_res < 1e-10:
                # Verify with more test matrices
                final_res = _residual_multi(U_init, V_init, W_init, n, m, p, A_list, B_list)
                if final_res < 1e-10:
                    best_U = U_init.copy()
                    best_V = V_init.copy()
                    best_W = W_init.copy()
                    best_rank = target_rank
                    break
        
        # If we found a valid decomposition at this rank, stop
        if best_rank == target_rank:
            break
    
    # Final snap to rational values
    if best_rank < rank_school:
        # Try to snap U, V, W to cleaner values
        U_snapped = _snap_to_rationals(best_U, n, m, p, V=best_V, W=best_W)
        V_snapped = _snap_to_rationals(best_V, n, m, p, U=best_U, W=best_W)
        W_snapped = _snap_to_rationals(best_W, n, m, p, U=best_U, V=best_V)
        
        # Verify the snapped version
        final_res = _residual_multi(U_snapped, V_snapped, W_snapped, n, m, p, A_list, B_list)
        if final_res < 1e-8:
            best_U = U_snapped
            best_V = V_snapped
            best_W = W_snapped
    
    # Convert to lists
    best_U_list = [[float(x) for x in row] for row in best_U]
    best_V_list = [[float(x) for x in row] for row in best_V]
    best_W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
