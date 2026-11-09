import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank

def _compute_tensor_fast(U, V, W, n, m, p):
    U_reshaped = U.reshape(n, m, -1) 
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)

def _adam_step(params, grads, m_params, v_params, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Apply Adam update to parameters."""
    for i in range(len(params)):
        m_params[i] = beta1 * m_params[i] + (1 - beta1) * grads[i]
        v_params[i] = beta2 * v_params[i] + (1 - beta2) * grads[i] ** 2
        m_hat = m_params[i] / (1 - beta1 ** t)
        v_hat = v_params[i] / (1 - beta2 ** t)
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

def _try_adam_drop(U, V, W, n, m, p, max_iter=5000, tol=1e-5, deadline=None, lr=0.01):
    """
    Adam-based local search to recover the tensor after dropping a column.
    Uses MSE objective for smooth gradients.
    """
    rank = U.shape[1]
    T_target = np.ones((n, m, p))
    
    # Flatten parameters for Adam
    U_flat = U.copy().flatten()
    V_flat = V.copy().flatten()
    W_flat = W.copy().flatten()
    
    # Adam state
    m_U = np.zeros_like(U_flat)
    v_U = np.zeros_like(U_flat)
    m_V = np.zeros_like(V_flat)
    v_V = np.zeros_like(V_flat)
    m_W = np.zeros_like(W_flat)
    v_W = np.zeros_like(W_flat)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        U_curr = U_flat.reshape(U.shape)
        V_curr = V_flat.reshape(V.shape)
        W_curr = W_flat.reshape(W.shape)
        
        T_decomp = _compute_tensor_fast(U_curr, V_curr, W_curr, n, m, p)
        error_mse = np.mean((T_decomp - T_target) ** 2)
        error_max = np.max(np.abs(T_decomp - T_target))
        
        if error_max < tol:
            return U_curr, V_curr, W_curr, True
        
        if error_max < best_error:
            best_error = error_max
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        # Compute gradients using automatic differentiation via finite differences or analytical
        # For efficiency, use analytical gradients
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] for each k
        # dT/dV[j,k,c] = U[i,j,c] * W[k,i,c] for each i
        # dT/dW[k,i,c] = U[i,j,c] * V[j,k,c] for each j
        
        U_r = U_curr.reshape(n, m, -1)
        V_r = V_curr.reshape(m, p, -1)
        W_r = W_curr.reshape(p, n, -1)
        
        # Gradient w.r.t U: dU[i,j,c] = sum_k (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        grad_U = np.zeros_like(U_r)
        for k in range(p):
            # (T[:, :, k] - 1) shape (n, m)
            diff_k = (T_decomp[:, :, k] - 1.0)  # (n, m)
            # V_r[:, k, :] shape (m, rank)
            # W_r[k, :, :] shape (n, rank)
            # grad_U[:, :, c] += diff_k[:, i] * V_r[j, k, c] * W_r[k, i, c]
            # Vectorized: grad_U[i, j, c] = sum_k diff_k[i, j] * V_r[j, k, c] * W_r[k, i, c]
            for i in range(n):
                for j in range(m):
                    grad_U[i, j, :] += diff_k[i, j] * V_r[j, k, :] * W_r[k, i, :]
        
        # Gradient w.r.t V: dV[j,k,c] = sum_i (T[i,j,k] - 1) * U[i,j,c] * W[k,i,c]
        grad_V = np.zeros_like(V_r)
        for i in range(n):
            for k in range(p):
                diff_ik = (T_decomp[i, :, k] - 1.0)  # (m,)
                for j in range(m):
                    grad_V[j, k, :] += diff_ik[j] * U_r[i, j, :] * W_r[k, i, :]
        
        # Gradient w.r.t W: dW[k,i,c] = sum_j (T[i,j,k] - 1) * U[i,j,c] * V[j,k,c]
        grad_W = np.zeros_like(W_r)
        for i in range(n):
            for k in range(p):
                diff_ik = (T_decomp[i, :, k] - 1.0)  # (m,)
                for j in range(m):
                    grad_W[k, i, :] += diff_ik[j] * U_r[i, j, :] * V_r[j, k, :]
        
        # Apply Adam update
        t_adam = it + 1
        _adam_step([U_flat], [grad_U.flatten()], [m_U], [v_U], t_adam, lr)
        _adam_step([V_flat], [grad_V.flatten()], [m_V], [v_V], t_adam, lr)
        _adam_step([W_flat], [grad_W.flatten()], [m_W], [v_W], t_adam, lr)
    
    U_final = U_flat.reshape(U.shape)
    V_final = V_flat.reshape(V.shape)
    W_final = W_flat.reshape(W.shape)
    
    T_final = _compute_tensor_fast(U_final, V_final, W_final, n, m, p)
    if np.allclose(T_final, T_target, atol=tol):
        return U_final, V_final, W_final, True
    return best_U, best_V, best_W, False

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    # Try to drop 1 column (Rank 44)
    # We try dropping each column one by one
    for drop_idx in range(rank_sb):
        if time.time() > deadline - 20:
            break
            
        U_test = U_sb.copy()
        V_test = V_sb.copy()
        W_test = W_sb.copy()
        
        # Zero out the column
        U_test[:, drop_idx] = 0.0
        V_test[:, drop_idx] = 0.0
        W_test[:, drop_idx] = 0.0
        
        # Run Adam-based local search to recover
        U_res, V_res, W_res, success = _try_adam_drop(
            U_test, V_test, W_test, n, m, p, 
            max_iter=5000,
            deadline=deadline - 20,
            lr=0.01
        )
        
        if success:
            best_U, best_V, best_W, best_rank = U_res, V_res, W_res, rank_sb - 1
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
