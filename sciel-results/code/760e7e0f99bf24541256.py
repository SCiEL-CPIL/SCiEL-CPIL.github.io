import time
import numpy as np

DIMENSIONS = [3, 3, 5]

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

def _round_to_half_integer(A):
    return np.round(A * 2.0) / 2.0

def _try_drop(U_init, V_init, W_init, n, m, p, max_iter=2000, tol=1e-5, deadline=None, lr=0.05):
    rank = U_init.shape[1]
    T_target = np.ones((n, m, p))
    
    U_flat = U_init.copy().flatten()
    V_flat = V_init.copy().flatten()
    W_flat = W_init.copy().flatten()
    
    m_U = np.zeros_like(U_flat)
    v_U = np.zeros_like(U_flat)
    m_V = np.zeros_like(V_flat)
    v_V = np.zeros_like(V_flat)
    m_W = np.zeros_like(W_flat)
    v_W = np.zeros_like(W_flat)
    
    best_error = np.inf
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        U_curr = U_flat.reshape(U_init.shape)
        V_curr = V_flat.reshape(V_init.shape)
        W_curr = W_flat.reshape(W_init.shape)
        
        T_decomp = _compute_tensor_fast(U_curr, V_curr, W_curr, n, m, p)
        error_max = np.max(np.abs(T_decomp - T_target))
        
        if error_max < tol:
            # Round to half-integers and verify
            U_r = _round_to_half_integer(U_curr)
            V_r = _round_to_half_integer(V_curr)
            W_r = _round_to_half_integer(W_curr)
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r, True
            # If rounding fails, keep floating point? The problem requires integer/half-integer.
            # So we must return rounded values if they work.
            # If they don't work, we continue searching.
            
        if error_max < best_error:
            best_error = error_max
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        # Vectorized Gradients
        U_r = U_curr.reshape(n, m, -1)
        V_r = V_curr.reshape(m, p, -1)
        W_r = W_curr.reshape(p, n, -1)
        
        diff = T_decomp - 1.0
        
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        t_adam = it + 1
        # Adam step
        m_U = 0.9 * m_U + 0.1 * grad_U.flatten()
        v_U = 0.999 * v_U + 0.001 * grad_U.flatten()**2
        m_U_hat = m_U / (1 - 0.9**t_adam)
        v_U_hat = v_U / (1 - 0.999**t_adam)
        U_flat -= lr * m_U_hat / (np.sqrt(v_U_hat) + 1e-8)
        
        m_V = 0.9 * m_V + 0.1 * grad_V.flatten()
        v_V = 0.999 * v_V + 0.001 * grad_V.flatten()**2
        m_V_hat = m_V / (1 - 0.9**t_adam)
        v_V_hat = v_V / (1 - 0.999**t_adam)
        V_flat -= lr * m_V_hat / (np.sqrt(v_V_hat) + 1e-8)
        
        m_W = 0.9 * m_W + 0.1 * grad_W.flatten()
        v_W = 0.999 * v_W + 0.001 * grad_W.flatten()**2
        m_W_hat = m_W / (1 - 0.9**t_adam)
        v_W_hat = v_W / (1 - 0.999**t_adam)
        W_flat -= lr * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)
    
    U_final = U_flat.reshape(U_init.shape)
    V_final = V_flat.reshape(V_init.shape)
    W_final = W_flat.reshape(W_init.shape)
    
    U_r = _round_to_half_integer(U_final)
    V_r = _round_to_half_integer(V_final)
    W_r = _round_to_half_integer(W_final)
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
        return U_r, V_r, W_r, True
    return best_U, best_V, best_W, False

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    # Try to drop 4 columns (Rank 41)
    # We try dropping the last 4 columns
    drop_indices = list(range(rank_sb - 4, rank_sb))
    U_test = U_sb.copy()
    V_test = V_sb.copy()
    W_test = W_sb.copy()
    U_test[:, drop_indices] = 0.0
    V_test[:, drop_indices] = 0.0
    W_test[:, drop_indices] = 0.0
    
    U_res, V_res, W_res, success = _try_drop(
        U_test, V_test, W_test, n, m, p, 
        max_iter=5000,
        deadline=deadline - 10,
        lr=0.05
    )
    
    if success:
        best_U, best_V, best_W, best_rank = U_res, V_res, W_res, rank_sb - 4
    else:
        # Try dropping 3 columns (Rank 42)
        drop_indices = list(range(rank_sb - 3, rank_sb))
        U_test = U_sb.copy()
        V_test = V_sb.copy()
        W_test = W_sb.copy()
        U_test[:, drop_indices] = 0.0
        V_test[:, drop_indices] = 0.0
        W_test[:, drop_indices] = 0.0
        
        U_res, V_res, W_res, success = _try_drop(
            U_test, V_test, W_test, n, m, p, 
            max_iter=5000,
            deadline=deadline - 10,
            lr=0.05
        )
        
        if success:
            best_U, best_V, best_W, best_rank = U_res, V_res, W_res, rank_sb - 3
        else:
            # Try dropping 1 column (Rank 44) as fallback
            for drop_idx in range(rank_sb):
                if time.time() > deadline - 5:
                    break
                U_test = U_sb.copy()
                V_test = V_sb.copy()
                W_test = W_sb.copy()
                U_test[:, drop_idx] = 0.0
                V_test[:, drop_idx] = 0.0
                W_test[:, drop_idx] = 0.0
                
                U_res, V_res, W_res, success = _try_drop(
                    U_test, V_test, W_res, n, m, p, 
                    max_iter=2000,
                    deadline=deadline - 5,
                    lr=0.05
                )
                
                if success:
                    best_U, best_V, best_W, best_rank = U_res, V_res, W_res, rank_sb - 1
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
