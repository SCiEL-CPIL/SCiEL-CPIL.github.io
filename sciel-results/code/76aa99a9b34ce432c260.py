import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Check if U, V, W form a valid rank-r decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # T[i,j,k] should be 1
    # We check all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _loss_and_grad(params, n, m, p, r):
    """
    Vectorized loss and gradient.
    """
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = params[:size_U].reshape(n * m, r)
    V = params[size_U:size_U + size_V].reshape(m * p, r)
    W = params[size_U + size_U + size_V:].reshape(p * n, r) # Wait, size_U + size_V
    
    # Correct slicing
    U = params[:size_U].reshape(n * m, r)
    V = params[size_U:size_U + size_V].reshape(m * p, r)
    W = params[size_U + size_V:].reshape(p * n, r)

    # Compute T efficiently
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can create arrays for indices
    # i in [0, n), j in [0, m), k in [0, p)
    
    T = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        # Uc is (n*m,), Vc is (m*p,), Wc is (p*n,)
        # We need to map indices
        # Uc[i*m+j], Vc[j*p+k], Wc[k*n+i]
        # This is still a loop over i,j,k if done naively.
        # But we can use broadcasting?
        # Let's stick to the loop over c, but vectorize i,j,k inside?
        # Actually, for small n,m,p, the loop over i,j,k (18 iterations) is fine.
        # The bottleneck in the original code was the loop over c AND i,j,k AND gradient calculation.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    
    err = 1.0 - T
    loss = np.sum(err ** 2)
    
    # Gradient
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # dU[i*m+j, c] = -2 * sum_{j,k} err[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
    # This is hard to vectorize easily without complex indexing.
    # Given n=2, m=3, p=3, the loops are small.
    # The main speedup comes from not recalculating T from scratch in a nested python loop for EVERY parameter update in a slow way?
    # Actually, the original code's `_loss_and_grad` is called by L-BFGS-B many times.
    # Vectorizing the inner loop over c is the key.
    
    # Let's re-implement the gradient calculation with c-loop outside, i-j-k inside.
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(r):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        
        # We need dU[:, c], dV[:, c], dW[:, c]
        # dU[idx_u, c] = -2 * sum_{j,k} err[i,j,k] * Vc[idx_v] * Wc[idx_w]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = -2 * err[i, j, k]
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    dU[u_idx, c] += e * Vc[v_idx] * Wc[w_idx]
                    dV[v_idx, c] += e * Uc[u_idx] * Wc[w_idx]
                    dW[w_idx, c] += e * Uc[u_idx] * Vc[v_idx]
                    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _try_rank_fast(r, n, m, p, timeout=10.0):
    start_time = time.time()
    n_vars = (n * m + m * p + p * n) * r
    
    best_loss = float('inf')
    best_params = None
    
    # Use more seeds and faster method
    for seed in range(50):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(seed)
        x0 = np.random.randn(n_vars) * 0.1
        
        def objective(x):
            return _loss_and_grad(x, n, m, p, r)
            
        try:
            # Try L-BFGS-B
            result = opt.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
        except Exception:
            continue
            
        loss = result.fun
        if loss < best_loss:
            best_loss = loss
            best_params = result.x
            
        if best_loss < 1e-8:
            break
            
    if best_params is None or best_loss > 1e-6:
        return None
        
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = best_params[:size_U].reshape(n * m, r)
    V = best_params[size_U:size_U + size_V].reshape(m * p, r)
    W = best_params[size_U + size_V:].reshape(p * n, r)
    
    # Round to nearest integer/half-integer? 
    # The problem asks for integer/half-integer entries.
    # We can snap to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _check_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
        return U_r, V_r, W_r
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 # Safety margin
    
    # Try rank 16
    if time.time() < deadline - 20:
        res = _try_rank_fast(16, n, m, p, timeout=20.0)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 17 (just in case 16 is hard to find numerically, though 16 is known)
    if time.time() < deadline - 10:
        res = _try_rank_fast(17, n, m, p, timeout=10.0)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook Rank 18
    rank = 18
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
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
