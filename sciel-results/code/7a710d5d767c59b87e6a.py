import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 8]

# ... (keep existing functions) ...

_TARGET_TENSOR_444 = None

def _get_target_tensor_444():
    global _TARGET_TENSOR_444
    if _TARGET_TENSOR_444 is None:
        _TARGET_TENSOR_444 = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i*4 + j
                    b = j*4 + k
                    c = i*4 + k
                    _TARGET_TENSOR_444[a, b, c] = 1.0
    return _TARGET_TENSOR_444

def _try_rank_19_decomposition(deadline):
    R = 19
    target = _get_target_tensor_444()
    
    def loss(x):
        U_flat = x[:R*16].reshape(R, 16)
        V_flat = x[R*16:2*R*16].reshape(R, 16)
        W_flat = x[2*R*16:].reshape(R, 16)
        
        T = np.einsum('ar, br, cr -> abc', U_flat, V_flat, W_flat)
        return np.sum((T - target)**2)

    # Try a few random restarts
    best_x = None
    best_loss = np.inf
    
    for seed in range(5):
        if time.time() > deadline - 5:
            break
        np.random.seed(seed)
        x0 = np.random.randn(R*48) * 0.5
        result = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 50})
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x
            
        if best_loss < 1e-8:
            break
            
    if best_loss < 1e-6:
        U_flat = best_x[:R*16].reshape(R, 16)
        V_flat = best_x[R*16:2*R*16].reshape(R, 16)
        W_flat = best_x[2*R*16:].reshape(R, 16)
        
        U = U_flat.T.tolist()
        V = V_flat.T.tolist()
        W = W_flat.T.tolist()
        
        # Permute W to k*4+i
        W_perm = [[0]*R for _ in range(16)]
        for i in range(4):
            for k in range(4):
                for r in range(R):
                    W_perm[k*4+i][r] = W[i*4+k][r]
                    
        # Verify
        if _verify_decomposition(U, V, W_perm, 4, 4, 4):
            return U, V, W_perm
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U_444, V_444, W_444 = None, None, None
    if time.time() < deadline - 20:
        U_444, V_444, W_444 = _try_rank_19_decomposition(deadline)
        
    if U_444 is not None:
        rank_total = 38
        U = [[0] * rank_total for _ in range(16)]
        V = [[0] * rank_total for _ in range(32)]
        W = [[0] * rank_total for _ in range(32)]
        
        for r in range(19):
            for i in range(16):
                U[i][r] = U_444[i][r]
            for j in range(4):
                for k in range(4):
                    V[j*8+k][r] = V_444[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r] = W_444[k*4+i][r]
                    
            r2 = 19 + r
            for i in range(16):
                U[i][r2] = U_444[i][r]
            for j in range(4):
                for k in range(4):
                    V[j*8+(4+k)][r2] = V_444[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    W[(4+k)*4+i][r2] = W_444[k*4+i][r]
                    
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to existing 98 rank
    U_444_s, V_444_s, W_444_s = _build_4x4x4_decomposition()
    # ... (construct 98 rank) ...
    # ...
    
    # Schoolbook fallback
    # ...

