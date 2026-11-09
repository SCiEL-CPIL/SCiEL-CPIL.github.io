"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

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

def _compute_error(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _polish(U, V, W, deadline, max_iter=100):
    """High-precision polish with adaptive learning rate."""
    n, m, p = DIMENSIONS
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = 0.01
    best_err = _compute_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        residuals = vals - 1.0
        
        VW = V_sel * W_sel
        UW = U_sel * W_sel
        UV = U_sel * V_sel
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        np.add.at(dU, idx_u, residuals[:, None] * VW)
        np.add.at(dV, idx_v, residuals[:, None] * UW)
        np.add.at(dW, idx_w, residuals[:, None] * UV)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err = _compute_error(U, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
        
        if it % 20 == 0 and it > 0:
            lr *= 0.7
    
    return best_U, best_V, best_W, best_err

def _try_rank(target_rank, num_attempts=5, max_iter=200, deadline=None):
    """Try to find a rank decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 15:
            return None
        
        np.random.seed(attempt * 104729 + target_rank)
        
        # Initialize with small random values
        scale = 1.0 / np.sqrt(target_rank)
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.random((p*n, target_rank)) * scale + 0.5
        
        lr = 0.1
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 15:
                break
            
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            
            vals = (U_sel * V_sel * W_sel).sum(axis=1)
            residuals = vals - 1.0
            
            VW = V_sel * W_sel
            UW = U_sel * W_sel
            UV = U_sel * V_sel
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = _compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
            
            if iteration % 30 == 0 and iteration > 0:
                lr *= 0.85
        
        if best_err < 1e-8:
            # Polish
            best_U, best_V, best_W, best_err = _polish(best_U, best_V, best_W, deadline, max_iter=50)
            
            if best_err < 1e-10:
                # Try to round to half-integers
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                err_r = _compute_error(U_r, V_r, W_r)
                if err_r < 1e-10:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 80 to 119
    for target_rank in range(80, 120):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(target_rank, num_attempts=3, max_iter=150, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
