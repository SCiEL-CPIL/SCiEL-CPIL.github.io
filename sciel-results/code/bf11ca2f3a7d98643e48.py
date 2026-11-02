import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0
    
    # Try to find a rank-24 decomposition
    for rank in [24, 25, 26, 27]:
        if time.time() > deadline:
            break
        if _find_decomposition(n, m, p, rank, deadline):
            U, V, W = _get_decomposition()
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 28
    return _strassen_28()

def _strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    u_coeffs = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    
    for k in range(4):
        base_v = 2 * k
        e_idx = base_v
        f_idx = base_v + 1
        g_idx = 8 + base_v
        h_idx = 9 + base_v
        
        base_w = 4 * k
        c11_idx = base_w
        c21_idx = base_w + 1
        c12_idx = base_w + 2
        c22_idx = base_w + 3
        
        for t in range(7):
            col = k * 7 + t
            
            for u_idx, val in enumerate(u_coeffs[t]):
                if val != 0:
                    U[u_idx][col] = val
            
            if t == 0:
                V[e_idx][col] = 1
                V[h_idx][col] = 1
            elif t == 1:
                V[e_idx][col] = 1
            elif t == 2:
                V[f_idx][col] = 1
                V[h_idx][col] = -1
            elif t == 3:
                V[g_idx][col] = 1
                V[e_idx][col] = -1
            elif t == 4:
                V[h_idx][col] = 1
            elif t == 5:
                V[e_idx][col] = 1
                V[f_idx][col] = 1
            elif t == 6:
                V[g_idx][col] = 1
                V[h_idx][col] = 1
            
            if t == 0:
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 1:
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
            elif t == 2:
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
            elif t == 3:
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
            elif t == 4:
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
            elif t == 5:
                W[c22_idx][col] = 1
            elif t == 6:
                W[c11_idx][col] = 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Reconstruct tensor
    # T[i, j, k] = sum_t U[t, i*m+j] * V[t, j*p+k] * W[t, k*n+i]
    # Check against target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                val = 0.0
                for t in range(rank):
                    val += U[t, i*m+j] * V[t, j*p+k] * W[t, k*n+i]
                if abs(val - target) > 1e-5:
                    return False
    return True

def _find_decomposition(n, m, p, rank, deadline):
    # Simple random search + gradient descent
    # This is a placeholder, needs implementation
    return False

def _get_decomposition():
    return None, None, None
