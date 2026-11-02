import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                T_recon[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def find_224_decomp(rank_target, iterations=50000):
    np.random.seed(42)
    for _ in range(iterations):
        if time_up():
            break
        U = np.random.randn(4, rank_target)
        V = np.random.randn(8, rank_target)
        W = np.random.randn(8, rank_target)
        
        # Fast check using einsum or similar
        # T[i,j,k] = sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]
        
        # Reshape for vectorization?
        # U is (4, r) -> (2, 2, r)
        # V is (8, r) -> (2, 4, r)
        # W is (8, r) -> (4, 2, r)
        
        U_r = U.reshape(2, 2, rank_target)
        V_r = V.reshape(2, 4, rank_target)
        W_r = W.reshape(4, 2, rank_target)
        
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        
        # Compute T
        T = np.zeros((2, 2, 4))
        for r in range(rank_target):
            # U_r[:, :, r] is (2,2)
            # V_r[:, :, r] is (2,4)
            # W_r[:, :, r] is (4,2)
            
            # This loop is slow.
            # Let's just do the scalar check in python, it's small.
            pass
        
        # Scalar check
        err = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    s = 0.0
                    for r in range(rank_target):
                        s += U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]
                    err += (s - 1.0)**2
                    if err > 1e-6:
                        break
                if err > 1e-6:
                    break
            if err > 1e-6:
                break
        
        if err < 1e-9:
            return U, V, W
            
    return None

def strassen_decomposition():
    # Same as before
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def construct():
    # Try to find rank 16 solution
    U224, V224, W224 = find_224_decomp(8)
    
    if U224 is not None:
        r = 8
        U = np.zeros((8, 2*r))
        V = np.zeros((16, 2*r))
        W = np.zeros((8, 2*r))
        
        for i in range(2):
            for j in range(2):
                U[i*4+j, 0:8] = U224[i*2+j, :]
                U[i*4+2+j, 8:16] = U224[i*2+j, :]
                
        for j in range(2):
            for k in range(4):
                V[j*4+k, 0:8] = V224[j*4+k, :]
                V[(j+2)*4+k, 8:16] = V224[j*4+k, :]
                
        for k in range(4):
            for i in range(2):
                W[k*2+i, 0:8] = W224[k*2+i, :]
                W[k*2+i, 8:16] = W224[k*2+i, :]
                
        valid, err = verify_decomposition(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    U, V, W = strassen_decomposition()
    return {"U": U, "V": V, "W": W}
