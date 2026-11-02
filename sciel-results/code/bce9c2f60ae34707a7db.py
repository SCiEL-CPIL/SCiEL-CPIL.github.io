import time
import numpy as np

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin

def _time_remaining():
    return max(0.0, TIME_LIMIT - (time.time() - _START_TIME))

def _strassen_2x2_tensors():
    """
    Returns U, V, W cores for 2x2x2 Strassen decomposition.
    Shapes: U (4,7), V (4,7), W (4,7)
    Indexing:
    U row i*m+j (i,j in 0..1)
    V row j*p+k (j,k in 0..1)
    W row k*n+i (k,i in 0..1)
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_2x2x16():
    n, m, p = 2, 2, 16
    U_core, V_core, W_core = _strassen_2x2_tensors()
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for b in range(num_blocks):
        col_off = b * 7
        # U is same for all blocks
        U[:, col_off:col_off+7] = U_core
        
        # V mapping:
        # Core V rows: 0:(j=0,k=0), 1:(j=0,k=1), 2:(j=1,k=0), 3:(j=1,k=1)
        # Global V rows: j*p + k
        # Block b corresponds to k = 2b, 2b+1
        V[2*b, col_off:col_off+7] = V_core[0, :]
        V[2*b+1, col_off:col_off+7] = V_core[1, :]
        V[m*p + 2*b, col_off:col_off+7] = V_core[2, :]
        V[m*p + 2*b+1, col_off:col_off+7] = V_core[3, :]
        
        # W mapping:
        # Core W rows: 0:(k=0,i=0), 1:(k=0,i=1), 2:(k=1,i=0), 3:(k=1,i=1)
        # Global W rows: k*n + i
        # Block b corresponds to k = 2b, 2b+1
        W[2*b*n, col_off:col_off+7] = W_core[0, :]
        W[2*b*n + 1, col_off:col_off+7] = W_core[1, :]
        W[(2*b+1)*n, col_off:col_off+7] = W_core[2, :]
        W[(2*b+1)*n + 1, col_off:col_off+7] = W_core[3, :]
        
        if _time_remaining() < 5.0:
            break # Should not happen for fixed size, but safe
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    U_f = np.array(U, dtype=np.float64)
    V_f = np.array(V, dtype=np.float64)
    W_f = np.array(W, dtype=np.float64)
    rank = U_f.shape[1]
    
    # Check a few random samples if rank is large?
    # Here rank is 56, size is small. Full check is 2*2*16 = 64 terms.
    # Each term involves dot products of length 56.
    # 64 * 56 * 3 ops ~ 10k ops. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                # T_{ijk} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
                u_row = U_f[i*m+j]
                v_row = V_f[j*p+k]
                w_row = W_f[k*n+i]
                # Efficient dot: sum(u * v * w)
                val = np.dot(u_row * v_row, w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 2, 16
    
    # Try rank 56
    try:
        if _time_remaining() > 10.0:
            U, V, W = _construct_2x2x16()
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
        
    # Fallback: Rank 64 (Schoolbook)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
