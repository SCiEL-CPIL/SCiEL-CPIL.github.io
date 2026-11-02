import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Schoolbook fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Decompose M_{2,3,6} into 3 blocks of M_{2,3,2}
    # Find a low-rank decomposition for M_{2,3,2}
    # n_sub=2, m_sub=3, p_sub=2
    
    def verify_sub(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m_s+j, c] * V[j*p_s+k, c] * W[k*n_s+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def find_sub_decomposition(n_s, m_s, p_s, target_rank, iterations=50000):
        """Find a rank-target_rank decomposition for M_{n_s, m_s, p_s}"""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(iterations):
            if time.time() > deadline - 20:
                break
                
            U = np.random.randn(n_s*m_s, target_rank)
            V = np.random.randn(m_s*p_s, target_rank)
            W = np.random.randn(p_s*n_s, target_rank)
            
            # Compute error
            err = 0.0
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m_s+j, c] * V[j*p_s+k, c] * W[k*n_s+i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                return best_U, best_V, best_W
        
        return best_U, best_V, best_W, best_err

    # Try to find rank 7 for M_{2,3,2}
    # If found, total rank 21
    # If not, try rank 8 for M_{2,3,2}, total rank 24
    
    sub_n, sub_m, sub_p = 2, 3, 2
    
    found_sub = None
    sub_rank = 0
    
    for r_sub in [7, 8]:
        if time.time() > deadline - 30:
            break
        U_s, V_s, W_s, err = find_sub_decomposition(sub_n, sub_m, sub_p, r_sub, iterations=20000)
        if err < 1e-9 and verify_sub(U_s, V_s, W_s, sub_n, sub_m, sub_p):
            found_sub = (U_s, V_s, W_s)
            sub_rank = r_sub
            break
    
    if found_sub:
        U_s, V_s, W_s = found_sub
        total_rank = sub_rank * 3
        
        # Construct the full decomposition
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # B is 3x6, split into 3 blocks of 3x2
        # Block b (0,1,2) corresponds to columns 2*b .. 2*b+1 of B
        # Output C is 2x6, split into 3 blocks of 2x2
        
        for b in range(3):
            # Indices in the full tensor
            # U is same for all blocks (A is shared)
            # V indices: j (0..2), k (2*b .. 2*b+1)
            # W indices: k (2*b .. 2*b+1), i (0..1)
            
            col_offset = b * sub_rank
            
            # U
            U_full[:, col_offset:col_offset+sub_rank] = U_s
            
            # V
            # V_s has shape (3*2, r) = (6, r)
            # V_full has shape (3*6, r) = (18, r)
            # For block b, j is 0..2, k is 2*b .. 2*b+1
            # V_full index: j * 6 + k
            for j in range(sub_m):
                for k_local in range(sub_p):
                    k_global = 2*b + k_local
                    V_full[j * p + k_global, col_offset:col_offset+sub_rank] = V_s[j * sub_p + k_local, :]
            
            # W
            # W_s has shape (2*2, r) = (4, r)
            # W_full has shape (6*2, r) = (12, r)
            # W_full index: k * 2 + i
            for k_local in range(sub_p):
                for i in range(sub_n):
                    k_global = 2*b + k_local
                    W_full[k_global * n + i, col_offset:col_offset+sub_rank] = W_s[k_local * sub_n + i, :]
        
        # Verify full decomposition
        def verify_full(U, V, W):
            r = U.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        if verify_full(U_full, V_full, W_full):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    return best_result
