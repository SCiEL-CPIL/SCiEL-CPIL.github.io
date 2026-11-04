import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Rank 49 Coefficients (Kronecker product of Strassen 2x2)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    perm_uv = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        K = P1 * 4 + P2
        perm_uv[S] = K
        
    perm_w = np.zeros(16, dtype=int)
    for S_c in range(16):
        k = S_c // 4
        i = S_c % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        Q1 = k1 * 2 + i1
        Q2 = k0 * 2 + i0
        K = Q1 * 4 + Q2
        perm_w[S_c] = K
        
    U4_fixed = U4[perm_uv, :]
    V4_fixed = V4[perm_uv, :]
    W4_fixed = W4[perm_w, :]
    
    def check_decomp_fast(U, V, W, rank):
        U_r = U.reshape(4, 4, rank)
        V_r = V.reshape(4, 4, rank)
        W_r = W.reshape(4, 4, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    if check_decomp_fast(U4_fixed, V4_fixed, W4_fixed, 49):
        # Try to find rank 48
        # We will use a simple random restart local search.
        
        def search_rank_48(time_limit):
            end_time = start_time + time_limit
            # We will try a few random initializations
            for _ in range(100):
                if time.time() > end_time:
                    break
                U = np.random.choice([-1, 0, 1], size=(16, 48))
                V = np.random.choice([-1, 0, 1], size=(16, 48))
                W = np.random.choice([-1, 0, 1], size=(16, 48))
                
                # Local search
                for _ in range(100):
                    if time.time() > end_time:
                        break
                    # Pick random entry
                    mat_idx = random.randint(0, 2)
                    row = random.randint(0, 15)
                    col = random.randint(0, 47)
                    
                    if mat_idx == 0:
                        mat = U
                    elif mat_idx == 1:
                        mat = V
                    else:
                        mat = W
                    
                    old_val = mat[row, col]
                    new_val = random.choice([-1, 0, 1])
                    if new_val == old_val:
                        continue
                    
                    mat[row, col] = new_val
                    
                    if check_decomp_fast(U, V, W, 48):
                        return U, V, W
                    
                    # Revert if not better? We don't have a metric.
                    # We can just keep it.
            
            return None

        # Give it 20 seconds
        res = search_rank_48(20)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        return {
            "U": U4_fixed.tolist(),
            "V": V4_fixed.tolist(),
            "W": W4_fixed.tolist()
        }
    else:
        # Fallback to Rank 64
        U64 = np.zeros((16, 64), dtype=int)
        V64 = np.zeros((16, 64), dtype=int)
        W64 = np.zeros((16, 64), dtype=int)
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r = i*16 + j*4 + k
                    U64[i*4+j, r] = 1
                    V64[j*4+k, r] = 1
                    W64[k*4+i, r] = 1
                    
        return {
            "U": U64.tolist(),
            "V": V64.tolist(),
            "W": W64.tolist()
        }
