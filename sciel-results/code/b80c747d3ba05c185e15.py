import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Target rank 42 is the standard Strassen-based bound for 2x4x6
    # We construct it by decomposing the multiplication into 2x2x2 blocks.
    # A is 2x4 -> 2 blocks of 2x2 along columns
    # B is 4x6 -> 2 blocks of 2x2 along rows, 3 blocks of 2x2 along cols
    # Total 2 * 3 = 6 blocks of 2x2x2 multiplications
    # Each 2x2x2 multiplication has rank 7 (Strassen)
    # Total rank = 6 * 7 = 42
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 @ 2x2 rank 7 decomposition
    # Indices in 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1},      # P7
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1},       # P7
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: -1},      # P2
        {1: 1, 3: 1},       # P3
        {0: 1, 2: 1},       # P4
        {0: -1, 1: 1},      # P5
        {3: 1},             # P6
        {0: 1},             # P7
    ]
    
    col = 0
    # Iterate over the 2x2 blocks of A (along j) and B (along j and k)
    # A has 2 blocks of size 2 in dimension m=4
    # B has 2 blocks of size 2 in dimension m=4 and 3 blocks of size 2 in dimension p=6
    
    for j_block in range(2):
        for k_block in range(3):
            for prod_idx in range(7):
                r = col + prod_idx
                
                # Fill U
                # U index: i * m + j
                # i in {0,1}, j in {j_block*2, j_block*2+1}
                for local_i, sign in strassen_U[prod_idx].items():
                    gi = local_i // 2
                    gj = local_i % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # Fill V
                # V index: j * p + k
                # j in {j_block*2, j_block*2+1}, k in {k_block*2, k_block*2+1}
                for local_jk, sign in strassen_V[prod_idx].items():
                    gj = local_jk // 2
                    gk = local_jk % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # Fill W
                # W index: k * n + i
                # k in {k_block*2, k_block*2+1}, i in {0,1}
                for local_ik, sign in strassen_W[prod_idx].items():
                    gi = local_ik // 2
                    gk = local_ik % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
            
    # Verification
    def _verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        result = np.zeros((n, p))
        for r in range(rank):
            M = U_3d[:, :, r] @ V_3d[:, :, r]
            W_r = W_3d[:, :, r]
            result += M * W_r.T
        
        target = np.ones((n, p))
        return np.allclose(result, target, atol=1e-9)

    if not _verify(U, V, W, n, m, p):
        # Fallback to naive if verification fails (should not happen)
        rank_naive = n * m * p
        U_naive = np.zeros((n*m, rank_naive))
        V_naive = np.zeros((m*p, rank_naive))
        W_naive = np.zeros((p*n, rank_naive))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, idx] = 1
                    V_naive[j * p + k, idx] = 1
                    W_naive[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
