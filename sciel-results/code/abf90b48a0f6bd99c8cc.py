"""Matrix multiplication 2x4x8 via optimized block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Strategy: Split k into 4 blocks of size 2.
    # For each k-block, solve 2x4x2 sub-problem.
    # Find a low-rank decomposition for 2x4x2 and replicate 4 times.
    # If 2x4x2 has rank r, total rank = 4*r.
    
    # Let's try to find a good decomposition for 2x4x2.
    # We'll use a structured approach with shared terms.
    
    # For 2x4x2, let's try rank 12 (giving total 48) or lower.
    # Construct using a combination of Strassen-like ideas.
    
    # Approach: Use 2 Strassen 2x2x2 blocks (for j-pairs (0,1) and (2,3)) 
    # but share some products across them.
    
    # Actually, let's try a direct construction for 2x4x2 with rank 11 or 12.
    
    # Let me try a different split: treat the 4 j-indices as a whole.
    # Use the fact that we only need 4 output entries (2x2).
    
    # I'll use a computational search with numpy to find a rank-r decomposition.
    
    def find_2x4x2_decomp(target_rank):
        """Find a rank-r decomposition for 2x4x2 matrix multiplication."""
        # T[i,j,k] = 1 for i in {0,1}, j in {0..3}, k in {0,1}
        # U shape: (8, r) where row = i*4+j
        # V shape: (8, r) where row = j*2+k  
        # W shape: (4, r) where row = k*2+i
        
        # Use random initialization and gradient-free search
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Initialize with random ±1 entries
            U = np.random.randint(-1, 2, (8, target_rank)).astype(float)
            V = np.random.randint(-1, 2, (8, target_rank)).astype(float)
            W = np.random.randint(-1, 2, (4, target_rank)).astype(float)
            
            # Compute error
            err = 0.0
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r]
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try to find 2x4x2 decomposition with decreasing rank
    sub_U = sub_V = sub_W = None
    sub_rank = None
    
    for r in range(12, 15):  # Try ranks 12, 13, 14
        if time.time() > deadline - 10:
            break
        sub_U, sub_V, sub_W, err = find_2x4x2_decomp(r)
        if sub_U is not None:
            sub_rank = r
            break
    
    if sub_rank is not None:
        # Build full 2x4x8 decomposition
        total_rank = 4 * sub_rank
        U = np.zeros((8, total_rank))
        V = np.zeros((32, total_rank))
        W = np.zeros((16, total_rank))
        
        for bk in range(4):
            col_offset = bk * sub_rank
            # U: same for all k-blocks (A doesn't depend on k)
            U[:, col_offset:col_offset+sub_rank] = sub_U
            # V: row j*8+k -> for block bk, k is in [2*bk, 2*bk+1]
            for j in range(4):
                for k_local in range(2):
                    k_global = 2 * bk + k_local
                    V[j*8+k_global, col_offset:col_offset+sub_rank] = sub_V[j*2+k_local]
            # W: row k*2+i -> for block bk
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * bk + k_local
                    W[k_global*2+i, col_offset:col_offset+sub_rank] = sub_W[k_local*2+i]
        
        # Verify
        max_err = 0.0
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0.0
                    for r in range(total_rank):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Strassen 2x2x2 blocks (rank 56)
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
