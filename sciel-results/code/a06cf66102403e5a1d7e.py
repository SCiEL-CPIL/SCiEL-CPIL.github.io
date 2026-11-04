"""Matrix multiplication 2x4x8 via optimized block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Direct Strassen-based construction: 2 j-blocks x 4 k-blocks x 7 = 56
    # This is deterministic and fast.
    # Try to improve by finding a better 2x4x2 sub-decomposition.
    
    # First, try a structured search for 2x4x2 with rank 7 or 8
    def find_2x4x2_decomp(target_rank):
        """Find a rank-r decomposition for 2x4x2 matrix multiplication using local search."""
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for attempt in range(100):
            if time.time() > deadline - 15:
                break
            
            # Initialize with structured random ±1
            U = np.random.randint(-1, 2, (8, target_rank)).astype(float)
            V = np.random.randint(-1, 2, (8, target_rank)).astype(float)
            W = np.random.randint(-1, 2, (4, target_rank)).astype(float)
            
            # Coordinate descent: optimize each entry
            for cd_iter in range(30):
                if time.time() > deadline - 15:
                    break
                
                # Optimize U rows
                for i in range(2):
                    for j in range(4):
                        idx = i * 4 + j
                        for val in [-1, 0, 1]:
                            old = U[idx].copy()
                            U[idx] = val
                            err = 0.0
                            for k in range(2):
                                for r in range(target_rank):
                                    s = U[idx, r] * V[j*2+k, r] * W[k*2+i, r]
                                # This is wrong, let me compute properly
                            U[idx] = old
                
                # Just evaluate and perturb
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
                
                # Small perturbation
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, 8)
                    col = np.random.randint(0, target_rank)
                    U[idx, col] += np.random.choice([-1, 1])
                elif np.random.rand() < 0.5:
                    idx = np.random.randint(0, 8)
                    col = np.random.randint(0, target_rank)
                    V[idx, col] += np.random.choice([-1, 1])
                else:
                    idx = np.random.randint(0, 4)
                    col = np.random.randint(0, target_rank)
                    W[idx, col] += np.random.choice([-1, 1])
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try rank 8 first (schoolbook), then 7
    sub_U = sub_V = sub_W = None
    sub_rank = None
    
    for r in [8, 7]:
        if time.time() > deadline - 15:
            break
        sub_U, sub_V, sub_W, err = find_2x4x2_decomp(r)
        if sub_U is not None:
            sub_rank = r
            break
    
    if sub_rank is not None:
        total_rank = 4 * sub_rank
        U = np.zeros((8, total_rank))
        V = np.zeros((32, total_rank))
        W = np.zeros((16, total_rank))
        
        for bk in range(4):
            col_offset = bk * sub_rank
            U[:, col_offset:col_offset+sub_rank] = sub_U
            for j in range(4):
                for k_local in range(2):
                    k_global = 2 * bk + k_local
                    V[j*8+k_global, col_offset:col_offset+sub_rank] = sub_V[j*2+k_local]
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * bk + k_local
                    W[k_global*2+i, col_offset:col_offset+sub_rank] = sub_W[k_local*2+i]
        
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
    
    # Fallback: Strassen 2x2x2 blocks (rank 56) - deterministic, fast
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
