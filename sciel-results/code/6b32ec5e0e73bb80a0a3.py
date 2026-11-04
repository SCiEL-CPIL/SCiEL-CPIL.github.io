"""Matrix multiplication 2x4x8 via Strassen block decomposition."""
import time

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Split into blocks:
    # i dimension: 2 -> 1 block of size 2
    # j dimension: 4 -> 2 blocks of size 2
    # k dimension: 8 -> 4 blocks of size 2
    # Total: 1 * 2 * 4 = 8 blocks, each using Strassen (rank 7)
    # Total rank = 8 * 7 = 56
    
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
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
                
                # U row indices: i*m + j
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                # V row indices: j*p + k
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                # W row indices: k*n + i
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1
                U[u_d][col] += 1
                V[v_e][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1
                U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1
                W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1
                V[v_h][col] -= 1
                W[w_c01][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1
                V[v_e][col] -= 1
                W[w_c00][col] += 1
                W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1
                U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1
                W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1
                U[u_a][col] -= 1
                V[v_e][col] += 1
                V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1
                U[u_d][col] -= 1
                V[v_g][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Verify with numpy if available
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank_school = n * m * p
            U2 = [[0] * rank_school for _ in range(n * m)]
            V2 = [[0] * rank_school for _ in range(m * p)]
            W2 = [[0] * rank_school for _ in range(p * n)]
            col2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j][col2] = 1
                        V2[j*p+k][col2] = 1
                        W2[k*n+i][col2] = 1
                        col2 += 1
            return {"U": U2, "V": V2, "W": W2}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
