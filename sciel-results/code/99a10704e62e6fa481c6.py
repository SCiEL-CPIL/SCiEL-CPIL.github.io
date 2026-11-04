import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition for M_{2,4,8}
    # 8 independent 2x2 Strassen blocks, each of rank 7, total rank 56
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for bi in range(1):
        for bj in range(2):
            for bk in range(4):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                # U indices (A matrix, 2x4)
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                # V indices (B matrix, 4x8)
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                # W indices (C matrix, 8x2) -> W[k*n+i, r]
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a, col] += 1
                U[u_d, col] += 1
                V[v_e, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] += 1
                W[w_c11, col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c, col] += 1
                U[u_d, col] += 1
                V[v_e, col] += 1
                W[w_c10, col] += 1
                W[w_c11, col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a, col] += 1
                V[v_f, col] += 1
                V[v_h, col] -= 1
                W[w_c01, col] += 1
                W[w_c11, col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d, col] += 1
                V[v_g, col] += 1
                V[v_e, col] -= 1
                W[w_c00, col] += 1
                W[w_c10, col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a, col] += 1
                U[u_b, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] -= 1
                W[w_c01, col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c, col] += 1
                U[u_a, col] -= 1
                V[v_e, col] += 1
                V[v_f, col] += 1
                W[w_c11, col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b, col] += 1
                U[u_d, col] -= 1
                V[v_g, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] += 1
                col += 1

    # Verify correctness
    if time.time() < deadline - 5:
        # Reconstruct A*B from decomposition and compare with direct product
        A = np.random.randint(-3, 3, (n, m))
        B = np.random.randint(-3, 3, (m, p))
        
        # Direct product
        C_direct = A @ B  # shape (n, p)
        
        # From decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r]^T
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # A_flat = A.reshape(n*m), B_flat = B.reshape(m*p)
        # C_flat = W @ (V^T @ B_flat) ... actually:
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_decomp = np.zeros((n, p), dtype=int)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_decomp[i, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        
        # Check if they match
        if not np.array_equal(C_direct, C_decomp):
            # Fallback: use schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
