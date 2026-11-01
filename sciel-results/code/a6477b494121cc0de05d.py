import time
import numpy as np

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 100
    
    n, m, p = 2, 2, 12
    
    # Fallback schoolbook rank-48
    fb = _schoolbook(n, m, p)
    
    # Rank-42 construction via 6 Strassen blocks of size 2
    # Strassen-7 for 2x2 matrix multiplication
    U_s = np.array([
        [1, 1, 0, 0, 1, 0, -1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, -1, 0]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 1, 0]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 0, -1, -1, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ], dtype=float)
    
    # Verify the Strassen block
    def _verify_block(U_b, V_b, W_b, ns, ms, ps):
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    s = 0.0
                    for r in range(len(U_b[0])):
                        s += U_b[i*ms+j][r] * V_b[j*ps+k][r] * W_b[k*ns+i][r]
                    if abs(s - (1.0 if j == i else 0.0)) > 1e-10:
                        return False
        return True
    
    if not _verify_block(U_s.tolist(), V_s.tolist(), W_s.tolist(), 2, 2, 2):
        return fb
    
    # Build rank-42 decomposition
    num_groups = 6
    rank = num_groups * 7
    
    U_full = np.zeros((n * m, rank), dtype=float)
    V_full = np.zeros((m * p, rank), dtype=float)
    W_full = np.zeros((p * n, rank), dtype=float)
    
    for g in range(num_groups):
        for r in range(7):
            col = g * 7 + r
            # U: A factors, same for all groups
            U_full[:, col] = U_s[:, r]
            
            # V: B factors for this group (columns 2g, 2g+1 of B)
            for j in range(m):
                for k in range(2):
                    v_row = j * p + (2 * g + k)
                    V_full[v_row, col] = V_s[j*2 + k, r]
            
            # W: C factors for this group (rows 2g, 2g+1 of C)
            for k in range(2):
                for i in range(n):
                    w_row = (2 * g + k) * n + i
                    W_full[w_row, col] = W_s[k*2 + i, r]
    
    # Verify full decomposition
    if _check(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        if time.time() < deadline:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If rank-42 verification failed or time is up, try a direct rank-42 with different structure
    # Alternative: use 3 groups of 4 with potentially rank-14 per group
    # But Strassen for 4x4 is rank ~28, so 2 groups of 4 would be rank 56+
    # Stick with 6 groups of 2
    
    # One more attempt: verify with numpy for speed
    if time.time() < deadline - 10:
        # Quick numpy verification
        tensor_check = np.zeros((n, m, p))
        for r in range(rank):
            u_vec = U_full[:, r]
            v_vec = V_full[:, r]
            w_vec = W_full[:, r]
            # u_vec[i*m+j], v_vec[j*p+k], w_vec[k*n+i]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # For each (i,j,k): contribution is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is not a simple outer product, so let's do it element-wise
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_check[i, j, k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                if i == j:
                    for k in range(p):
                        expected[i, j, k] = 1.0
        
        if np.allclose(tensor_check, expected, atol=1e-10):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return fb
# EVOLVE-BLOCK-END
