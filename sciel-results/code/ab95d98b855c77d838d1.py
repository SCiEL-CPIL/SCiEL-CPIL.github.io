import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 145.0

    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12

    # Build tensor
    T = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_fast(U, V, W):
        R = U.shape[1]
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)  # (Nm*Mp, R)
        T_rec_flat = UV_flat @ W.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)

    # Construct Rank 12 decomposition using block-diagonal strategy
    # Decompose 3x4x4 into 1x4x4 (row 2) and 2x4x4 (rows 0,1)
    def make_rank12():
        R = 12
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        
        # Subproblem 1: i=2 (1x4x4)
        # Uses columns 0 to 3
        for r in range(4):
            # U index: i*m + j = 2*4 + r = 8 + r
            U[8 + r, r] = 1.0
            # V index: j*p + k = r*4 + k
            for k in range(4):
                V[r*4 + k, r] = 1.0
            # W index: k*n + i = k*3 + 2
            for k in range(4):
                W[k*3 + 2, r] = 1.0
                
        # Subproblem 2: i=0,1 (2x4x4)
        # Uses columns 4 to 11
        # r in 4..11 corresponds to i' in {0,1}, j' in {0,1,2,3}
        for i_prime in range(2):
            for j_prime in range(4):
                r = 4 + i_prime * 4 + j_prime
                # U index: i*m + j = i_prime*4 + j_prime
                U[i_prime*4 + j_prime, r] = 1.0
                # V index: j*p + k = j_prime*4 + k
                for k in range(4):
                    V[j_prime*4 + k, r] = 1.0
                # W index: k*n + i = k*3 + i_prime
                for k in range(4):
                    W[k*3 + i_prime, r] = 1.0
                    
        return U, V, W

    U, V, W = make_rank12()
    
    if verify_fast(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to direct if verification fails
        R = n * m * p
        U_dir = np.zeros((Nm, R))
        V_dir = np.zeros((Mp, R))
        W_dir = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_dir[i*m+j, idx] = 1.0
                    V_dir[j*p+k, idx] = 1.0
                    W_dir[k*n+i, idx] = 1.0
                    idx += 1
        return {
            "U": U_dir.tolist(),
            "V": V_dir.tolist(),
            "W": W_dir.tolist()
        }
