import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook solution (rank 63)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    target = 1.0
                    err += (pred - target) ** 2
        # Also check off-support entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if (a,b,c) is on support
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k2 = b % p
                    k3 = c // n
                    i3 = c % n
                    if not (i == i3 and j == j2 and k2 == k3):
                        pred = np.dot(U[a, :], V[b, :] * W[c, :])
                        err += pred ** 2
        return np.sqrt(err)
    
    # Try ALS for various ranks
    for target_rank in [27, 30, 35, 40, 45, 50, 55]:
        if time.time() > deadline - 10:
            break
        
        best_err = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k: sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1
                        # M[k,l] = V[j*p+k,l] * W[k*n+i,l]
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        t = np.ones(p)
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        t = np.ones(n)
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        t = np.ones(m)
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
        
        if best_err < 1e-8 and target_rank < best_rank:
            best_U, best_V, best_W, best_rank = best_U_r, best_V_r, best_W_r, target_rank
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
