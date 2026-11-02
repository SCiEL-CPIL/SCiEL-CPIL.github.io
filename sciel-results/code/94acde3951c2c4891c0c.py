import time
import numpy as np

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 12, 32, 24
    
    # Schoolbook
    rank_s = n*m*p  # 96
    U_s = np.zeros((nm, rank_s))
    V_s = np.zeros((mp, rank_s))
    W_s = np.zeros((pn, rank_s))
    for c in range(rank_s):
        i = c // (m*p)
        rem = c % (m*p)
        j = rem // p
        k = rem % p
        U_s[i*m+j, c] = 1
        V_s[j*p+k, c] = 1
        W_s[k*n+i, c] = 1
    
    best = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_s
    
    # Try lower ranks
    for target_rank in [88, 84, 80, 76]:
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            
            U_t = np.random.randn(nm, target_rank) * 0.1
            V_t = np.random.randn(mp, target_rank) * 0.1
            W_t = np.random.randn(pn, target_rank) * 0.1
            
            # Alternating minimization
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W given U, V
                for i in range(n):
                    for k in range(p):
                        # System: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U_t[i*m+j, :] * V_t[j*p+k, :]
                        # A @ W_row = ones(m)
                        try:
                            W_t[k*n+i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        except:
                            break
                    
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V_t[j*p+k, :] * W_t[k*n+i, :]
                        try:
                            U_t[i*m+j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            break
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U_t[i*m+j, :] * W_t[k*n+i, :]
                        try:
                            V_t[j*p+k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        except:
                            break
                
                # Check error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_t[i*m+j, :], V_t[j*p+k, :] * W_t[k*n+i, :])
                            err += (s - 1)**2
                if err < 1e-6:
                    found = True
                    break
            
            if found:
                best = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
                best_rank = target_rank
                break
        
        if found:
            break
    
    return best
