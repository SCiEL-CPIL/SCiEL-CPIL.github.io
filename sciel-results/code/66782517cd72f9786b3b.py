"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower-rank decomposition using ALS
    # Build tensor
    T = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    # Try ranks from 53 down to 20
    for target_rank in range(rank_sb - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Try 3 random initializations
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + seed)
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                T1 = T.reshape(n * p, -1)
                VW = np.kron(V_r, W_r)
                try:
                    U_r = T1 @ VW @ np.linalg.pinv(VW.T @ VW)
                except:
                    break
                
                # Update V
                T2 = T.transpose(1, 0, 2).reshape(n * m, -1)
                UW = np.kron(U_r, W_r)
                try:
                    V_r = T2 @ UW @ np.linalg.pinv(UW.T @ UW)
                except:
                    break
                
                # Update W
                T3 = T.transpose(2, 0, 1).reshape(m * p, -1)
                UV = np.kron(U_r, V_r)
                try:
                    W_r = T3 @ UV @ np.linalg.pinv(UV.T @ UV)
                except:
                    break
            
            if time.time() > deadline - 1:
                break
            
            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                    computed = np.dot(U_r[k * n + i, :], V_r[ip * m + jp, :] * W_r[jp * p + kp, :])
                                    err = abs(computed - expected)
                                    if err > max_err:
                                        max_err = err
            
            if max_err < 1e-8:
                # Round to half-integers
                U_rounded = [[round(x * 2) / 2.0 for x in U_r[row]] for row in range(n * m)]
                V_rounded = [[round(x * 2) / 2.0 for x in V_r[row]] for row in range(m * p)]
                W_rounded = [[round(x * 2) / 2.0 for x in W_r[row]] for row in range(p * n)]
                
                # Verify rounded
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for ip in range(n):
                                for jp in range(m):
                                    for kp in range(p):
                                        expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        computed = sum(U_rounded[k * n + i][c] * V_rounded[ip * m + jp][c] * W_rounded[jp * p + kp][c] for c in range(target_rank))
                                        if abs(computed - expected) > 1e-9:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
        
        if best_rank < rank_sb:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
