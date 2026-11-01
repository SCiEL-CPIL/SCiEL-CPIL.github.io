"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12
    
    def compute_residual(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for kp in range(p):
                        for kp2 in range(p):
                            for ip in range(n):
                                expected = 1.0 if (i == ip and j == jp and kp == kp2) else 0.0
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j, c] * V[jp*p+kp, c] * W[kp2*n+ip, c]
                                max_err = max(max_err, abs(s - expected))
        return max_err
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    np.random.seed(42)
    random.seed(42)
    
    for rank in range(35, 60):
        if time.time() > deadline - 5:
            break
        if rank >= best_rank:
            break
            
        found = False
        max_trials = 30
        for trial in range(max_trials):
            if time.time() > deadline - 3:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            for iteration in range(500):
                if time.time() > deadline - 2:
                    break
                
                # Update W
                for kp2 in range(p):
                    for ip in range(n):
                        E = np.zeros(rank)
                        for jp in range(m):
                            E += U[ip*m+jp, :] * V[jp*p+kp2, :]
                        denom = np.dot(E, E)
                        if denom > 1e-14:
                            W[kp2*n+ip, :] = E / denom
                        else:
                            W[kp2*n+ip, :] = E / (denom + 1e-14)
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        C = np.zeros(rank)
                        for kp in range(p):
                            C += V[j*p+kp, :] * W[kp*n+i, :]
                        denom = np.dot(C, C)
                        if denom > 1e-14:
                            U[i*m+j, :] = C / denom
                        else:
                            U[i*m+j, :] = C / (denom + 1e-14)
                
                # Update V
                for jp in range(m):
                    for kp in range(p):
                        D = np.zeros(rank)
                        for i in range(n):
                            D += U[i*m+jp, :] * W[kp*n+i, :]
                        denom = np.dot(D, D)
                        if denom > 1e-14:
                            V[jp*p+kp, :] = D / denom
                        else:
                            V[jp*p+kp, :] = D / (denom + 1e-14)
                
                # Check
                err = compute_residual(U, V, W)
                if err < 1e-9:
                    # Refine
                    for _ in range(100):
                        for kp2 in range(p):
                            for ip in range(n):
                                E = np.zeros(rank)
                                for jp in range(m):
                                    E += U[ip*m+jp, :] * V[jp*p+kp2, :]
                                denom = np.dot(E, E)
                                if denom > 1e-16:
                                    W[kp2*n+ip, :] = E / denom
                                else:
                                    W[kp2*n+ip, :] = E / (denom + 1e-16)
                        for i in range(n):
                            for j in range(m):
                                C = np.zeros(rank)
                                for kp in range(p):
                                    C += V[j*p+kp, :] * W[kp*n+i, :]
                                denom = np.dot(C, C)
                                if denom > 1e-16:
                                    U[i*m+j, :] = C / denom
                                else:
                                    U[i*m+j, :] = C / (denom + 1e-16)
                        for jp in range(m):
                            for kp in range(p):
                                D = np.zeros(rank)
                                for i in range(n):
                                    D += U[i*m+jp, :] * W[kp*n+i, :]
                                denom = np.dot(D, D)
                                if denom > 1e-16:
                                    V[jp*p+kp, :] = D / denom
                                else:
                                    V[jp*p+kp, :] = D / (denom + 1e-16)
                        
                        if compute_residual(U, V, W) < 1e-11:
                            break
                    
                    if compute_residual(U, V, W) < 1e-9:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
