import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}

    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = 25
    for R in range(25, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 1000 + R)
            
            # Initialize W randomly
            W = np.random.randn(p * n, R)
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Alternating minimization
            converged = False
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # M[i,j] shape (p, R)
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve M @ u = 1
                        # u = pinv(M) @ 1
                        try:
                            u = np.linalg.pinv(M) @ np.ones(p)
                        except:
                            u = np.zeros(R)
                        U[i * m + j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # N[j,k] shape (n, R)
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v = np.linalg.pinv(N) @ np.ones(n)
                        except:
                            v = np.zeros(R)
                        V[j * p + k, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        # P[k,i] shape (m, R)
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w = np.linalg.pinv(P) @ np.ones(m)
                        except:
                            w = np.zeros(R)
                        W[k * n + i, :] = w
                
                # Check residual
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            res = abs(val - 1.0)
                            if res > max_res:
                                max_res = res
                            
                if max_res < 1e-9:
                    converged = True
                    break
                    
            if converged:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded solution is valid
                max_res_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            res = abs(val - 1.0)
                            if res > max_res_rounded:
                                max_res_rounded = res
                            
                if max_res_rounded < 1e-9:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                else:
                    # If rounding fails, maybe return the real one? 
                    # But interface says integer/half-integer.
                    # If I return real, it might be invalid.
                    # Let's try to return the real one but cast to float?
                    # The prompt says "integer/half-integer entries".
                    # I will assume that if I can't find a half-integer solution, I should not return this one.
                    # Or maybe I can scale?
                    # Let's just return fallback if rounding fails.
                    pass

    return fallback

# EVOLVE-BLOCK-END
