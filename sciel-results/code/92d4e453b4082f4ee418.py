import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Ranks to try
    ranks_to_try = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W randomly
            # Shapes: (n*m, R), (m*p, R), (p*n, R)
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # For each i, j
                for i in range(n):
                    for j in range(m):
                        # A is p x R
                        A = np.zeros((p, R))
                        for k in range(p):
                            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                            A[k] = V[j * p + k] * W[k * n + i]
                        b = np.ones(p)
                        # Solve A x = b
                        try:
                            U[i * m + j] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i] = U[i * m + j] * W[k * n + i]
                        b = np.ones(n)
                        try:
                            V[j * p + k] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j] = U[i * m + j] * V[j * p + k]
                        b = np.ones(m)
                        try:
                            W[k * n + i] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                            err += (val - 1) ** 2
                if err < 1e-6:
                    break
            
            # Check if solution is good
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1) ** 2
                        
            if err < 1e-5:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i * m + j], V_snap[j * p + k] * W_snap[k * n + i])
                            if abs(val - 1) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                
                if exact:
                    # Convert to list of lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
