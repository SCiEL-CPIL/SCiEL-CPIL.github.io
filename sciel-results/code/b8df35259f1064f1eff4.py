import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Schoolbook rank is n*m*p = 30
    # We want to find a rank-minimal decomposition.
    # Known bounds for M_{2,3,5}:
    # Lower bound is often around 22-24.
    # Let's try to find a decomposition with rank R < 30.
    # If we find one, we return it. Otherwise, we return schoolbook.
    
    # The tensor T is defined such that T[i, j, k] = 1 for all i, j, k.
    # We want U (n*m, R), V (m*p, R), W (p*n, R) such that:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] = 1
    
    # This is a system of n*m*p equations.
    # We can use alternating least squares (ALS) or similar iterative methods.
    # To improve the chances of finding a low-rank solution, we can:
    # 1. Use random initialization.
    # 2. Use multiple restarts.
    # 3. Try different ranks.
    
    # Let's try ranks from 22 to 29.
    # If we find a solution, we return it.
    
    # To make the search more efficient, we can use a smarter initialization
    # or a better solver.
    
    # For now, let's stick with the ALS approach but optimize it.
    # We can vectorize the inner loops.
    
    def check_solution(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0)**2
        return err
    
    def solve_for_U(V, W, R):
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return U
    
    def solve_for_V(U, W, R):
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                try:
                    V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return V
    
    def solve_for_W(U, V, R):
        W = np.zeros((p*n, R))
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                try:
                    W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass
        return W
    
    best_rank = n*m*p
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 22 to 29
    for R in range(22, 30):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate
            for it in range(50):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(V, W, R)
                V = solve_for_V(U, W, R)
                W = solve_for_W(U, V, R)
                
                err = check_solution(U, V, W, R)
                if err < 1e-10:
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break
            if best_rank == R:
                break
        if best_rank == R:
            break
    
    if best_U is None:
        # Fall back to schoolbook
        R = n*m*p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
