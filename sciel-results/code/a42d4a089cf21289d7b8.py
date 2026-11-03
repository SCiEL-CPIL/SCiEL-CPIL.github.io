import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_integer(U_int, V_int, W_int):
        # U_int, V_int, W_int are lists of lists of integers (scaled by 2)
        # Check tensor product exactly
        n, m, p = DIMENSIONS
        rank = len(U_int[0])
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                    if s != 8:  # 2*2*2 = 8
                        return False
                        
        # Check zero entries
        for a in range(n*m):
            i1 = a // m
            j1 = a % m
            for b in range(m*p):
                j2 = b // p
                k2 = b % p
                if j1 != j2:
                    for c in range(p*n):
                        s = 0
                        for r in range(rank):
                            s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                        if s != 0:
                            return False
                else:
                    for c in range(p*n):
                        k1 = c // n
                        i2 = c % n
                        if i1 == i2 and k1 == k2:
                            continue
                        else:
                            s = 0
                            for r in range(rank):
                                s += U_int[a][r] * V_int[b][r] * W_int[c][r]
                            if s != 0:
                                return False
        return True

    def rationalize_to_half_int(M, tol=1e-5):
        # M is a numpy array
        # Returns list of lists of integers (scaled by 2) or None
        rows, cols = M.shape
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                val = M[i, j]
                # Try to find integer/2 close to val
                # val = k/2 => 2*val = k
                k_float = val * 2
                k_int = round(k_float)
                if abs(k_float - k_int) < tol:
                    result[i][j] = k_int
                else:
                    return None
        return result

    def als_decompose(T, R, max_iters=100, tol=1e-8):
        np.random.seed(42) # Fixed seed for reproducibility in initial attempt, maybe vary later
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        err = float('inf')
        for it in range(max_iters):
            # Update U
            for a in range(n*m):
                A = np.zeros((m*p * p*n, R))
                y = np.zeros(m*p * p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
                
            # Update V
            for b in range(m*p):
                A = np.zeros((n*m * p*n, R))
                y = np.zeros(n*m * p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
                
            # Update W
            for c in range(p*n):
                A = np.zeros((n*m * m*p, R))
                y = np.zeros(n*m * m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        y[idx] = T[a, b, c]
                        idx += 1
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                return U, V, W, err
                
        return U, V, W, err

    # Search for minimal R
    # Start from a lower bound. 
    # Simple bound: n*m*p = 36.
    # Let's try R from 35 down to 30.
    
    best_result = schoolbook()
    
    # We will iterate R from 35 down to 30.
    # If time is running out, break and return schoolbook.
    
    for R in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(10): # Reduced restarts to fit in time
            if time.time() > deadline - 5:
                break
            
            # Vary seed slightly
            np.random.seed(42 + restart * 100 + R)
            U, V, W, err = als_decompose(T, R, max_iters=50)
            
            if err < 1e-4: # Threshold for trying to rationalize
                # Try to rationalize
                U_int = rationalize_to_half_int(U, tol=1e-5)
                V_int = rationalize_to_half_int(V, tol=1e-5)
                W_int = rationalize_to_half_int(W, tol=1e-5)
                
                if U_int and V_int and W_int:
                    if verify_integer(U_int, V_int, W_int):
                        # Success
                        U_out = [[x / 2.0 for x in row] for row in U_int]
                        V_out = [[x / 2.0 for x in row] for row in V_int]
                        W_out = [[x / 2.0 for x in row] for row in W_int]
                        return {"U": U_out, "V": V_out, "W": W_out}
                        
    # Fallback to schoolbook
    return best_result
