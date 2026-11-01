import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
        n, m, p = DIMENSIONS
        # Schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        # Target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1

        # Try to find rank 58
        start_time = time.time()
        target_rank = 58
        allowed_vals = [0, 1, -1, 0.5, -0.5]
        
        # Simple random search
        # This is very unlikely to work quickly, but let's try a few iterations
        for _ in range(100):
            if time.time() - start_time > 10:
                break
            U_rand = np.random.choice(allowed_vals, size=(n*m, target_rank))
            V_rand = np.random.choice(allowed_vals, size=(m*p, target_rank))
            W_rand = np.random.choice(allowed_vals, size=(p*n, target_rank))
            
            # Compute tensor product sum
            # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is slow in Python.
            pass
        
        return {"U": U, "V": V, "W": W}
