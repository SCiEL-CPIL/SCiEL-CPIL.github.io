import numpy as np
import time
from itertools import product
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, n, m, p):
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def search_decomp(n, m, p, R, trials=5000):
        entry_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for _ in range(trials):
            if time.time() > deadline - 10:
                break
            U = np.random.choice(entry_values, size=(n * m, R))
            V = np.random.choice(entry_values, size=(m * p, R))
            W = np.random.choice(entry_values, size=(p * n, R))
            if verify(U, V, W, n, m, p):
                return U, V, W
        return None

    # Try to find rank 11 decomposition for 2x3x3
    U11, V11, W11 = search_decomp(2, 3, 3, 11, trials=20000)
    
    if U11 is not None:
        # Construct rank 22 decomposition for 2x3x6
        R = 22
        U = np.zeros((2 * 3, R))
        V = np.zeros((3 * 6, R))
        W = np.zeros((6 * 2, R))
        
        # First block (k=0,1,2)
        U[:, :11] = U11
        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :11] = V11[j * 3 + k, :]
        for k in range(3):
            for i in range(2):
                W[k * 2 + i, :11] = W11[k * 2 + i, :]
                
        # Second block (k=3,4,5)
        U[:, 11:] = U11
        for j in range(3):
            for k in range(3):
                V[j * 6 + (k + 3), 11:] = V11[j * 3 + k, :]
        for k in range(3):
            for i in range(2):
                W[(k + 3) * 2 + i, 11:] = W11[k * 2 + i, :]
                
        if verify(U, V, W, 2, 3, 6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try to find rank 12 decomposition for 2x3x3
    U12, V12, W12 = search_decomp(2, 3, 3, 12, trials=20000)
    
    if U12 is not None:
        # Construct rank 24 decomposition for 2x3x6
        R = 24
        U = np.zeros((2 * 3, R))
        V = np.zeros((3 * 6, R))
        W = np.zeros((6 * 2, R))
        
        U[:, :12] = U12
        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :12] = V12[j * 3 + k, :]
        for k in range(3):
            for i in range(2):
                W[k * 2 + i, :12] = W12[k * 2 + i, :]
                
        U[:, 12:] = U12
        for j in range(3):
            for k in range(3):
                V[j * 6 + (k + 3), 12:] = V12[j * 3 + k, :]
        for k in range(3):
            for i in range(2):
                W[(k + 3) * 2 + i, 12:] = W12[k * 2 + i, :]
                
        if verify(U, V, W, 2, 3, 6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
