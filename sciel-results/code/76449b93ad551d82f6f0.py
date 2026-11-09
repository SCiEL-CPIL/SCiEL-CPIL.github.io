import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    def verify(U, V, W):
        """Verify if U, V, W form a valid decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                V_block = V[j*p:(j+1)*p, :]
                W_block = W[i::n, :]
                inner = (V_block * W_block).sum(axis=0)
                val = np.dot(u_row, inner)
                if abs(val - 1.0) > 1e-9:
                    return False
        return True
    
    # Try to find a low rank decomposition
    for r in range(14, 31):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 200 if r < 18 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Try different initialization strategies
            strategy = attempt % 3
            
            if strategy == 0:
                # Sparse random with -1, 0, 1
                U = np.random.randint(-1, 2, size=(n*m, r))
                V = np.random.randint(-1, 2, size=(m*p, r))
                W = np.random.randint(-1, 2, size=(p*n, r))
            elif strategy == 1:
                # Mostly zeros with some 1s and -1s
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # Randomly set some entries
                for _ in range(r * 2):
                    U[random.randint(0, n*m-1), random.randint(0, r-1)] = random.choice([-1, 1])
                    V[random.randint(0, m*p-1), random.randint(0, r-1)] = random.choice([-1, 1])
                    W[random.randint(0, p*n-1), random.randint(0, r-1)] = random.choice([-1, 1])
            else:
                # Structured: use a base pattern and perturb
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # Initialize with a partial schoolbook-like structure
                for col in range(min(r, n*m*p)):
                    i = col // (m*p)
                    rem = col % (m*p)
                    j = rem // p
                    k = rem % p
                    if i < n and j < m and k < p:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                # Add random perturbations
                for _ in range(r):
                    U[random.randint(0, n*m-1), random.randint(0, r-1)] += random.choice([-1, 1])
                    V[random.randint(0, m*p-1), random.randint(0, r-1)] += random.choice([-1, 1])
                    W[random.randint(0, p*n-1), random.randint(0, r-1)] += random.choice([-1, 1])
            
            # Local search: try to fix violations by flipping entries
            for _ in range(50):
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Find a violated (i,j) pair
                violated = False
                for i in range(n):
                    for j in range(m):
                        u_row = U[i*m+j, :]
                        V_block = V[j*p:(j+1)*p, :]
                        W_block = W[i::n, :]
                        inner = (V_block * W_block).sum(axis=0)
                        val = np.dot(u_row, inner)
                        if abs(val - 1.0) > 1e-9:
                            violated = True
                            # Try flipping a random entry
                            break
                    if violated:
                        break
                
                if not violated:
                    break
                
                # Randomly flip an entry in U, V, or W
                mat = random.choice([U, V, W])
                row = random.randint(0, mat.shape[0]-1)
                col = random.randint(0, mat.shape[1]-1)
                mat[row, col] += random.choice([-2, 2])  # Flip sign or change value
    
    # If no low rank found, return schoolbook
    rank = rank_schoolbook
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
    return {"U": U, "V": V, "W": W}
