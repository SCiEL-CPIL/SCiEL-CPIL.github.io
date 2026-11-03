import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor using numpy for speed."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    # Compute sum of products for this (i,j,k) triple
                    total = np.dot(U_arr[ui], V_arr[vk] * W_arr[wk])
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_best, V_best, W_best = get_schoolbook()
    best_rank = len(U_best[0])
    
    # Try random search for lower ranks with structured initialization
    # Start from higher ranks and work down
    
    for R in range(124, 24, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Use more trials for smaller ranks since they're harder
        max_trials = max(20, int((deadline - time.time()) * 5))
        
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Use structured initialization patterns
            pattern = trial % 4
            
            U = [[0.0] * R for _ in range(n * m)]
            V = [[0.0] * R for _ in range(m * p)]
            W = [[0.0] * R for _ in range(p * n)]
            
            if pattern == 0:
                # Block diagonal structure
                block_size = R // 5
                for i in range(n):
                    for j in range(m):
                        for r in range(block_size * i, block_size * (i + 1)):
                            U[i * m + j][r] = random.uniform(-1, 1) if random.random() < 0.3 else 0.0
                for j in range(m):
                    for k in range(p):
                        for r in range(block_size * j, block_size * (j + 1)):
                            V[j * p + k][r] = random.uniform(-1, 1) if random.random() < 0.3 else 0.0
                for k in range(p):
                    for i in range(n):
                        for r in range(block_size * k, block_size * (k + 1)):
                            W[k * n + i][r] = random.uniform(-1, 1) if random.random() < 0.3 else 0.0
                            
            elif pattern == 1:
                # Striped pattern
                for idx in range(n * m):
                    for r in range(R):
                        if (idx + r) % 3 == 0:
                            U[idx][r] = random.uniform(-1, 1)
                for idx in range(m * p):
                    for r in range(R):
                        if (idx + r) % 3 == 0:
                            V[idx][r] = random.uniform(-1, 1)
                for idx in range(p * n):
                    for r in range(R):
                        if (idx + r) % 3 == 0:
                            W[idx][r] = random.uniform(-1, 1)
                            
            elif pattern == 2:
                # Random sparse with varying density
                density = 0.1 + 0.3 * (trial % 4) / 4.0
                for idx in range(n * m):
                    for r in range(R):
                        if random.random() < density:
                            U[idx][r] = random.uniform(-1, 1)
                for idx in range(m * p):
                    for r in range(R):
                        if random.random() < density:
                            V[idx][r] = random.uniform(-1, 1)
                for idx in range(p * n):
                    for r in range(R):
                        if random.random() < density:
                            W[idx][r] = random.uniform(-1, 1)
                
            else:
                # Integer-valued random for potential exact solutions
                for idx in range(n * m):
                    for r in range(R):
                        if random.random() < 0.2:
                            U[idx][r] = random.choice([-1, 0, 1])
                for idx in range(m * p):
                    for r in range(R):
                        if random.random() < 0.2:
                            V[idx][r] = random.choice([-1, 0, 1])
                for idx in range(p * n):
                    for r in range(R):
                        if random.random() < 0.2:
                            W[idx][r] = random.choice([-1, 0, 1])
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # If no improvement found, return schoolbook
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
