import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    # Try to find a decomposition with rank 41
    target_rank = rank_schoolbook - 1
    
    start_time = time.time()
    deadline = 100  # 20 seconds margin
    
    # Random search
    best_error = float('inf')
    best_solution = None
    
    for _ in range(1000):
        if time.time() - start_time > deadline:
            break
            
        # Generate random rank-1 tensors
        U = np.random.randint(-3, 4, size=(n * m, target_rank))
        V = np.random.randint(-3, 4, size=(m * p, target_rank))
        W = np.random.randint(-3, 4, size=(p * n, target_rank))
        
        # Compute the tensor
        # T = sum_l U[:,l] * V[:,l] * W[:,l]
        # We only need to check the target indices
        error = 0
        for idx in target_indices:
            i, j, k = idx
            val = 0
            for l in range(target_rank):
                val += U[i, l] * V[j, l] * W[k, l]
            error += (val - 1) ** 2
            
        # Check other indices (should be 0)
        # This is expensive, so we skip it for now
        # But we must ensure it's 0
        
        if error == 0:
            # Check other indices
            valid = True
            for i in range(n * m):
                for j in range(m * p):
                    for k in range(p * n):
                        if (i, j, k) not in target_indices:
                            val = 0
                            for l in range(target_rank):
                                val += U[i, l] * V[j, l] * W[k, l]
                            if val != 0:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_solution = (U, V, W)
                break
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return schoolbook
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
