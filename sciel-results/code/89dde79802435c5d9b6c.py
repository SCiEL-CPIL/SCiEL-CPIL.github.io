"""Matrix multiplication 2x4 by 4x5 tensor decomposition with rank minimization."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, max_iters=20000):
    """Try to find a rank-r decomposition with integer/half-integer entries."""
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Sparse random initialization: each row has a few non-zero entries
        nnz_per_row = max(1, rank // 4)
        
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        
        for row in range(n*m):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                U[row][col] = random.choice(vals)
        
        for row in range(m*p):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                V[row][col] = random.choice(vals)
        
        for row in range(p*n):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                W[row][col] = random.choice(vals)
        
        # Fast check using numpy
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape for efficient tensor check
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from low to high, with time budget for each
    # For 2x4x5, known lower bound is around 20-25
    # Try from 18 upward
    for rank in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Allocate more time for lower ranks (more promising)
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Give more iterations to lower ranks
        iters = min(50000, max(5000, int(remaining * 300)))
        
        result = _try_rank(rank, n, m, p, deadline, max_iters=iters)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
