"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j
    # Actually: C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * ... 
    # The correct check: for random A (n x m) and B (m x p), check A@B == sum_c (A * U_c) * (V_c * B) * W_c
    # More precisely: C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: C[i,k] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k]
    # Wait, the standard form is:
    # C[i,k] = sum_c (sum_j U[i,c]*A[i,j]*V[j,c]) * W[k,c] ... this is getting confusing.
    # 
    # Let me use the direct tensor check:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # because T[i,j,k]=1 for matrix multiplication.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, max_iters=50000):
    """Try to find a rank-r decomposition with integer/half-integer entries."""
    for it in range(max_iters):
        # Random initialization with small integers/half-integers
        vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p*n)]
        
        # Check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
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
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try decreasing ranks
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    for rank in range(20, best_rank):  # Try from rank 20 upward
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, max_iters=10000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
