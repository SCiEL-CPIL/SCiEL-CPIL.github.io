"""Matrix multiplication 4x5 x 5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n + i, r] convention
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Best known: schoolbook rank 100
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find lower rank decompositions
    # Start from a reasonable upper bound and go down
    # For 4x5 x 5x5, let's try ranks from 80 down to see if we can find something
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each (i, j, k), check that sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all valid (i,j,k) in the matrix multiplication
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Create the full tensor and check
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Build the tensor using einsum
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # For each r, the contribution is outer product of:
        # U[:, r] reshaped to (n, m), V[:, r] reshaped to (m, p), W[:, r] reshaped to (p, n)
        
        total = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            # The contribution to T[i,j,k] from term r is u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            # We need to sum over k for each (i,j)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
        
        # Check if total is all ones
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return U, V, W, rank
    
    # Get schoolbook as initial best
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank decompositions using random search
    # Strategy: try different ranks, use random sparse matrices and check
    
    # Let's try ranks from 90 down to 70
    for target_rank in range(90, 69, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random attempts
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Generate random sparse matrices with entries in {-1, 0, 1}
            # Each column should be sparse
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # For each rank term, randomly assign non-zero entries
            for r in range(target_rank):
                # U column: pick a few random positions
                num_u = random.randint(1, 3)
                u_idx = random.sample(range(n * m), num_u)
                for idx in u_idx:
                    U[idx, r] = random.choice([-1, 1])
                
                num_v = random.randint(1, 3)
                v_idx = random.sample(range(m * p), num_v)
                for idx in v_idx:
                    V[idx, r] = random.choice([-1, 1])
                
                num_w = random.randint(1, 3)
                w_idx = random.sample(range(p * n), num_w)
                for idx in w_idx:
                    W[idx, r] = random.choice([-1, 1])
            
            # Verify
            if verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < n * m * p:
            # Found improvement, try to go lower
            continue
        else:
            # If we couldn't improve at this rank, we could try lower but it's unlikely
            # Let's continue trying lower ranks
            pass
    
    # If we found an improvement, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
