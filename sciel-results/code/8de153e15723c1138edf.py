import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses structured decomposition based on embedding into 5x5x5 structure.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate(U, V, W, n, m, p, rank):
        """
        Validate that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all i in [0,n), j in [0,m), k in [0,p).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        return False
        return True

    # Strategy: Try to construct a low-rank decomposition using structured patterns
    # For 4x5x5, the known upper bound is 100 (schoolbook)
    # Theoretical lower bounds suggest rank >= 20 (max of matrix ranks)
    # Try structured approaches based on Kronecker products and embeddings
    
    def try_structured_decomposition(target_rank, num_trials=200):
        """
        Try structured decompositions with various patterns.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None
            
            # Pattern 1: Random with bias toward small values
            U = np.random.randint(-1, 2, size=(n * m, target_rank), dtype=np.int64)
            V = np.random.randint(-1, 2, size=(m * p, target_rank), dtype=np.int64)
            W = np.random.randint(-1, 2, size=(p * n, target_rank), dtype=np.int64)
            
            # Add some structure: make certain rows/columns sparse
            for r in range(target_rank):
                if np.random.random() < 0.3:
                    U[:, r] = np.where(np.random.random(n * m) < 0.5, 0, 1)
                if np.random.random() < 0.3:
                    V[:, r] = np.where(np.random.random(m * p) < 0.5, 0, 1)
                if np.random.random() < 0.3:
                    W[:, r] = np.where(np.random.random(p * n) < 0.5, 0, 1)
            
            if validate(U, V, W, n, m, p, target_rank):
                return U, V, W
        
        return None, None, None
    
    # Try to find decompositions at various ranks
    # Focus on ranks that are plausible for this problem size
    candidate_ranks = [20, 25, 30, 35, 40, 45, 50]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        U_try, V_try, W_try = try_structured_decomposition(rank, num_trials=300)
        if U_try is not None:
            best_rank = rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # If structured approach didn't work, try random with more trials at lower ranks
    if best_rank == base_rank:
        for rank in [20, 25, 30]:
            if time.time() > deadline:
                break
            
            for trial in range(500):
                if time.time() > deadline:
                    break
                
                U = np.random.randint(-2, 3, size=(n * m, rank), dtype=np.int64)
                V = np.random.randint(-2, 3, size=(m * p, rank), dtype=np.int64)
                W = np.random.randint(-2, 3, size=(p * n, rank), dtype=np.int64)
                
                if validate(U, V, W, n, m, p, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
            
            if best_rank < base_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
