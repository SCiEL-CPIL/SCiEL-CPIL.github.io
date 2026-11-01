"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the factorization exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rng = np.random.RandomState(42)
    for trial in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_fact = np.zeros((n, p))
        for r in range(rank):
            uA = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    uA[i] += U_arr[i*m+j, r] * A[i,j]
            vB = np.zeros(p)
            for k in range(p):
                for jp in range(m):
                    vB[k] += V_arr[jp*p+k, r] * B[jp,k]
            for i in range(n):
                for k in range(p):
                    C_fact[i,k] += W_arr[k*n+i, r] * uA[i] * vB[k]
        
        if not np.allclose(C_fact, C_expected, atol=1e-9):
            return False
    return True

def _construct_rank24():
    """Trivial rank-24 decomposition."""
    n, m, p = 2, 3, 4
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

def _try_structured_lower_rank(deadline):
    """Try to find a lower-rank decomposition using structured approaches."""
    n, m, p = 2, 3, 4
    
    # Try rank 18: based on known results that rank(2x3x4) might be 18
    # Use a block approach: decompose 2x3x4 using the 2x3x3 rank-8 structure
    # plus additional terms for the 4th column
    
    # Known rank-8 decomposition for 2x3x3 (from literature):
    # This is complex. Let me try a simpler structured approach.
    
    # Attempt: Use the fact that 2x3x4 can be viewed as 2x3x3 with an extra column.
    # The 2x3x3 part has rank 8, and we need 3 more for the extra column = rank 11?
    # But this naive bound is likely not tight.
    
    # Let's try a random search for rank 20, 19, 18
    rng = np.random.RandomState(789)
    
    for target_rank in [20, 19, 18]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with structured random factors
            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)
            
            # Scale to help convergence
            U *= 0.5
            V *= 0.5
            W *= 0.5
            
            # Quick verification
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin out of 120s
    
    # First, try to find a lower-rank decomposition
    result = _try_structured_lower_rank(deadline)
    
    if result is not None:
        return result
    
    # Fall back to rank-24
    return _construct_rank24()
# EVOLVE-BLOCK-END
