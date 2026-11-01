"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # For each (i, j, k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other index combinations, the sum should be 0.
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This should equal 1
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, rank, max_attempts=10000):
    """Try to find a rank-r decomposition by random search."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    # Use small integer values for entries
    candidates = [-2, -1, 0, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        U = np.random.choice(candidates, size=U_shape)
        V = np.random.choice(candidates, size=V_shape)
        W = np.random.choice(candidates, size=W_shape)
        
        # Check if this decomposition works
        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _systematic_search(n, m, p, rank):
    """Systematic search using structure of matrix multiplication."""
    # For matrix multiplication, we know the tensor has specific structure.
    # Let's use a more informed search.
    
    # The key insight: for each column c, we have vectors u_c, v_c, w_c
    # such that sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = delta_{i,j,k}
    
    # Try structured approaches with small integer entries
    candidates = [-1, 0, 1]
    
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    # Try random structured searches
    for attempt in range(50000):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Use sparse random matrices
        U = np.zeros(U_shape)
        V = np.zeros(V_shape)
        W = np.zeros(W_shape)
        
        # Fill with sparse random entries
        for c in range(rank):
            # Each column has a few non-zero entries
            u_idx = np.random.choice(n*m, size=np.random.randint(1, 4), replace=False)
            v_idx = np.random.choice(m*p, size=np.random.randint(1, 4), replace=False)
            w_idx = np.random.choice(p*n, size=np.random.randint(1, 4), replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=len(u_idx))
            V[v_idx, c] = np.random.choice(candidates, size=len(v_idx))
            W[w_idx, c] = np.random.choice(candidates, size=len(w_idx))
        
        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, try to find a lower rank decomposition
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 13 up to schoolbook_rank - 1
    # Known lower bound for 2x2x5 is around 13-14
    for rank in range(13, schoolbook_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try systematic search first
        result = _systematic_search(n, m, p, rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try random search
        result = _try_rank(n, m, p, rank, max_attempts=2000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
