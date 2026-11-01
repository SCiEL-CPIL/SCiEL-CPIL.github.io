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
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_structured(n, m, p, rank, max_attempts=20000):
    """Try to find a rank-r decomposition using structured random search."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    # Use small half-integer values
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Generate sparse matrices with structured patterns
        U = np.zeros(U_shape)
        V = np.zeros(V_shape)
        W = np.zeros(W_shape)
        
        for c in range(rank):
            # Each column has 1-3 non-zero entries
            n_u = np.random.randint(1, 4)
            n_v = np.random.randint(1, 4)
            n_w = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            w_idx = np.random.choice(p*n, size=n_w, replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=n_u)
            V[v_idx, c] = np.random.choice(candidates, size=n_v)
            W[w_idx, c] = np.random.choice(candidates, size=n_w)
        
        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_densified(n, m, p, rank, max_attempts=5000):
    """Try densified random search with small entries."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    candidates = [-2, -1, 0, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        U = np.random.choice(candidates, size=U_shape)
        V = np.random.choice(candidates, size=V_shape)
        W = np.random.choice(candidates, size=W_shape)
        
        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 13 up to schoolbook_rank - 1
    for rank in range(13, schoolbook_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try structured search first (more likely to find solutions)
        result = _try_rank_structured(n, m, p, rank, max_attempts=30000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try densified search
        result = _try_rank_densified(n, m, p, rank, max_attempts=5000)
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
