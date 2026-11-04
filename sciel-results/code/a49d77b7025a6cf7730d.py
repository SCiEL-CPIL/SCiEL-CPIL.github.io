"""Find rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. U: (n*m, r), V: (m*p, r), W: (p*n, r)"""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, max_trials=200):
    """Try to find a valid decomposition at given rank."""
    best_result = None
    np.random.seed(int(time.time()) % 100000)
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Initialize with small integers or half-integers
        scale = 1.0
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Random initialization with small values
        for c in range(rank):
            # Mix of strategies
            if c % 3 == 0:
                U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
            elif c % 3 == 1:
                U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
            else:
                U[:, c] = np.random.choice([-2, -1, 0, 1, 2], size=n*m)
                V[:, c] = np.random.choice([-2, -1, 0, 1, 2], size=m*p)
                W[:, c] = np.random.choice([-2, -1, 0, 1, 2], size=p*n)
        
        # Convert to lists of floats
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    # Try to find lower rank decompositions
    # Start from rank 49 down to some reasonable minimum
    # For 2x5x5, theoretical lower bound is max(2*5, 5*5, 5*2) = 25
    # but practical lower bound for matrix mult is higher
    
    min_rank_to_try = 30  # Don't go too low, unlikely to find
    
    for rank in range(n*m*p - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, deadline, max_trials=50)
        if result is not None:
            best = result
            # Found one, try to go lower
            continue
        else:
            # Could not find at this rank, try a few more ranks lower
            # but don't spend too much time
            break
    
    return best
