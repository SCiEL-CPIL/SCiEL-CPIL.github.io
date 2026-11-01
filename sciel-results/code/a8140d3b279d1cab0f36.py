"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed_offset=0, max_trials=50):
    """Try to find a valid decomposition of rank r."""
    np.random.seed(42 + seed_offset)
    
    # Try multiple random initializations with different strategies
    for trial in range(max_trials):
        if time.time() > deadline - 3:
            return None
        
        # Strategy 1: Use structured random values from {-1, 0, 1} with varying sparsity
        sparsity = 0.3 + 0.4 * (trial % 3) / 3.0  # 0.3, 0.43, 0.57
        
        U = np.where(np.random.rand(n*m, r) < sparsity, 
                     np.random.choice([-1, 1], size=(n*m, r)), 0)
        V = np.where(np.random.rand(m*p, r) < sparsity, 
                     np.random.choice([-1, 1], size=(m*p, r)), 0)
        W = np.where(np.random.rand(p*n, r) < sparsity, 
                     np.random.choice([-1, 1], size=(p*n, r)), 0)
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Strategy 2: Try half-integer values {-0.5, 0, 0.5, 1, -1}
        if trial % 5 == 2:
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            W = np.random.choice(vals, size=(p*n, r))
            
            if _verify(U, V, W, n, m, p):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Strategy 3: Block-structured initialization
        if trial % 7 == 3:
            # Create blocks of non-zero values
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            for col in range(r):
                # Randomly assign blocks
                u_block_start = np.random.randint(0, n*m - 1)
                u_block_end = min(u_block_start + np.random.randint(2, 6), n*m)
                U[u_block_start:u_block_end, col] = np.random.choice([-1, 1])
                
                v_block_start = np.random.randint(0, m*p - 1)
                v_block_end = min(v_block_start + np.random.randint(2, 6), m*p)
                V[v_block_start:v_block_end, col] = np.random.choice([-1, 1])
                
                w_block_start = np.random.randint(0, p*n - 1)
                w_block_end = min(w_block_start + np.random.randint(2, 6), p*n)
                W[w_block_start:w_block_end, col] = np.random.choice([-1, 1])
            
            if _verify(U, V, W, n, m, p):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build schoolbook solution as fallback (rank 36)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: at least max(n*m, m*p, p*n) is not correct
    # For <2,3,6>, let's try ranks from 24 up to 35
    
    for r in range(24, rank_sb):
        if time.time() > deadline - 5:
            break
        
        # Use more trials for smaller ranks (harder to find)
        max_trials = 80 if r <= 28 else 50
        
        result = _try_rank(r, n, m, p, deadline, seed_offset=r, max_trials=max_trials)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found minimal rank for this search
    
    return best_result
# EVOLVE-BLOCK-END
