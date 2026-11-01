"""Matrix multiplication 2x3 x 3x10 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition with rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using float arithmetic for screening."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_joint_row_exploitation(n, m, p, time_budget, seed=42):
    """
    Exploit the small n=2 dimension by trying to find a decomposition where
    the rank is less than n*m*p by sharing structure across the two rows.
    
    For n=2, we have two row-vector multiplications:
    Row 0: (1x3) * (3x10) -> 1x10, rank 30 schoolbook
    Row 1: (1x3) * (3x10) -> 1x10, rank 30 schoolbook
    
    Total schoolbook: 60. We try to find shared intermediate terms.
    
    Strategy: For each k (column of output), we need to compute:
    (A[0,:] * B[:,k]) and (A[1,:] * B[:,k])
    
    We look for decompositions where some of the intermediate products are shared.
    This is equivalent to finding a tensor decomposition of the 2x3x10 tensor
    with rank < 60.
    
    We use a structured search: try to find linear combinations of the
    intermediate terms that satisfy both rows simultaneously.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}
    # We want to find U, V, W with rank r < 60 such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from 54 down to 48 (exploiting n=2 structure)
    for r in range(54, 47, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random sparse matrices with ±1 entries
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Generate random ±1 matrices
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            # Make some entries zero to increase sparsity
            U[rng.random((n*m, r)) < 0.3] = 0
            V[rng.random((m*p, r)) < 0.3] = 0
            W[rng.random((p*n, r)) < 0.3] = 0
            
            # Quick verification
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Exact verification
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try structured approach: fix some entries and solve for others
        if time.time() > deadline - 5:
            break
        
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start with a structured pattern
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Assign random ±1 to a subset of entries
            for c in range(r):
                # Choose which (i,j) pairs to activate in U
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.4:
                            U[i*m+j, c] = rng.choice([-1, 1])
                
                # Choose which (j,k) pairs to activate in V
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.2:
                            V[j*p+k, c] = rng.choice([-1, 1])
                
                # Choose which (k,i) pairs to activate in W
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.4:
                            W[k*n+i, c] = rng.choice([-1, 1])
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_winograd_style(n, m, p, time_budget, seed=42):
    """
    Try Winograd-style transforms to reduce rank.
    
    For matrix multiplication, Winograd's method can reduce the number of
    multiplications by using convolution-like transforms.
    
    For 2x3 * 3x10, we can try to group the k-dimension and use
    transforms to share intermediate results.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    
    # Try different block sizes for the k-dimension
    for block_size in [2, 3, 5]:
        if time.time() > deadline - 5:
            break
        
        n_blocks = p // block_size
        if n_blocks * block_size != p:
            continue
        
        # For each block of k-values, try to use a Winograd-like transform
        # The idea: instead of computing each k separately, use a transform
        # that combines multiple k-values with shared intermediate terms.
        
        # This is a heuristic; we try random ±1 patterns with block structure
        for r in range(50, 55):
            if time.time() > deadline - 5:
                break
            
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                
                # Assign entries with block structure in k
                for c in range(r):
                    # U: random ±1
                    for i in range(n):
                        for j in range(m):
                            if rng.random() < 0.5:
                                U[i*m+j, c] = rng.choice([-1, 1])
                    
                    # V: block structure in k
                    for j in range(m):
                        for kb in range(n_blocks):
                            # Assign same value to all k in the block
                            if rng.random() < 0.4:
                                val = rng.choice([-1, 1])
                                for k in range(kb*block_size, (kb+1)*block_size):
                                    V[j*p+k, c] = val
                    
                    # W: block structure in k
                    for i in range(n):
                        for kb in range(n_blocks):
                            if rng.random() < 0.5:
                                val = rng.choice([-1, 1])
                                for k in range(kb*block_size, (kb+1)*block_size):
                                    W[k*n+i, c] = val
                
                if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try joint row exploitation
    time_left = deadline - time.time()
    if time_left > 10:
        result = _try_joint_row_exploitation(n, m, p, time_left, seed=12345)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    # Try Winograd-style if time remains
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_winograd_style(n, m, p, time_left, seed=99999)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    # One more attempt with different seed if time remains
    time_left = deadline - time.time()
    if time_left > 10 and best_rank == n * m * p:
        result = _try_joint_row_exploitation(n, m, p, time_left, seed=77777)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return best
# EVOLVE-BLOCK-END
