"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Reshape for broadcasting: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # We need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # Create index arrays
    i_idx = np.arange(n).reshape(-1, 1, 1)  # (n,1,1)
    j_idx = np.arange(m).reshape(1, -1, 1)  # (1,m,1)
    k_idx = np.arange(p).reshape(1, 1, -1)  # (1,1,p)
    
    # Compute U indices: i*m+j -> shape (n,m,1)
    U_idx = (i_idx * m + j_idx)  # (n,m,1)
    # V indices: j*p+k -> shape (1,m,p)
    V_idx = (j_idx * p + k_idx)  # (1,m,p)
    # W indices: k*n+i -> shape (n,1,p)
    W_idx = (k_idx * n + i_idx)  # (n,1,p)
    
    # Extract: U[U_idx, c] for all c -> (n,m,r)
    # V[V_idx, c] for all c -> (1,m,p,r) -> need to broadcast to (n,m,p,r)
    # W[W_idx, c] for all c -> (n,1,p,r) -> need to broadcast to (n,m,p,r)
    
    # U_expanded: (n,m,r)
    U_exp = U[U_idx, :]  # (n,m,r)
    # V_expanded: (1,m,p,r) -> broadcast to (n,m,p,r)
    V_exp = V[V_idx, :]  # (1,m,p,r)
    # W_expanded: (n,1,p,r) -> broadcast to (n,m,p,r)
    W_exp = W[W_idx, :]  # (n,1,p,r)
    
    # Sum over c: (n,m,p)
    result = np.sum(U_exp[:, :, np.newaxis, :] * V_exp * W_exp, axis=-1)
    
    return np.allclose(result, 1.0, atol=tol)

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    target_rank = n * m * p  # 36 for schoolbook
    
    # Build the schoolbook solution as fallback
    rank_sb = target_rank
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {
        'U': U_sb.tolist(),
        'V': V_sb.tolist(),
        'W': W_sb.tolist()
    }
    best_rank = target_rank
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) / max(n,m,p) is too loose
    # For <2,3,6>, let's try from rank 12 up (since p*n=12, m*p=18, n*m=6)
    # Actually, a reasonable lower bound is max(n*m, m*p, p*n) / max(n,m,p) = 18/6 = 3
    # But practically, let's start from a more reasonable range
    
    # Try ranks from 12 to 35
    for r in range(12, target_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_trials = 100 if r < 20 else 50 if r < 30 else 20
        
        for trial in range(max_trials):
            if time.time() > deadline - 2:
                break
            
            # Use structured random initialization with integer/half-integer values
            np.random.seed(42 + trial * 1000 + r)
            
            # Try different value sets
            if trial % 3 == 0:
                values = np.array([-1, 0, 1])
            elif trial % 3 == 1:
                values = np.array([-1, -0.5, 0, 0.5, 1])
            else:
                values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
            
            U_cand = np.random.choice(values, size=(n * m, r))
            V_cand = np.random.choice(values, size=(m * p, r))
            W_cand = np.random.choice(values, size=(p * n, r))
            
            # Fast verification
            if _verify_fast(U_cand, V_cand, W_cand, n, m, p):
                best_rank = r
                best_result = {
                    'U': U_cand.tolist(),
                    'V': V_cand.tolist(),
                    'W': W_cand.tolist()
                }
                break
        
        if best_rank == r:
            break  # Found minimal rank for this search
    
    return best_result
# EVOLVE-BLOCK-END
