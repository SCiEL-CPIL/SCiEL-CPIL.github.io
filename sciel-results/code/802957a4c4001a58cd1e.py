"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE_S = 150.0
SAFETY_MARGIN_S = 2.0
EPS = 1e-12
MAX_RANK = 54

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False, "Dimension mismatch"
    if U.shape[1] != r or V.shape[1] != r or W.shape[1] != r:
        return False, "Rank mismatch"
    
    # Build the multiplication tensor T
    # T[i, j, k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # In the trace convention: W[k*n+i, c]
    # The tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal delta[i,k] for all i, j, k
    # Actually, the multiplication tensor M satisfies:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T_{i,j,k} = delta_{i,k} (for the structure of multiplication)
    # But the decomposition must satisfy:
    # T_{i,j,k} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{i,k}^{(c)}  (with appropriate indexing)
    
    # Let's construct the full tensor and check
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # The decomposition gives for each (i, j, k):
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (note W is indexed as W[k*n+i] = W_reshaped[k, i])
    
    # Build target tensor: T[i,j,k] = 1 if the product A[i,j]*B[j,k] is part of C[i,k]
    # For standard matrix multiplication, every (i,j,k) triple contributes exactly once
    # so T[i,j,k] = 1 for all i,j,k
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} has:
    # T_{i,j,k} = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # So every triple (i,j,k) has coefficient 1.
    
    target = np.ones((n, m, p))
    
    # Compute decomposed tensor
    # D[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    D = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                D[i, j, k] = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
    
    if not np.allclose(D, target, atol=1e-9, rtol=1e-9):
        return False, "Tensor mismatch"
    
    return True, "OK"

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _random_search(n, m, p, max_time=5.0):
    """Try random rank-reduction via low-rank approximations.
    Returns (U, V, W) or None if no improvement found."""
    start = time.time()
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from a few below schoolbook down
    for r in range(min(best_rank - 1, 53), max(20, best_rank - 10) - 1, -1):
        if time.time() - start > max_time:
            break
        for trial in range(5):
            if time.time() - start > max_time:
                break
            # Try to find a rank-r decomposition
            # This is very hard in general; we'll just do a simple heuristic
            # Generate random factor matrices and check
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            valid, msg = _verify(U_rand, V_rand, W_rand, n, m, p)
            if valid:
                best_rank = r
                best_result = (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
                break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Verify it's correct
    valid, msg = _verify(U, V, W, n, m, p)
    if not valid:
        # This should never happen
        pass
    
    # Try to improve with random search (very unlikely to work, but cheap to try)
    start = time.time()
    improved = _random_search(n, m, p, max_time=min(5.0, DEADLINE_S - (time.time() - start) - SAFETY_MARGIN_S))
    
    if improved is not None:
        U, V, W = improved
    
    # Ensure we return lists of lists with integer/half-integer entries
    # The schoolbook solution uses all 0s and 1s, so it's fine.
    # For any random solution, we might need to round.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
