"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150  # seconds wall-clock limit
SAFETY_MARGIN = 5  # seconds to return before deadline

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W (each of shape (rows, rank)) form an exact decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all i,j,k.
    """
    r = U.shape[1]
    # Build the tensor T from the decomposition
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        # Reshape
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        w_mat = wc.reshape(p, n)
        # For each c, contribution to T[i,j,k] is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # We can compute this as: for each i, j: u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        # v_mat[j,:] has shape (p,), w_mat[:,i] has shape (p,), their element-wise product has shape (p,)
        # So T[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_mat[i, j] * (v_mat[j, :] * w_mat[:, i])
    return np.allclose(T, 1.0, atol=tol)


def _try_block_decomposition(n, m, p, max_rank=None, time_limit=30):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    For 3x4 x 4x5, we can try various rank values and use random initialization
    with a simple optimization approach.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    if max_rank is None:
        max_rank = best_rank
    
    # Try ranks from a lower bound up to best_rank - 1
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But we can try a range of ranks
    min_rank_to_try = max(n * m, m * p, p * n)  # = 20
    
    # Try a few specific ranks
    ranks_to_try = list(range(min_rank_to_try, best_rank))
    # Also try some intermediate values
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        if rank >= best_rank:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Simple gradient-free search: try to adjust to make T close to 1
            # This is a rough approach; for a proper solution we'd need more sophisticated methods
            # For now, just check if by chance we get close (very unlikely)
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    return best_rank, best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try to find a better decomposition within time budget
    time_remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if time_remaining > 5:
        rank, U, V, W = _try_block_decomposition(n, m, p, time_limit=time_remaining)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Convert to lists of lists for the expected format
    U_list = [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
