"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Need: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Vectorized: reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)  # U3[i,j,t]
    V3 = V_arr.reshape(m, p, rank)  # V3[j,k,t]
    W3 = W_arr.reshape(p, n, rank)  # W3[k,i,t]
    
    # Compute T[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    # Use einsum: 'ijt,jkt,kit->ijk'
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=1e-7)

def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification assuming numpy arrays already."""
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-7

def _als_update(U, V, W, n, m, p):
    """One full ALS sweep: update U, V, W in sequence. Returns new U, V, W."""
    rank = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U: for each (i,j), solve A @ x = b where A[k,t] = V[j*p+k,t]*W[k*n+i,t]
    for i in range(n):
        for j in range(m):
            # A is (p, rank): A[k,t] = V[j*p+k,t] * W[k*n+i,t]
            A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :].T  # (p, rank)
            b = np.ones(p)
            # Solve min ||A x - b||
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                U_new[i*m + j, :] = x
            except:
                pass
    
    # Update V: for each (j,k), solve A @ x = b where A[i,t] = U[i*m+j,t]*W[k*n+i,t]
    for j in range(m):
        for k in range(p):
            A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :].T  # (n, rank)
            b = np.ones(n)
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                V_new[j*p + k, :] = x
            except:
                pass
    
    # Update W: for each (k,i), solve A @ x = b where A[j,t] = U[i*m+j,t]*V[j*p+k,t]
    for k in range(p):
        for i in range(n):
            A = U[i*m + np.arange(m), :] * V[np.arange(m)*p + k, :].T  # (m, rank)
            b = np.ones(m)
            try:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                W_new[k*n + i, :] = x
            except:
                pass
    
    return U_new, V_new, W_new

def _als_update_vectorized(U, V, W, n, m, p):
    """Vectorized ALS update for speed."""
    rank = U.shape[1]
    
    # Precompute index arrays
    idx_i = np.arange(n)
    idx_j = np.arange(m)
    idx_k = np.arange(p)
    
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)
    
    # Update U[i*m+j, :] : A[k,t] = V[j*p+k,t]*W[k*n+i,t], solve A x = 1_p
    for i in range(n):
        for j in range(m):
            # V[j*p+k, :] for k=0..p-1
            V_slice = V[j*p:(j+1)*p, :]  # (p, rank)
            # W[k*n+i, :] for k=0..p-1
            W_slice = W[np.arange(p)*n + i, :]  # (p, rank)
            A = V_slice * W_slice  # (p, rank)
            b = np.ones(p)
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                U_new[i*m+j, :] = U[i*m+j, :]
    
    # Update V[j*p+k, :] : A[i,t] = U[i*m+j,t]*W[k*n+i,t], solve A x = 1_n
    for j in range(m):
        for k in range(p):
            U_slice = U[np.arange(n)*m + j, :]  # (n, rank)
            W_slice = W[k*n + np.arange(n), :]  # (n, rank)
            A = U_slice * W_slice  # (n, rank)
            b = np.ones(n)
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                V_new[j*p+k, :] = V[j*p+k, :]
    
    # Update W[k*n+i, :] : A[j,t] = U[i*m+j,t]*V[j*p+k,t], solve A x = 1_m
    for k in range(p):
        for i in range(n):
            U_slice = U[i*m + np.arange(m), :]  # (m, rank)
            V_slice = V[np.arange(m)*p + k, :]  # (m, rank)
            A = U_slice * V_slice  # (m, rank)
            b = np.ones(m)
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                W_new[k*n+i, :] = W[k*n+i, :]
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, n, m, p):
    """Compute max absolute error of the decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))

def _try_rank(r, n, m, p, max_iter=200, seed=None):
    """Try to find a rank-r decomposition using ALS."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize: use structured initialization based on block structure
    # For matrix multiplication, we can initialize with the "obvious" pattern
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Structured init: distribute the rank across (i,j,k) triples
    # Each term t corresponds to some (i_t, j_t, k_t)
    # U[i*m+j, t] = 1 if i==i_t and j==j_t, else small random
    # V[j*p+k, t] = 1 if j==j_t and k==k_t, else small random
    # W[k*n+i, t] = 1 if k==k_t and i==i_t, else small random
    
    # Assign each rank term to a (i,j,k) triple cyclically
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for t in range(r):
        i_t, j_t, k_t = triples[t % len(triples)]
        U[i_t*m + j_t, t] = 1.0
        V[j_t*p + k_t, t] = 1.0
        W[k_t*n + i_t, t] = 1.0
        # Add small random perturbation to break symmetry
        U[i_t*m + j_t, t] += np.random.randn() * 0.01
        V[j_t*p + k_t, t] += np.random.randn() * 0.01
        W[k_t*n + i_t, t] += np.random.randn() * 0.01
    
    # Also add small random values elsewhere to allow exploration
    U += np.random.randn(n*m, r) * 0.005
    V += np.random.randn(m*p, r) * 0.005
    W += np.random.randn(p*n, r) * 0.005
    
    best_err = _compute_error(U, V, W, n, m, p)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        U, V, W = _als_update_vectorized(U, V, W, n, m, p)
        
        if iteration % 5 == 0 or iteration == max_iter - 1:
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
            if err < 1e-8:
                return U, V, W, err
    
    return None, None, None, best_err

def _try_rank_random_init(r, n, m, p, max_iter=150, seed=None):
    """Try with pure random initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    best_err = _compute_error(U, V, W, n, m, p)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        U, V, W = _als_update_vectorized(U, V, W, n, m, p)
        
        if iteration % 5 == 0 or iteration == max_iter - 1:
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
            if err < 1e-8:
                return U, V, W, err
    
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try ranks from lower to higher (we want minimum rank)
    # Theoretical lower bound: for (n,m,p) matrix mult, rank >= (n+m+p)
    # For 4,4,8: lower bound is 16, but actual minimum is likely higher
    # Try a range of ranks
    
    ranks_to_try = []
    # Try from a reasonable lower bound up to schoolbook
    lower = max(20, n + m + p)
    upper = schoolbook_rank
    
    # Try a spread of ranks, starting from lower
    # We'll try: lower, lower+5, lower+10, ..., up to upper
    step = 5
    for r in range(lower, upper + 1, step):
        ranks_to_try.append(r)
    if upper not in ranks_to_try:
        ranks_to_try.append(upper)
    
    # Also try some specific values
    for r in [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]:
        if r >= lower and r <= upper and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try = sorted(set(ranks_to_try))
    
    for r in ranks_to_try:
        if time.time() > _DEADLINE - 15:
            break
        
        if r >= best_rank:
            break  # No point trying ranks >= current best
        
        found = False
        # Try multiple initializations
        for seed in range(3):
            if time.time() > _DEADLINE - 10:
                break
            
            # Structured init
            U, V, W, err = _try_rank(r, n, m, p, max_iter=100, seed=seed)
            if U is not None and _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
            
            # Random init
            U, V, W, err = _try_rank_random_init(r, n, m, p, max_iter=80, seed=seed+100)
            if U is not None and _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
        
        if found:
            # Found a valid decomposition for this rank, try to go lower
            continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
