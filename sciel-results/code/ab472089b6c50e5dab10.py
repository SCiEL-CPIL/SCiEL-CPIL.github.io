import numpy as np
import time
import random

def _elapsed():
    return time.time()

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _is_valid_entry(a, b, c, n, m, p):
    ia, ja = divmod(a, m)
    jb, kb = divmod(b, p)
    kc, ic = divmod(c, n)
    return ja == jb and ia == ic

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    # Check main condition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: for all (a,b,c) not valid, sum should be 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                ia, ja = divmod(a, m)
                jb, kb = divmod(b, p)
                kc, ic = divmod(c, n)
                if ja == jb and ia == ic:
                    continue
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(s) > 1e-9:
                    return False
    
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification using matrix multiplication structure."""
    # Build the tensor T where T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Check that T[i*m+j, j*p+k, k*n+i] == 1 for all i,j,k
    # and T[a,b,c] == 0 for all other (a,b,c)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Sample off-diagonal entries
    random.seed(42)
    for _ in range(500):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        ia, ja = divmod(a, m)
        jb, kb = divmod(b, p)
        kc, ic = divmod(c, n)
        if ja == jb and ia == ic:
            continue
        s = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(s) > 1e-9:
            return False
    
    return True

def _snap_to_half(X):
    """Snap values to nearest half-integer."""
    return np.round(X * 2) / 2

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_local_search(n, m, p, rank, seed=0, time_limit=5.0):
    """
    Try to find a rank-`rank` decomposition using local search.
    
    Uses a structured initialization based on random permutations and
    then performs coordinate descent.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Random initialization with some structure
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize columns to avoid scale issues
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-10:
            U[:, r] /= norm_u
        if norm_v > 1e-10:
            V[:, r] /= norm_v
        if norm_w > 1e-10:
            W[:, r] /= norm_w
        # Scale back to maintain product
        scale = (norm_u * norm_v * norm_w) ** (1/3) if norm_u * norm_v * norm_w > 1e-30 else 1.0
        U[:, r] *= scale ** 0.5
        V[:, r] *= scale ** 0.5
        W[:, r] *= scale ** 0.5
    
    best_err = float('inf')
    
    def compute_error(U, V, W):
        err = 0.0
        # On-diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        # Off-diagonal (sampled)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    ia, ja = divmod(a, m)
                    jb, kb = divmod(b, p)
                    kc, ic = divmod(c, n)
                    if ja == jb and ia == ic:
                        continue
                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += s ** 2
        return err
    
    # Coordinate descent
    step_size = 0.5
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Pick random matrix, row, col
        which = rng.randint(3)
        if which == 0:
            mat, other1, other2 = U, V, W
        elif which == 1:
            mat, other1, other2 = V, U, W
        else:
            mat, other1, other2 = W, U, V
        
        row = rng.randint(mat.shape[0])
        col = rng.randint(mat.shape[1])
        
        # Try a few perturbations
        best_local = mat[row, col]
        best_local_err = best_err
        
        for _ in range(10):
            delta = rng.randn() * step_size
            mat[row, col] += delta
            new_err = compute_error(U, V, W)
            if new_err < best_local_err:
                best_local_err = new_err
                best_local = mat[row, col]
            else:
                mat[row, col] = best_local
        
        mat[row, col] = best_local
        best_err = best_local_err
        
        if best_err < 1e-12:
            break
    
    # Try to round to half-integers
    if best_err < 1e-6:
        U_r = _snap_to_half(U)
        V_r = _snap_to_half(V)
        W_r = _snap_to_half(W)
        
        if _verify_fast(U_r, V_r, W_r, n, m, p, rank):
            return U_r, V_r, W_r
    
    return None

def _try_rank_algebraic(n, m, p, rank, seed=0, time_limit=5.0):
    """
    Try an algebraic-inspired approach: use structured patterns
    that might yield lower rank.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Try block-diagonal structure
    # Partition the rank into blocks and assign each block to a subset of indices
    
    # For now, fall back to local search with different initialization
    return _try_rank_local_search(n, m, p, rank, seed=seed+1000, time_limit=time_limit)

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)
    
    # Theoretical lower bound
    # For matrix multiplication, rank >= n*m*p / max(n,m,p) is a loose bound
    # Better: rank >= n*m + m*p + p*n - n - m - p + 1 (for certain cases)
    # For 3x5x6: 15 + 30 + 18 - 3 - 5 - 6 + 1 = 50
    # Let's try ranks from 89 down to 50
    
    # Try a few ranks with local search
    ranks_to_try = [85, 80, 75, 70, 65, 60, 55, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try with a few different seeds
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            res = _try_rank_local_search(n, m, p, rank, seed=rank + attempt * 100, time_limit=10.0)
            
            if res is not None:
                U, V, W = res
                best_rank = rank
                best_result = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    # If no lower rank found, return schoolbook
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
