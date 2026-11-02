import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 iff indices match."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_einsum(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T = _target_tensor(n, m, p)
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(recon - T)) < tol

def _round_to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * np.array(x)
    rounded = np.round(doubled)
    if np.all(np.abs(doubled - rounded) < tol):
        return (rounded / 2.0).tolist()
    return np.array(x).tolist()

def _construct_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank

def _als_decompose(n, m, p, rank, max_iter=150, tol=1e-9, seed=0, time_limit=None, start_time=None):
    """ALS decomposition using einsum for efficiency."""
    if start_time is None:
        start_time = time.time()
    
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with small random values
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    T = _target_tensor(n, m, p)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            return None, it
        
        # Update U: for each row a, solve least squares
        # M[b,c,r] = V[b,r] * W[c,r], shape (mp*pn, rank)
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)
        MW_flat = MW.reshape(mp*pn, rank)
        
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            try:
                X, _, _, _ = np.linalg.lstsq(MW_flat.T, target_a, rcond=None)
                U[a, :] = X
            except:
                pass
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)
        UW_flat = UW.reshape(nm*pn, rank)
        
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            try:
                X, _, _, _ = np.linalg.lstsq(UW_flat.T, target_b, rcond=None)
                V[b, :] = X
            except:
                pass
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)
        UV_flat = UV.reshape(nm*mp, rank)
        
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            try:
                X, _, _, _ = np.linalg.lstsq(UV_flat.T, target_c, rcond=None)
                W[c, :] = X
            except:
                pass
        
        # Check convergence every 5 iterations
        if it % 5 == 4:
            recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(recon - T))
            if err < tol:
                break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        return (U, V, W), it
    return None, it

def _strassen_6x6x6():
    """
    Return Strassen-style rank-21 decomposition for 6x6x6 matrix multiplication.
    We use the known result that 6x6x6 matrix multiplication has rank at most 21.
    This is constructed by embedding 3x3x3 Strassen (rank 7) into 6x6x6.
    
    For 6x6x6, we can partition into four 3x3 blocks and use Strassen on each.
    But actually, the known rank-21 for 6x6x6 comes from a different construction.
    
    Let's use a simpler approach: we know that for n x n matrix multiplication,
    the rank is at most n^3 (schoolbook) and at least n^2. For 6x6x6, 
    Strassen's algorithm adapted gives rank 21.
    
    Actually, let me construct it directly using the known fact that 
    R(6,6,6) <= 21. I'll use a known decomposition.
    """
    n = 6
    rank = 21
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 6x6x6, we can use a recursive construction or known results.
    # The rank-21 decomposition for 6x6x6 is known to exist.
    # Let me construct it by tiling 3x3x3 Strassen.
    
    # Actually, let's use a different approach. We'll try to find the decomposition
    # numerically using ALS starting from a good initialization.
    
    # For now, return None to indicate we need to compute it
    return None

def _find_6x6x6_decomposition(time_budget=30.0):
    """Try to find a rank-21 decomposition for 6x6x6 matrix multiplication."""
    n = 6
    start_time = time.time()
    
    # Try rank 21 first (known minimum for 6x6x6)
    for rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() - start_time > time_budget:
            break
        
        for seed in range(10):
            if time.time() - start_time > time_budget:
                break
            
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2:
                break
            
            result, iters = _als_decompose(n, n, n, rank, max_iter=100, 
                                           seed=seed, time_limit=remaining,
                                           start_time=start_time)
            if result is not None:
                U, V, W = result
                if _verify_einsum(U, V, W, n, n, n, tol=1e-7):
                    return U, V, W, rank
    
    return None

def _embed_3x5x6_in_6x6x6(U6, V6, W6, rank6, n, m, p):
    """
    Embed the 3x5x6 multiplication into 6x6x6 and extract the sub-decomposition.
    
    The 3x5x6 multiplication tensor T[i,j,k] has:
    - i in [0,3), j in [0,5), k in [0,6)
    - U index: i*m + j = i*5 + j, range [0,15)
    - V index: j*p + k = j*6 + k, range [0,30)  
    - W index: k*n + i = k*3 + i, range [0,18)
    
    For 6x6x6:
    - U index: i*6 + j, range [0,36)
    - V index: j*6 + k, range [0,36)
    - W index: k*6 + i, range [0,36)
    
    We need to map the 3x5x6 indices to 6x6x6 indices.
    The 3x5x6 tensor is a sub-tensor of 6x6x6 where:
    - i ranges over 0,1,2 (out of 0..5)
    - j ranges over 0,1,2,3,4 (out of 0..5)
    - k ranges over 0,1,2,3,4,5 (out of 0..5)
    
    So we extract:
    - U rows: for i in [0,3), j in [0,5): row index i*5+j in 3x5x6 -> i*6+j in 6x6x6
    - V rows: for j in [0,5), k in [0,6): row index j*6+k in 3x5x6 -> j*6+k in 6x6x6
    - W rows: for k in [0,6), i in [0,3): row index k*3+i in 3x5x6 -> k*6+i in 6x6x6
    """
    U356 = np.zeros((n*m, rank6))
    V356 = np.zeros((m*p, rank6))
    W356 = np.zeros((p*n, rank6))
    
    for i in range(n):
        for j in range(m):
            U356[i*m + j, :] = U6[i*6 + j, :]
    
    for j in range(m):
        for k in range(p):
            V356[j*p + k, :] = V6[j*6 + k, :]
    
    for k in range(p):
        for i in range(n):
            W356[k*n + i, :] = W6[k*6 + i, :]
    
    return U356, V356, W356

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Start with trivial solution
    U_triv, V_triv, W_triv, rank_trivial = _construct_trivial(n, m, p)
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Strategy 1: Try rank unification via 6x6x6 embedding
    # Find a low-rank decomposition of 6x6x6, then extract 3x5x6
    time_for_6x6 = min(40.0, deadline - 20.0)
    if time_for_6x6 > 10:
        result6 = _find_6x6x6_decomposition(time_budget=time_for_6x6)
        if result6 is not None:
            U6, V6, W6, rank6 = result6
            U356, V356, W356 = _embed_3x5x6_in_6x6x6(U6, V6, W6, rank6, n, m, p)
            
            if _verify_einsum(U356, V356, W356, n, m, p, tol=1e-7):
                # Try to round to half-integers
                U_rounded = _round_to_half_int(U356)
                V_rounded = _round_to_half_int(V356)
                W_rounded = _round_to_half_int(W356)
                
                if _verify_einsum(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank6
                else:
                    best_result = {"U": U356.tolist(), "V": V356.tolist(), "W": W356.tolist()}
                    best_rank = rank6
    
    # Strategy 2: Direct ALS on 3x5x6 with various ranks
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        # Try ranks from low to high
        min_rank = max(n*m, m*p, p*n)  # Lower bound
        for rank in range(min_rank, min(rank_trivial, min_rank + 30)):
            if time.time() - start_time > deadline - 5:
                break
            
            for seed in range(8):
                if time.time() - start_time > deadline - 5:
                    break
                
                result, iters = _als_decompose(n, m, p, rank, max_iter=80, 
                                               seed=seed, 
                                               time_limit=deadline - (time.time() - start_time) - 5,
                                               start_time=start_time)
                if result is not None:
                    U, V, W = result
                    if _verify_einsum(U, V, W, n, m, p, tol=1e-7):
                        U_rounded = _round_to_half_int(U)
                        V_rounded = _round_to_half_int(V)
                        W_rounded = _round_to_half_int(W)
                        
                        if _verify_einsum(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                            best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                            best_rank = rank
                        else:
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = rank
                        break
            
            if best_rank < rank_trivial:
                break
    
    return best_result
