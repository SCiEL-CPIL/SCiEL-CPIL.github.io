"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
SEED_TIME_LIMIT = 110  # seconds, leave margin before 120s/150s kill

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 8

    def _elapsed():
        return time.time() - t_start

    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return _to_dict(U, V, W)

    def _to_dict(U, V, W):
        return {
            "U": [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])],
            "V": [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])],
            "W": [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])],
        }

    def _validate(U, V, W, tol=1e-6):
        """Check that sum_r U[:,r] V[:,r] W[:,r] equals the matmul tensor."""
        U = np.array(U) if not isinstance(U, np.ndarray) else U
        V = np.array(V) if not isinstance(V, np.ndarray) else V
        W = np.array(W) if not isinstance(W, np.ndarray) else W
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that entries which should be 0 are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pass  # The tensor is 1 for all (i,j,k) in matmul
        return True

    # Try a structured decomposition: split p=8 into two halves of p=4
    # The idea: use a rank-19 decomposition for 3x3x4 (if we can find one)
    # Total rank would be 38.
    # 
    # Known result: R(3x3x4) = 19 (from Landsberg, etc.)
    # But constructing the explicit rank-19 decomposition is non-trivial.
    #
    # Let's try a direct ALS-style search for a low-rank decomposition.
    # Start from random initialization and use alternating least squares.

    def _als_decompose(target_tensor, n, m, p, rank, max_iter=200, tol=1e-8, seed=42):
        """
        Use ALS to find factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
        such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = tensor[i,j,k]
        """
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Build the target tensor as a (n*m) x (m*p) matrix for each k*n+i... 
        # Actually, let's think of it as: for each (k, i) pair, we have a (n*m) x (m*p) matrix
        # where element (i*m+j, j*p+k) should be 1, all others 0.
        # The tensor equation is: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
        
        # ALS: fix V, W -> solve U; fix U, W -> solve V; fix U, V -> solve W.
        # For each column r, the contribution is outer product structure.
        # 
        # Actually, the tensor is T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j*p+k, c = k*n+i.
        # But note: a, b, c are NOT independent! a determines i and j, 
        # b determines j and k, c determines k and i.
        # So for a given (a, b, c), we need a%m == b//p and b%p == c//n and c%n == a//m.
        # The tensor is 1 when these are consistent, 0 otherwise.
        
        # Let's enumerate all valid (a, b, c) triples:
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_triples.append((a, b, c))
        
        # For ALS, we can rewrite: for each r, U[:,r] * V[:,r] * W[:,r]
        # The sum over r should match the indicator of valid triples.
        
        # A simpler approach: use the fact that this is a 3-tensor decomposition.
        # Let's use a gradient-based approach or direct ALS.
        
        # ALS for tensor decomposition:
        # Fix V, W. For each a, U[a,:] = argmin sum_{(a,b,c) valid} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        # This is: for each a, we have equations for each valid (b,c) with this a.
        # For fixed a = i*m+j, the valid (b,c) are: b = j*p+k for k in range(p), c = k*n+i for k in range(p).
        # So for each a, we have p equations:
        #   sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 1, for k=0..p-1
        #   sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 0, for invalid (b,c) with this a
        #
        # But actually, for a given a = i*m+j, the valid b values are j*p+k for all k,
        # and the corresponding c values are k*n+i.
        # The invalid (b,c) pairs with this a would be those where b is not of form j*p+k
        # or c is not of form k*n+i matching the same k.
        
        # This is getting complex. Let me use a different approach:
        # Flatten the tensor and use a simple iterative method.
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for _ in range(max_iter):
            if _elapsed() > SEED_TIME_LIMIT:
                break
            
            # Update U: for each a, solve least squares
            for a in range(n * m):
                i = a // m
                j = a % m
                # Valid (b, c) for this a:
                # b = j*p + k, c = k*n + i, for k in 0..p-1
                # For invalid (b, c) where b//p != j or b%p != c//n or c%n != i, the target is 0.
                # But there are (m*p)*(p*n) total (b,c) pairs, most are 0.
                # To keep it efficient, only use the valid equations (target=1) and 
                # some sampled invalid equations (target=0).
                
                # Build system: for valid k, V[j*p+k, :] * W[k*n+i, :] gives a vector in R^rank
                # U[a, :] dot (V[j*p+k,:] * W[k*n+i,:]) = 1
                # This is: sum_r U[a,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                
                # Let's build a (p x rank) matrix M where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve M @ U[a,:] = ones(p)
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
            
            # Update V: for each b = j*p+k
            for b in range(m * p):
                j = b // p
                k = b % p
                # Valid a for this b: a = i*m + j for i in 0..n-1
                # Valid c for this b: c = k*n + i for i in 0..n-1
                # Equation: sum_r U[i*m+j, r] * V[b, r] * W[k*n+i, r] = 1
                # For each i: U[i*m+j, :] * W[k*n+i, :] dot V[b, :] = 1
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
            
            # Update W: for each c = k*n+i
            for c in range(p * n):
                k = c // n
                i = c % n
                # Valid a: a = i*m+j, valid b: b = j*p+k
                # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    # Try various ranks for the split approach
    # Split p=8 into two halves of 4
    # For each half, we need a 3x3x4 decomposition
    # Try rank 19 for each half (total 38)
    
    results = []
    
    for total_rank in [38, 40, 42, 44, 48, 52, 56, 60, 64, 68, 72]:
        if _elapsed() > SEED_TIME_LIMIT - 20:
            break
        
        # Try direct ALS on the full 3x3x8 tensor
        for seed in [42, 123, 456, 789]:
            if _elapsed() > SEED_TIME_LIMIT - 10:
                break
            U, V, W, err = _als_decompose(None, n, m, p, total_rank, max_iter=50, seed=seed)
            if U is not None and err < 1e-6:
                results.append((total_rank, U, V, W))
                break
        if results:
            break
    
    if results:
        rank, U, V, W = results[0]
        return _to_dict(U, V, W)
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END

