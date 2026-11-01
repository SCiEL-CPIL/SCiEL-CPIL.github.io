import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        result = np.zeros_like(T)
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(result, T, atol=tol)

    # Strategy: exploit the 2x16 structure.
    # The 2x16 matrix can be viewed as 4 blocks of 2x4.
    # Or, more naturally for tensor decomposition, we can look for structure.
    
    # Let's try to find a rank-32 decomposition first, as it's a natural candidate
    # (half of schoolbook).
    # If that fails, try rank 24, etc.
    # But random search for exact rank is hard.
    
    # Alternative approach: Use the structure of the indices.
    # a = 2i + j (i in 0..1, j in 0..1)
    # b = 2j + k (j in 0..1, k in 0..15)
    # c = 16k + i (k in 0..15, i in 0..1)
    
    # Let's try a rank-32 decomposition using a specific structure.
    # Consider the tensor as a sum of rank-1 terms.
    # We can try to use a Fourier-like structure on the k dimension.
    
    # For now, let's implement a more efficient random search with ALS.
    # But first, let's check if rank 32 is possible with a simple structured guess.
    
    # Actually, let's just do a robust random search for lower ranks.
    # The key is to be fast.
    
    def try_rank_als(R, n_restarts=5, max_iters=50):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each a, solve A @ U[a,:] = T[a].flatten()
                # A[(b,c), r] = V[b,r] * W[c,r]
                A = (V.T * W.T).T.reshape(mp * pn, R)
                # Actually, A is (mp*pn, R). We want to solve for U[a,:]
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # So for fixed a, T[a].flatten() = A @ U[a,:]
                # A is (mp*pn, R), U[a,:] is (R,)
                # So we solve A @ x = T[a].flatten()
                try:
                    U = np.linalg.lstsq(A, T.reshape(nm, mp * pn).T, rcond=None)[0].T
                except:
                    pass
                
                # Update V
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b, T[:, b, :].flatten() = B @ V[b,:]
                # B[(a,c), r] = U[a,r] * W[c,r]
                B = (U.T * W.T).T.reshape(nm * pn, R)
                try:
                    V = np.linalg.lstsq(B, T[:, :, :].transpose(1, 0, 2).reshape(mp, nm * pn).T, rcond=None)[0].T
                except:
                    pass
                
                # Update W
                # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed c, T[:, :, c].flatten() = C @ W[c,:]
                # C[(a,b), r] = U[a,r] * V[b,r]
                C = (U.T * V.T).T.reshape(nm * mp, R)
                try:
                    W = np.linalg.lstsq(C, T.reshape(nm, mp, pn).transpose(0, 1, 2).reshape(pn, nm * mp).T, rcond=None)[0].T
                except:
                    pass
                
                # Check convergence
                if check_decomposition(U, V, W, tol=1e-6):
                    return U, V, W
            
            if check_decomposition(U, V, W, tol=1e-6):
                return U, V, W
        
        return None

    # Try ranks from 32 upwards (since 32 is a natural candidate)
    # Actually, let's try 32, 33, 34... up to 64
    # But start with 32 as it's the most likely improvement.
    
    best_rank = 64
    best_U = None
    best_V = None
    best_W = None
    
    for R in [32, 33, 34, 35, 36, 40, 48, 64]:
        if time.time() > deadline:
            break
        
        result = try_rank_als(R, n_restarts=3, max_iters=30)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If no lower rank found, use schoolbook
    if best_U is None:
        best_rank = n * m * p
        U = np.zeros((nm, best_rank))
        V = np.zeros((mp, best_rank))
        W = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
