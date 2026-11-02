"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]=1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        recon += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
    return np.allclose(recon, 1.0, atol=1e-9)

def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    n, m, p = DIMENSIONS
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _als_decompose(n, m, p, rank, seed, max_iters=200):
    """
    Attempt to find a rank-`rank` decomposition using ALS.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.default_rng(seed)
    
    # Better initialization: use structured random matrices
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Scale to reasonable magnitude
    U /= np.linalg.norm(U) * 0.5 + 1e-10
    V /= np.linalg.norm(V) * 0.5 + 1e-10
    W /= np.linalg.norm(W) * 0.5 + 1e-10
    
    T = _build_tensor(n, m, p)
    
    prev_res = np.inf
    stall_count = 0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_res = np.inf
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return None
        
        # 1. Update U
        for i in range(n):
            for j in range(m):
                w_idx = [k * n + i for k in range(p)]
                W_slice = W[w_idx, :]  # (p, rank)
                V_slice = V[j*p:(j+1)*p, :]  # (p, rank)
                A = V_slice.T * W_slice.T  # (rank, p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                except:
                    pass
        
        # 2. Update V
        for j in range(m):
            for k in range(p):
                u_idx = [i * m + j for i in range(n)]
                U_slice = U[u_idx, :]  # (n, rank)
                w_idx = [k * n + i for i in range(n)]
                W_slice = W[w_idx, :]  # (n, rank)
                A = U_slice.T * W_slice.T  # (rank, n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                except:
                    pass
        
        # 3. Update W
        for k in range(p):
            for i in range(n):
                u_idx = [i * m + j for j in range(m)]
                U_slice = U[u_idx, :]  # (m, rank)
                v_idx = [j * p + k for j in range(m)]
                V_slice = V[v_idx, :]  # (m, rank)
                A = U_slice.T * V_slice.T  # (rank, m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            
        res = np.linalg.norm(recon - 1.0)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-10:
            return U, V, W
            
        if res < prev_res:
            prev_res = res
            stall_count = 0
        else:
            stall_count += 1
            
        if stall_count > 15:
            # Try different perturbation strategies
            if it % 30 == 0:
                # Small perturbation
                U += rng.standard_normal((n*m, rank)) * 0.05
                V += rng.standard_normal((m*p, rank)) * 0.05
                W += rng.standard_normal((p*n, rank)) * 0.05
            else:
                # Reset to best found so far with perturbation
                U = best_U + rng.standard_normal((n*m, rank)) * 0.1
                V = best_V + rng.standard_normal((m*p, rank)) * 0.1
                W = best_W + rng.standard_normal((p*n, rank)) * 0.1
            stall_count = 0
            
    return None

def _kronecker_decompose(n, m, p, rank, seed):
    """
    Try to exploit Kronecker product structure.
    For m=4=2*2, we can factor the middle dimension.
    """
    if m != 4 or n != 2:
        return None
    
    rng = np.random.default_rng(seed)
    
    # Factor m = m1 * m2 where m1=2, m2=2
    m1, m2 = 2, 2
    
    # We need to find decompositions for smaller problems
    # and combine them using Kronecker products
    
    # Strategy: Try to find rank-r1 decomposition for (n, m1, p) 
    # and rank-r2 decomposition for (n, m2, p), then combine
    # This is heuristic and may not always work, but worth trying
    
    # For now, just use standard ALS but with structured initialization
    # that respects the Kronecker structure
    
    # Initialize with Kronecker-like structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill with random values but with some structure
    for c in range(rank):
        # Each column gets a random pattern
        U[:, c] = rng.standard_normal(n * m)
        V[:, c] = rng.standard_normal(m * p)
        W[:, c] = rng.standard_normal(p * n)
    
    return _als_decompose(n, m, p, rank, seed, max_iters=150)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Lower bound for rank is max(n*m, m*p, p*n) = 28
    # Try ranks from 28 upwards with increasing effort
    
    # Time budget allocation:
    # - Ranks 28-30: 30 trials each (high priority, most likely to succeed if possible)
    # - Ranks 31-35: 15 trials each
    # - Ranks 36-40: 8 trials each
    # - Ranks 41-45: 5 trials each
    # - Ranks 46-55: 3 trials each
    
    rank_trials = {}
    for r in range(28, 31):
        rank_trials[r] = 30
    for r in range(31, 36):
        rank_trials[r] = 15
    for r in range(36, 41):
        rank_trials[r] = 8
    for r in range(41, 46):
        rank_trials[r] = 5
    for r in range(46, 56):
        rank_trials[r] = 3
    
    for rank in range(28, 56):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        num_trials = rank_trials.get(rank, 3)
        
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
                
            seed = rank * 1000 + trial * 7
            
            # Alternate between standard ALS and Kronecker-inspired initialization
            if trial % 2 == 0:
                res = _als_decompose(n, m, p, rank, seed)
            else:
                res = _kronecker_decompose(n, m, p, rank, seed)
            
            if res is not None:
                U, V, W = res
                # Try to round to half-integers
                rounded = _try_round(U.tolist(), V.tolist(), W.tolist())
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # If rounding fails, try harder rounding with different strategies
                # Try rounding to integers first
                U_int = [[int(round(x)) for x in row] for row in U.tolist()]
                V_int = [[int(round(x)) for x in row] for row in V.tolist()]
                W_int = [[int(round(x)) for x in row] for row in W.tolist()]
                if _verify(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int, "V": V_int, "W": W_int}
                
                # Try rounding to half-integers with different tolerances
                for tol in [0.1, 0.15, 0.2]:
                    U_r2 = [[round(x / tol) * tol / 2 for x in row] for row in U.tolist()]
                    V_r2 = [[round(x / tol) * tol / 2 for x in row] for row in V.tolist()]
                    W_r2 = [[round(x / tol) * tol / 2 for x in row] for row in W.tolist()]
                    if _verify(U_r2, V_r2, W_r2, n, m, p):
                        # Ensure all are half-integers
                        if all(abs(x*2 - round(x*2)) < 1e-9 for row in U_r2 for x in row) and \
                           all(abs(x*2 - round(x*2)) < 1e-9 for row in V_r2 for x in row) and \
                           all(abs(x*2 - round(x*2)) < 1e-9 for row in W_r2 for x in row):
                            return {"U": U_r2, "V": V_r2, "W": W_r2}
    
    return best_result
# EVOLVE-BLOCK-END
