"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 130  # seconds, leave 20s margin for 150s limit

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]=1."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                result[i, j, k] = s
    return np.allclose(result, 1.0, atol=1e-9)

def _try_round(U, V, W):
    """Try to round entries to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    n, m, p = DIMENSIONS
    if _verify(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None

def _construct_rank(n, m, p, rank, seed):
    """
    Attempt to find a rank-`rank` decomposition using ALS with a specific seed.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    T = _build_tensor(n, m, p)
    
    # Pre-compute indices for faster access? No, lists are fine for small n,m,p.
    # But we can vectorize the matrix construction inside the loop.
    
    max_iters = 500
    tol = 1e-10
    
    prev_res = np.inf
    stall_count = 0
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            return None
        
        # 1. Update U
        # For each (i,j), solve min_u || T[i,j,:] - u @ M_u ||^2
        # M_u[k,c] = V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                # Vectorized construction of A (rank x p)
                # V_slice: (p, rank) -> V[j*p:(j+1)*p, :]
                # W_slice: (p, rank) -> W[k*n+i, :] for k in 0..p-1
                # We need A[c,k] = V[j*p+k, c] * W[k*n+i, c]
                # This is element-wise product of V_slice.T (rank x p) and W_slice.T (rank x p)?
                # V_slice is (p, rank). V_slice.T is (rank, p).
                # W rows are k*n+i. Let's extract them.
                # Indices for W: [0*n+i, 1*n+i, ..., (p-1)*n+i]
                w_idx = [k * n + i for k in range(p)]
                W_slice = W[w_idx, :]  # (p, rank)
                # A = V_slice.T * W_slice.T  -> (rank, p)
                A = V[j*p:(j+1)*p, :].T * W_slice.T
                # Solve A @ u = T[i,j,:].T
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                except:
                    pass
        
        # 2. Update V
        for j in range(m):
            for k in range(p):
                # M_v[i,c] = U[i*m+j, c] * W[k*n+i, c]
                # U rows: i*m+j for i in 0..n-1
                u_idx = [i * m + j for i in range(n)]
                U_slice = U[u_idx, :]  # (n, rank)
                # W rows: k*n+i for i in 0..n-1
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
                # M_w[j,c] = U[i*m+j, c] * V[j*p+k, c]
                # U rows: i*m+j for j in 0..m-1
                u_idx = [i * m + j for j in range(m)]
                U_slice = U[u_idx, :]  # (m, rank)
                # V rows: j*p+k for j in 0..m-1
                v_idx = [j * p + k for j in range(m)]
                V_slice = V[v_idx, :]  # (m, rank)
                A = U_slice.T * V_slice.T  # (rank, m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        # Vectorized reconstruction
        # Sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # This is a bit tricky to fully vectorize without large intermediates, but n,m,p are small.
        # Let's stick to the loop for safety and simplicity, it's fast enough.
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            # Outer product of U[:,:,c], V[:,:,c], W[:,:,c] ? No.
            # It's sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # We can do this by iterating c and adding the outer product of the slices?
            # No, U is (n,m), V is (m,p), W is (p,n).
            # The term is U_c[i,j] * V_c[j,k] * W_c[k,i].
            # This is a 3-way product.
            # Let's just do the triple loop for residual check, it's only 56 ops * rank.
            pass
            
        # Faster residual check:
        # res = || T - recon ||
        # Since T is all 1s, we just need to check if recon is close to 1.
        # Let's compute recon efficiently.
        # Actually, let's just compute the sum directly.
        recon = np.zeros((n, m, p))
        for c in range(rank):
            # u_c: (n, m), v_c: (m, p), w_c: (p, n)
            # We want sum_{i,j,k} (u_c[i,j] v_c[j,k] w_c[k,i] - 1)^2 ? No, just the values.
            # Let's use einsum or just loop.
            # np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            recon += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            
        res = np.linalg.norm(recon - 1.0)
        
        if res < tol:
            return U, V, W
            
        if res < prev_res:
            prev_res = res
            stall_count = 0
        else:
            stall_count += 1
            
        if stall_count > 20:
            # Restart or break? Let's try a small perturbation
            U += rng.standard_normal((n*m, rank)) * 0.1
            V += rng.standard_normal((m*p, rank)) * 0.1
            W += rng.standard_normal((p*n, rank)) * 0.1
            stall_count = 0
            
    return None

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
    # Try ranks from 28 upwards
    # We have ~130s.
    # Each ALS run takes maybe 0.1-0.5s?
    # Let's prioritize lower ranks.
    
    # Strategy:
    # 1. Try rank 28 with many seeds.
    # 2. Try rank 29 with many seeds.
    # etc.
    
    # However, if rank 28 is impossible (which it often is for matmul), we waste time.
    # But we must try.
    
    # Let's allocate time per rank.
    # Total time budget: 130s.
    # Let's try ranks 28, 29, 30, 31, 32...
    # If we find one, return immediately.
    
    max_rank_to_try = 40 # Don't go too high, schoolbook is 56.
    
    for rank in range(28, max_rank_to_try + 1):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        # Number of trials per rank
        # Give more trials to lower ranks? Or equal?
        # Let's do 20 trials per rank initially.
        num_trials = 20
        
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
                
            seed = rank * 1000 + trial
            res = _construct_rank(n, m, p, rank, seed)
            if res is not None:
                U, V, W = res
                # Try to round to half-integers
                rounded = _try_round(U.tolist(), V.tolist(), W.tolist())
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # If rounding fails, maybe the solution is valid but not half-integer?
                # The problem requires integer/half-integer entries.
                # If ALS converges to a valid decomposition, it might be close to rational numbers.
                # But we MUST return half-integers.
                # If _try_round fails, we cannot return this solution.
                # We should continue searching.
                pass
    
    return best_result
# EVOLVE-BLOCK-END
