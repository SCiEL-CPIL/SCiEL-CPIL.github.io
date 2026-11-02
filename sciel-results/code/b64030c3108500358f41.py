import time
import numpy as np
from scipy.linalg import lstsq
from itertools import product

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 115.0  # Safety margin

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error_fast(U, V, W, T):
    """
    Compute ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||_F^2
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = reconstructed - T
    return np.sum(diff * diff)

def _kron_init(n, m, p, rank, seed=0):
    """
    Generate initialization based on Kronecker structure or random.
    For rank 50, we can use the exact schoolbook solution.
    For lower ranks, we use random or projected schoolbook.
    """
    rng = np.random.RandomState(seed)
    
    if rank == n * m * p:
        # Exact schoolbook solution
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    else:
        # Random initialization for lower ranks
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize
        for mat in [U, V, W]:
            norms = np.linalg.norm(mat, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms
        return U, V, W

def _als_step_fast(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration.
    """
    rank = U.shape[1]
    
    # Update U
    # T.reshape(n*m, -1) = U @ (V[:,r] * W[:,r]).reshape(-1)
    # A_u[r, idx] = V[j,r] * W[k,r] where idx corresponds to (j,k) in flattened T
    # T is indexed [i, j, k]. Flattened as [i, j*p + k]?
    # T.shape is (n*m, m*p, p*n).
    # Reshape T to (n*m, m*p*p*n).
    # The index in the second dimension is (j, k) flattened?
    # T[i*m+j, j*p+k, k*n+i].
    # If we fix i, the remaining indices are j and k.
    # The "other" part of the tensor for U update is V[j,r] * W[k,r].
    # We need to align the indices.
    
    # Let's construct A_u explicitly but vectorized.
    # A_u has shape (rank, m*p*p*n)
    # For a given r, A_u[r, :] should be the vector [V[j,r]*W[k,r]] for all j,k
    # ordered such that it matches T.reshape(n*m, -1)
    
    # T.reshape(n*m, m*p*p*n)
    # Index in second dim: idx = (j*p + k) * (p*n) + (k*n + i)? No.
    # T is 3D. Reshape to 2D (dim0, dim1*dim2).
    # dim0 = n*m. Index: i*m + j.
    # dim1 = m*p. Index: j*p + k.
    # dim2 = p*n. Index: k*n + i.
    
    # So T_flat[i*m+j, (j*p+k)*(p*n) + (k*n+i)]
    
    # We want to solve for U.
    # T_flat[i*m+j, :] = sum_r U[i*m+j, r] * A_u[r, :]
    # So A_u[r, (j*p+k)*(p*n) + (k*n+i)] = V[j*p+k, r] * W[k*n+i, r]
    
    # We can build A_u by looping over j, k, i?
    # Or vectorized:
    # Create meshgrid for j, k, i
    js = np.repeat(np.arange(m), p*n) # Not quite, need to match the flattened index structure
    # Let's just loop over the non-zero structure or use broadcasting.
    
    # Efficient construction:
    # A_u is rank x (m*p*p*n)
    # We can fill it by iterating r? No, iterate over the tensor indices.
    
    # Let's use a helper to build the design matrix for one mode.
    # Mode U: variable is U[i*m+j, r].
    # Coefficient for U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r].
    
    # We will build A_u, A_v, A_w.
    
    A_u = np.zeros((rank, m*p*p*n), dtype=np.float64)
    A_v = np.zeros((rank, n*m*p*n), dtype=np.float64)
    A_w = np.zeros((rank, n*m*m*p), dtype=np.float64)
    
    # Pre-compute index mappings
    # For A_u:
    # col_idx = (j*p + k) * (p*n) + (k*n + i)
    # row_idx = r
    # val = V[j*p+k, r] * W[k*n+i, r]
    
    # This loop is O(n*m*p*rank). 2*5*5*50 = 2500 ops. Very fast.
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # A_u
                    idx_u = (j*p + k) * (p*n) + (k*n + i)
                    A_u[r, idx_u] = V[j*p + k, r] * W[k*n + i, r]
                    
                    # A_v
                    # Variable V[j*p+k, r]. Coeff U[i*m+j, r] * W[k*n+i, r]
                    # T_flat_v = T.transpose(1,0,2).reshape(m*p, -1)
                    # Index in T_flat_v: (j*p+k, (i*m+j)*(p*n) + (k*n+i))?
                    # T.transpose(1,0,2) shape (m*p, n*m, p*n)
                    # Reshape: (m*p, n*m*p*n)
                    # Index: (j*p+k, (i*m+j)*(p*n) + (k*n+i))
                    idx_v = (i*m + j) * (p*n) + (k*n + i)
                    A_v[r, idx_v] = U[i*m + j, r] * W[k*n + i, r]
                    
                    # A_w
                    # Variable W[k*n+i, r]. Coeff U[i*m+j, r] * V[j*p+k, r]
                    # T_flat_w = T.transpose(2,0,1).reshape(p*n, -1)
                    # Index: (k*n+i, (i*m+j)*(m*p) + (j*p+k))
                    idx_w = (i*m + j) * (m*p) + (j*p + k)
                    A_w[r, idx_w] = U[i*m + j, r] * V[j*p + k, r]
    
    # Solve
    T_flat_u = T.reshape(n*m, m*p*p*n)
    U_new, _, _, _ = lstsq(A_u, T_flat_u.T, cond=None)
    U = U_new.T
    
    T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    V_new, _, _, _ = lstsq(A_v, T_flat_v.T, cond=None)
    V = V_new.T
    
    T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    W_new, _, _, _ = lstsq(A_w, T_flat_w.T, cond=None)
    W = W_new.T
    
    return U, V, W

def _try_rank_als(n, m, p, rank, T, seed=0, time_limit=5.0):
    start_time = time.time()
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for init_seed in range(seed, seed + 2):
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = _kron_init(n, m, p, rank, init_seed)
        cur_err = _tensor_error_fast(U, V, W, T)
        
        max_iters = 100
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = _als_step_fast(U, V, W, T, n, m, p)
            cur_err = _tensor_error_fast(U, V, W, T)
            
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-12:
                break
    
    if best_U is None:
        return None
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook (rank 50)
    U, V, W = _kron_init(n, m, p, n*m*p, seed=0)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    # Theoretical lower bound is high, but let's try a few steps down.
    # 49, 48, 47, 46, 45
    for rank in range(n*m*p - 1, 44, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        success = False
        for seed in range(2):
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
                
            try:
                U_new, V_new, W_new, err = _try_rank_als(n, m, p, rank, T, seed=seed, time_limit=4.0)
                if err < 1e-10:
                    # Verify
                    if _tensor_error_fast(U_new, V_new, W_new, T) < 1e-10:
                        best_rank = rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        success = True
                        break
            except Exception:
                continue
                
        if success:
            continue
        else:
            # If we fail to reduce rank, and we've already tried a few, break
            # Schoolbook is 50. If 49 fails, 48 likely fails.
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
