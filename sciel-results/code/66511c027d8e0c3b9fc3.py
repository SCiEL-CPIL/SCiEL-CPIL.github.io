import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each r, outer product of u_r (nm), v_r (mp), w_r (pn)
    # T[i,j,k] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We can check by: for each r, form outer and accumulate
    # But faster: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape: for fixed i,j,k, this is a scalar.
    # Alternative check: 
    # C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Our decomposition: C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r])
    # So define for each r: M_r[i,k] = W[k*n+i,r] * (sum_j U[i*m+j,r]*V[j*p+k,r])
    # Then C = sum_r M_r
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    r = U.shape[1]
    for t in range(r):
        u = U[:, t].reshape(n, m)
        v = V[:, t].reshape(m, p)
        w = W[:, t].reshape(p, n)
        # M_t[i,k] = w[k,i] * sum_j u[i,j]*v[j,k]
        # sum_j u[i,j]*v[j,k] is (u @ v)[i,k]
        uv = u @ v
        # M_t[i,k] = w[k,i] * uv[i,k]
        M_t = uv * w.T  # w is (p,n), w.T is (n,p), elementwise multiply
        C_decomp += M_t
    
    return np.allclose(C_true, C_decomp, atol=1e-10)

def _random_candidate(n, m, p, rank, seed):
    """Generate a random sparse candidate."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t in range(rank):
        # Each factor has sparse structure
        # U row: pick a few non-zero entries
        nnz_u = rng.randint(1, 4)
        rows_u = rng.choice(n * m, nnz_u, replace=False)
        vals_u = rng.choice([-1, 1], nnz_u)
        for row, val in zip(rows_u, vals_u):
            U[row, t] = val
        
        nnz_v = rng.randint(1, 4)
        rows_v = rng.choice(m * p, nnz_v, replace=False)
        vals_v = rng.choice([-1, 1], nnz_v)
        for row, val in zip(rows_v, vals_v):
            V[row, t] = val
        
        nnz_w = rng.randint(1, 4)
        rows_w = rng.choice(p * n, nnz_w, replace=False)
        vals_w = rng.choice([-1, 1], nnz_w)
        for row, val in zip(rows_w, vals_w):
            W[row, t] = val
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute residual norm for verification."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    r = U.shape[1]
    for t in range(r):
        u = U[:, t].reshape(n, m)
        v = V[:, t].reshape(m, p)
        w = W[:, t].reshape(p, n)
        uv = u @ v
        M_t = uv * w.T
        C_decomp += M_t
    
    return np.linalg.norm(C_true - C_decomp)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_fb = np.zeros((n * m, schoolbook_rank))
    V_fb = np.zeros((m * p, schoolbook_rank))
    W_fb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank via random search
    # Known lower bounds: rank >= max(n*m, m*p, p*n) / something
    # For 4x4x7: schoolbook is 112. Strassen-like could be lower.
    # Try ranks from a reasonable lower bound up to schoolbook-1
    
    # Lower bound: each output entry C[i,k] depends on all m inputs
    # A trivial lower bound is p*n = 28 (one per output entry if we could do it in 1 rank)
    # But realistically, try from 60 upward
    min_rank_to_try = 50
    
    seed_counter = 0
    for rank in range(min_rank_to_try, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            seed_counter += 1
            U, V, W = _random_candidate(n, m, p, rank, seed_counter)
            
            # Check residual
            residual = _compute_residual(U, V, W, n, m, p)
            
            if residual < 1e-10:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found valid, try lower rank
        
        if best_rank < schoolbook_rank:
            break  # Found improvement, stop
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
