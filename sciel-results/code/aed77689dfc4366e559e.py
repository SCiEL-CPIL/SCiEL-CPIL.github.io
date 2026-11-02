import time
import itertools
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _try_rank23_scipy(n, m, p):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    rank = 23
    
    # Indices where T is 1
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
                
    # Flatten U, V, W
    # x = [U_flat, V_flat, W_flat]
    # U is (dim_a, rank), V is (dim_b, rank), W is (dim_c, rank)
    # Size: 3 * rank * dim (since dims are equal 9)
    
    def residuals(x):
        u_flat = x[:rank*dim_a]
        v_flat = x[rank*dim_a : rank*dim_a + rank*dim_b]
        w_flat = x[rank*dim_a + rank*dim_b :]
        
        U = u_flat.reshape(rank, dim_a).T # (dim_a, rank)
        V = v_flat.reshape(rank, dim_b).T
        W = w_flat.reshape(rank, dim_c).T
        
        res = np.zeros(len(indices))
        for idx, (a, b, c) in enumerate(indices):
            # Sum_r U[a,r] V[b,r] W[c,r]
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            res[idx] = val - 1.0
        return res

    # Initial guess: Random
    random.seed(42)
    best_result = None
    best_norm = float('inf')
    
    # Try a few random initializations
    for i in range(10):
        x0 = np.random.randn(rank * (dim_a + dim_b + dim_c))
        # Scale down to avoid overflow
        x0 *= 0.1
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.success or sol.cost < 1e-12:
                # Reconstruct U, V, W
                u_flat = sol.x[:rank*dim_a]
                v_flat = sol.x[rank*dim_a : rank*dim_a + rank*dim_b]
                w_flat = sol.x[rank*dim_a + rank*dim_b :]
                
                U = u_flat.reshape(rank, dim_a).T
                V = v_flat.reshape(rank, dim_b).T
                W = w_flat.reshape(rank, dim_c).T
                
                # Verify full tensor
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    return U, V, W
        except Exception as e:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find rank 23 using scipy
    if time.time() < deadline:
        result = _try_rank23_scipy(n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
