import numpy as np
from scipy.optimize import minimize
import time

def _solve_tensor_decomposition(n, m, p, rank, max_iters=1000, tol=1e-6, seed=42):
    """
    Finds a rank-decomposition for n x m by m x p matrix multiplication.
    Returns U, V, W or None if not found.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    def objective(params):
        # Unpack
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        # Compute tensor T
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We only need to check the specific entries corresponding to the multiplication
        # Condition: sum_r U[i*m+j', r] V[j''*p+k, r] W[k*n+i, r] == delta_{j', j''}
        
        error = 0.0
        # To speed up, we can compute the full tensor and check, or just the required entries.
        # Full tensor is n*m x m*p x p*n. For 3x3x3, 9x9x9 = 729.
        # For 3x3x4, 9x12x12 = 1296.
        # This is small enough to compute fully.
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_flat[:, r], V_flat[:, r], W_flat[:, r])
            
        # Check constraints
        # For each i in [0, n), k in [0, p), j' in [0, m), j'' in [0, m):
        # Index u = i*m + j'
        # Index v = j''*p + k
        # Index w = k*n + i
        # T[u, v, w] should be 1 if j' == j'', else 0.
        
        for i in range(n):
            for k in range(p):
                for jp in range(m):
                    for jq in range(m):
                        u = i * m + jp
                        v = jq * p + k
                        w = k * n + i
                        target = 1.0 if jp == jq else 0.0
                        diff = T[u, v, w] - target
                        error += diff * diff
                        
        return error

    # Initial params
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Optimization
    result = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': tol})
    
    if result.fun < 1e-4: # Tolerance for success
        U_opt = result.x[:n*m*rank].reshape(n*m, rank)
        V_opt = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        return U_opt, V_opt, W_opt
    else:
        return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the decomposition is exact."""
    T = np.zeros((n*m, m*p, p*n))
    rank = U.shape[1]
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    for i in range(n):
        for k in range(p):
            for jp in range(m):
                for jq in range(m):
                    u = i * m + jp
                    v = jq * p + k
                    w = k * n + i
                    target = 1.0 if jp == jq else 0.0
                    if abs(T[u, v, w] - target) > 1e-5:
                        return False
    return True

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to solve 3x3x3 (rank 23)
    if time.time() < deadline - 30:
        res_333 = _solve_tensor_decomposition(3, 3, 3, 23, max_iters=500, seed=1)
        if res_333 and _verify_decomposition(*res_333, 3, 3, 3):
            U1, V1, W1 = res_333
        else:
            U1, V1, W1 = None, None, None
    else:
        U1, V1, W1 = None, None, None
        
    # Try to solve 3x3x4 (rank 17)
    if time.time() < deadline - 10:
        res_334 = _solve_tensor_decomposition(3, 3, 4, 17, max_iters=500, seed=2)
        if res_334 and _verify_decomposition(*res_334, 3, 3, 4):
            U2, V2, W2 = res_334
        else:
            U2, V2, W2 = None, None, None
    else:
        U2, V2, W2 = None, None, None
        
    # Combine
    if U1 is not None and U2 is not None:
        # Combine U, V, W
        # U1 is 9x23, U2 is 9x17
        # We need to map the indices correctly.
        # U1 corresponds to B columns 0..2
        # U2 corresponds to B columns 3..6
        
        U_full = np.zeros((9, 40))
        V_full = np.zeros((21, 40))
        W_full = np.zeros((21, 40))
        
        # Copy U1
        U_full[:, 0:23] = U1
        
        # Copy V1
        # V1 is for 3x3 B part. V1 shape (9, 23).
        # V_full shape (21, 40).
        # Rows 0..8 of V_full correspond to B rows 0..2, cols 0..2.
        V_full[0:9, 0:23] = V1
        
        # Copy W1
        # W1 is for 3x3 C part. W1 shape (9, 23).
        # W_full shape (21, 40).
        # W index is k*n + i. k is col of C.
        # For 3x3 part, k is 0..2.
        # So W_full[0:9, 0:23] = W1?
        # W1[i, r] corresponds to C[i, k] where k is col index in 0..2.
        # In W_full, index is k*3 + i.
        # So for k=0, indices 0,1,2. k=1, indices 3,4,5. k=2, indices 6,7,8.
        # W1 is flattened how?
        # In _solve_tensor_decomposition, W is (p*n, rank).
        # For 3x3x3, p=3, n=3. W is 9x23.
        # W[k*3+i, r].
        # So W_full[0:9, 0:23] = W1 is correct for the first 3 columns of C.
        
        W_full[0:9, 0:23] = W1
        
        # Copy U2
        U_full[:, 23:40] = U2
        
        # Copy V2
        # V2 is for 3x4 B part. V2 shape (12, 17).
        # V2[j*4 + k, r] where j is row of B (0..2), k is col of B in local 0..3 (global 3..6).
        # In V_full, index is j*7 + k_global.
        # j=0, k_global=3..6 -> indices 3,4,5,6
        # j=1, k_global=3..6 -> indices 10,11,12,13
        # j=2, k_global=3..6 -> indices 17,18,19,20
        
        for j in range(3):
            v2_row_offset = j * 4
            v_full_row_offset = j * 7 + 3
            V_full[v_full_row_offset : v_full_row_offset + 4, 23:40] = V2[v2_row_offset : v2_row_offset + 4, :]
            
        # Copy W2
        # W2 is for 3x4 C part. W2 shape (12, 17).
        # W2[k_local*3 + i, r] where k_local is col of C in local 0..3 (global 3..6).
        # In W_full, index is k_global*3 + i.
        # k_global=3, i=0..2 -> indices 9,10,11
        # k_global=4, i=0..2 -> indices 12,13,14
        # k_global=5, i=0..2 -> indices 15,16,17
        # k_global=6, i=0..2 -> indices 18,19,20
        
        for k_local in range(4):
            w2_row_offset = k_local * 3
            w_full_row_offset = (3 + k_local) * 3
            W_full[w_full_row_offset : w_full_row_offset + 3, 23:40] = W2[w2_row_offset : w2_row_offset + 3, :]
            
        # Verify combined
        if _verify_decomposition(U_full, V_full, W_full, 3, 3, 7):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    return best_result
