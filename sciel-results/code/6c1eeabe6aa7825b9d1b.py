import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T

def _compute_error_int(U, V, W, T):
    """
    Compute the sum of squared errors using integer arithmetic.
    U, V, W are integer arrays (scaled by 2).
    T is integer array (target values, 1 or 0).
    Note: The reconstruction is sum_r U[:,r] V[:,r] W[:,r].
    If U,V,W are scaled by 2, the result is scaled by 8.
    So we compare (recon/8) vs T.
    Or equivalently: recon_int vs 8*T.
    """
    # U: (N, R), V: (M, R), W: (P, R)
    # recon[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target_scaled = 8 * T
    diff = recon - target_scaled
    return np.sum(diff * diff)

def _local_search_int_fast(U, V, W, T, values, max_iter=5000, timeout=5.0):
    """
    Discrete local search with incremental error updates for speed.
    U, V, W are integer arrays.
    values are the allowed integer values (scaled by 2).
    """
    start_time = time.time()
    current_error = _compute_error_int(U, V, W, T)
    
    n_u, n_v, n_w = U.size, V.size, W.size
    total_vars = n_u + n_v + n_w
    
    shape_U = U.shape
    shape_V = V.shape
    shape_W = W.shape
    
    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
        if current_error == 0:
            break
            
        # Pick a random variable
        var_idx = random.randint(0, total_vars - 1)
        
        if var_idx < n_u:
            mat = U
            idx = var_idx
            shape = shape_U
        elif var_idx < n_u + n_v:
            mat = V
            idx = var_idx - n_u
            shape = shape_V
        else:
            mat = W
            idx = var_idx - n_u - n_v
            shape = shape_W
            
        old_val = mat.flat[idx]
        
        # Try all possible new values
        best_new_val = old_val
        best_new_error = current_error
        
        for new_val in values:
            if new_val == old_val:
                continue
            mat.flat[idx] = new_val
            new_error = _compute_error_int(U, V, W, T)
            if new_error < best_new_error:
                best_new_error = new_error
                best_new_val = new_val
        
        # Apply the best change
        if best_new_val != old_val:
            mat.flat[idx] = best_new_val
            current_error = best_new_error
        else:
            mat.flat[idx] = old_val

    return current_error

def _try_rank(R, n, m, p, T, values, time_limit):
    """
    Try to find a decomposition of rank R using discrete search.
    """
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Number of restarts based on time
    num_restarts = 30
    time_per_restart = time_limit / num_restarts
    
    # Values are half-integers, so scaled by 2 they are integers.
    int_values = [int(v * 2) for v in values]
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Sparse initialization: start with zeros, then add random non-zero entries
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        
        # Add some random non-zero values
        sparsity = 0.3
        for _ in range(int(R * sparsity)):
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            r_idx = random.randint(0, R - 1)
            val = random.choice([v for v in int_values if v != 0])
            U[u_idx, r_idx] += val
            V[v_idx, r_idx] += val
            W[w_idx, r_idx] += val
        
        # Run local search
        err = _local_search_int_fast(U, V, W, T, int_values, max_iter=2000, timeout=time_per_restart)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error == 0:
            break
            
    if best_error == 0:
        return best_U / 2.0, best_V / 2.0, best_W / 2.0
    return None

def _construct_strassen_decomposition():
    """
    Construct a rank-22 decomposition for 2x3 * 3x4 matrix multiplication
    using block structure and Strassen's algorithm.
    """
    n, m, p = 2, 3, 4
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: A1 * B1 (2x1 * 1x4)
    # A1 = A[:, 0], B1 = B[0, :]
    # C1 = A1 * B1
    # C1[i, k] = A[i, 0] * B[0, k]
    # u_idx = i, v_idx = k, w_idx = k*n + i
    # i in 0..1, k in 0..3
    r = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0  # A[i, 0]
            v_idx = 0 * p + k  # B[0, k]
            w_idx = k * n + i  # C[i, k]
            
            U[u_idx, r] = 1.0
            V[v_idx, r] = 1.0
            W[w_idx, r] = 1.0
            r += 1
            
    # Part 2: A2 * B2 (2x2 * 2x4)
    # A2 = A[:, 1:3], B2 = B[1:3, :]
    # Split B2 into B_L (cols 0,1) and B_R (cols 2,3)
    # Use Strassen for A2 * B_L and A2 * B_R
    
    # A2 indices in A:
    # a00 = A[0, 1] -> u=3
    # a01 = A[0, 2] -> u=4
    # a10 = A[1, 1] -> u=5
    # a11 = A[1, 2] -> u=6
    # Wait, m=3. A is 2x3.
    # A[0, 0] -> 0
    # A[0, 1] -> 1
    # A[0, 2] -> 2
    # A[1, 0] -> 3
    # A[1, 1] -> 4
    # A[1, 2] -> 5
    
    # So A2 is:
    # [ A[0, 1], A[0, 2] ]
    # [ A[1, 1], A[1, 2] ]
    # Indices:
    # a = A[0, 1] -> 1
    # b = A[0, 2] -> 2
    # c = A[1, 1] -> 4
    # d = A[1, 2] -> 5
    
    # B2 indices in B:
    # B is 3x4.
    # B[0, 0..3] -> 0..3
    # B[1, 0..3] -> 4..7
    # B[2, 0..3] -> 8..11
    
    # B_L (cols 0, 1):
    # e = B[1, 0] -> 4
    # f = B[1, 1] -> 5
    # g = B[2, 0] -> 8
    # h = B[2, 1] -> 9
    
    # B_R (cols 2, 3):
    # e' = B[1, 2] -> 6
    # f' = B[1, 3] -> 7
    # g' = B[2, 2] -> 10
    # h' = B[2, 3] -> 11
    
    # C indices for B_L (k=0, 1):
    # C[0, 0] -> 0
    # C[0, 1] -> 1
    # C[1, 0] -> 2
    # C[1, 1] -> 3
    
    # C indices for B_R (k=2, 3):
    # C[0, 2] -> 4
    # C[0, 3] -> 5
    # C[1, 2] -> 6
    # C[1, 3] -> 7
    
    # Strassen Products for Left (B_L)
    # P1 = (a+d)(e+h)
    U[1, r] = 1; U[5, r] = 1
    V[4, r] = 1; V[9, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # P2 = (c+d)e
    U[4, r] = 1; U[5, r] = 1
    V[4, r] = 1
    W[2, r] = 1; W[3, r] = -1
    r += 1
    
    # P3 = a(f-h)
    U[1, r] = 1
    V[5, r] = 1; V[9, r] = -1
    W[1, r] = 1; W[3, r] = -1
    r += 1
    
    # P4 = d(g+e)
    U[5, r] = 1
    V[8, r] = 1; V[4, r] = 1
    W[0, r] = -1; W[1, r] = 1
    r += 1
    
    # P5 = (a+b)(h-g)
    U[1, r] = 1; U[2, r] = 1
    V[9, r] = 1; V[8, r] = -1
    W[0, r] = 1; W[2, r] = 1
    r += 1
    
    # P6 = (c-a)(e+f)
    U[4, r] = 1; U[1, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[3, r] = 1
    r += 1
    
    # P7 = (b-d)(g+h)
    U[2, r] = 1; U[5, r] = -1
    V[8, r] = 1; V[9, r] = 1
    W[0, r] = 1
    r += 1
    
    # Strassen Products for Right (B_R)
    # P1' = (a+d)(e'+h')
    U[1, r] = 1; U[5, r] = 1
    V[6, r] = 1; V[11, r] = 1
    W[4, r] = 1; W[7, r] = 1
    r += 1
    
    # P2' = (c+d)e'
    U[4, r] = 1; U[5, r] = 1
    V[6, r] = 1
    W[6, r] = 1; W[7, r] = -1
    r += 1
    
    # P3' = a(f'-h')
    U[1, r] = 1
    V[7, r] = 1; V[11, r] = -1
    W[5, r] = 1; W[7, r] = -1
    r += 1
    
    # P4' = d(g'+e')
    U[5, r] = 1
    V[10, r] = 1; V[6, r] = 1
    W[4, r] = -1; W[5, r] = 1
    r += 1
    
    # P5' = (a+b)(h'-g')
    U[1, r] = 1; U[2, r] = 1
    V[11, r] = 1; V[10, r] = -1
    W[4, r] = 1; W[6, r] = 1
    r += 1
    
    # P6' = (c-a)(e'+f')
    U[4, r] = 1; U[1, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[7, r] = 1
    r += 1
    
    # P7' = (b-d)(g'+h')
    U[2, r] = 1; U[5, r] = -1
    V[10, r] = 1; V[11, r] = 1
    W[4, r] = 1
    r += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try the Strassen-based decomposition
    U_str, V_str, W_str = _construct_strassen_decomposition()
    recon = np.einsum('ur,vr,wr->uvw', U_str, V_str, W_str)
    if np.allclose(recon, T, atol=1e-9):
        best_result = {'U': U_str.tolist(), 'V': V_str.tolist(), 'W': W_str.tolist()}
        best_rank = U_str.shape[1]
        # If we found a valid decomposition, we can return immediately
        # as it is significantly better than schoolbook
        # and searching for lower ranks is unlikely to succeed quickly
        return best_result

    # Define search space
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Try ranks lower than schoolbook
    start_time = time.time()
    deadline = 140.0
    
    for R in range(15, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
            
        ranks_left = rank_schoolbook - R
        time_budget = (deadline - (time.time() - start_time)) / ranks_left
        time_budget = min(time_budget, 20.0)
        
        if time_budget < 1.0:
            break
            
        result = _try_rank(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(recon, T, atol=1e-9):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
                
    return best_result
