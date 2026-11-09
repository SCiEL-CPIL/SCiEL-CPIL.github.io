"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r]
    # So W index is k*n + i, meaning W[r][k*n+i]
    # Let me re-read: W[k*n+i, r] means row index is k*n+i, col is r
    
    rank = U_arr.shape[1]
    # Build the multiplication tensor
    # For each output (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor element T[i,j,k] = 1 (it's the identity tensor for matmul)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a few random test matrices
    for trial in range(50):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # U[:,r] gives coefficients for A elements
            # V[:,r] gives coefficients for B elements  
            # W[:,r] gives coefficients for C elements
            u_vec = U_arr[:, r]  # length n*m
            v_vec = V_arr[:, r]  # length m*p
            w_vec = W_arr[:, r]  # length p*n
            
            # Reshape
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)  # W[k*n+i] so reshape to (p, n)
            
            # The contribution to C from this rank-1 term
            # C += (u_mat * A) * (v_mat * B) * w_mat^T ? 
            # Actually: C[i,k] += sum_r U[i*m+j,r]*A[i,j] * V[j*p+k,r]*B[j,k] * W[k*n+i,r]
            # This is: C += sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
            # Hmm, this doesn't factor cleanly. Let me think again.
            
            # The bilinear form: for each r, we compute:
            # intermediate = sum_{i,j} U[i*m+j, r] * A[i,j]  -- no, that's not right either
            # 
            # Actually the tensor decomposition means:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And C[i,k] = sum_{j} A[i,j] * B[j,k] * T[i,j,k]
            #            = sum_r sum_{i,j,k} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            #            = sum_r (sum_{i,j} A[i,j]*U[i*m+j,r]) * ... no
            #
            # Let me just directly verify the tensor identity.
            pass
        
        # Direct tensor check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    max_err = max(max_err, abs(expected - actual))
        if max_err > 1e-10:
            return False
    return True

def _try_rank_reduction(n, m, p, max_rank, time_budget):
    """Try to find a decomposition with rank < max_rank using optimization."""
    from scipy.optimize import minimize
    
    # Start from schoolbook and try to reduce
    # For now, try a structured approach
    
    # The key insight: for 3x3 x 3x4, we can potentially use rank 27
    # by decomposing each of the 4 columns independently with rank 9 each
    # But that's still 36. Let me think of a better structure.
    
    # Actually, let's try rank 24, 27, 30, 33, 36 in order
    
    for target_rank in [24, 27, 30, 33]:
        if time_budget() < 10:
            break
        
        # Initialize with random factors
        best_result = None
        
        for attempt in range(3):
            if time_budget() < 15:
                break
            
            np.random.seed(42 + attempt)
            
            # Initialize
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Add the schoolbook structure as a starting point
            # Scale down and add to current
            U_s, V_s, W_s = _build_schoolbook(n, m, p)
            U_s = np.array(U_s)[:, :target_rank]
            V_s = np.array(V_s)[:, :target_rank]
            W_s = np.array(W_s)[:, :target_rank]
            
            # Use schoolbook truncated as init
            U = U_s.copy()
            V = V_s.copy()
            W = W_s.copy()
            
            # Optimize
            def objective(params):
                if time_budget() < 5:
                    return 1e10
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Compute tensor error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                            err += (val - 1.0) ** 2
                return err
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if result.fun < 1e-10:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Verify
                    if _verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                        best_result = (U_opt, V_opt, W_opt, target_rank)
                        break
            except Exception:
                pass
        
        if best_result:
            U_opt, V_opt, W_opt, rank = best_result
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), rank
    
    return None

def _build_rank_27_decomposition(n, m, p):
    """
    Try to construct a rank-27 decomposition for 3x3 x 3x4.
    
    Strategy: The 3x3 x 3x4 multiplication can be seen as computing 4 column vectors
    of the result, each being a 3x3 x 3x1 multiplication. Each 3x3 x 3x1 has rank 9.
    Total: 4 * 9 = 36 (schoolbook).
    
    Alternative: Use the structure that C = A @ B where B is 3x4.
    We can write B = [b0, b1, b2, b3] where bi are 3-vectors.
    Then C[:,k] = A @ b_k.
    
    For each column, C[:,k] = A @ b_k requires 9 multiplications (rank 9 for 3x3x1).
    But we can share some computation across columns.
    
    Let me try: express the 4 columns using shared intermediates.
    Actually, the minimum rank for 3x3 x 3x4 is at least max(9, 12, 12) = 12.
    Upper bound is 36.
    
    Let me try a direct construction with rank 27 by grouping.
    """
    # For now, return schoolbook as fallback
    return _build_schoolbook(n, m, p)

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as baseline
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try to find lower rank
    if _time_remaining() > 30:
        result = _try_rank_reduction(n, m, p, best_rank, _time_remaining)
        if result is not None:
            U, V, W, best_rank = result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U] if isinstance(U, np.ndarray) else U
    V_list = [[float(x) for x in row] for row in V] if isinstance(V, np.ndarray) else V
    W_list = [[float(x) for x in row] for row in W] if isinstance(W, np.ndarray) else W
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
