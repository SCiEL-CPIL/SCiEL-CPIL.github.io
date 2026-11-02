import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard
    start_time = time.time()
    deadline = 145  # 5s safety margin from 150s limit
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Dimensions for U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # We will try to find a rank-minimal decomposition.
    # The theoretical lower bound is max(n*m, m*p, p*n) = 30, but it's likely higher.
    # Schoolbook is 120.
    # Let's try to reduce the rank.
    
    # Strategy:
    # 1. Try to find a solution with rank < 120 using least_squares.
    # 2. The current program tries rank 119.
    # 3. Let's try a few ranks: 119, 118, ... down to maybe 100 if time permits.
    # 4. We will use the schoolbook solution as initialization for the first R columns.
    
    def try_rank(R):
        # Check time
        if time.time() > deadline:
            return None
        
        num_params = R * (dim_U + dim_V + dim_W)
        
        # Initialize x0
        x0 = np.zeros(num_params)
        
        # Fill with schoolbook terms up to R
        # Schoolbook terms are indexed by (i, j, k)
        # We need to map (i, j, k) to a column index in the schoolbook solution
        # Schoolbook column index = i * m * p + j * p + k
        # Wait, the schoolbook construction in the code is:
        # for i in range(n):
        #   for j in range(m):
        #     for k in range(p):
        #       col += 1
        # So col = i * m * p + j * p + k
        
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if count >= R:
                        break
                    col_idx = i * m * p + j * p + k
                    # In our parameter vector x, the r-th term is at offset r * (dim_U + dim_V + dim_W)
                    # The U part is at offset 0, V at dim_U, W at dim_U + dim_V
                    # U index: i * m + j
                    # V index: j * p + k
                    # W index: k * n + i
                    
                    # We are constructing the r-th term in the decomposition.
                    # Let r = count
                    r = count
                    offset = r * (dim_U + dim_V + dim_W)
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    x0[offset + u_idx] = 1.0
                    x0[offset + dim_U + v_idx] = 1.0
                    x0[offset + dim_U + dim_V + w_idx] = 1.0
                    count += 1
                    if count >= R:
                        break
                if count >= R:
                    break
            if count >= R:
                break
        
        # If we didn't fill enough, the rest are 0 (which is fine, they are free parameters)
        
        def residual(x):
            U_flat = x[0 : R * dim_U]
            V_flat = x[R * dim_U : R * dim_U + R * dim_V]
            W_flat = x[R * dim_U + R * dim_V : ]
            
            U = U_flat.reshape(R, dim_U)
            V = V_flat.reshape(R, dim_V)
            W = W_flat.reshape(R, dim_W)
            
            U_3d = U.T.reshape(n, m, R)
            V_3d = V.T.reshape(m, p, R)
            W_3d = W.T.reshape(p, n, R)
            
            tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            res = tensor_sum - T
            return res.flatten()
        
        try:
            # Use 'lm' method for speed, but it can be unstable. 'trf' is more robust.
            # Given the small size, 'lm' might be okay if we initialize well.
            res = least_squares(residual, x0, method='lm', max_nfev=5000)
            
            if res.success or res.cost < 1e-12:
                # Extract solution
                U_flat = res.x[0 : R * dim_U]
                V_flat = res.x[R * dim_U : R * dim_U + R * dim_V]
                W_flat = res.x[R * dim_U + R * dim_V : ]
                
                U = U_flat.reshape(dim_U, R)
                V = V_flat.reshape(dim_V, R)
                W = W_flat.reshape(dim_W, R)
                
                # Verify
                U_3d = U.T.reshape(n, m, R)
                V_3d = V.T.reshape(m, p, R)
                W_3d = W.T.reshape(p, n, R)
                tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                error = np.linalg.norm(tensor_sum - T)
                
                if error < 1e-9:
                    # Try to round to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    U_3d_h = U_half.T.reshape(n, m, R)
                    V_3d_h = V_half.T.reshape(m, p, R)
                    W_3d_h = W_half.T.reshape(p, n, R)
                    tensor_sum_h = np.einsum('ijr,jkr,kir->ijk', U_3d_h, V_3d_h, W_3d_h)
                    error_h = np.linalg.norm(tensor_sum_h - T)
                    
                    if error_h < 1e-9:
                        return U_half.tolist(), V_half.tolist(), W_half.tolist(), R
                    else:
                        # Return floats if they work? The problem says "integer/half-integer entries".
                        # If we can't round, we might return the float solution if it's valid?
                        # "exactly decompose" implies the values must be exact.
                        # If the float solution is exact (error < 1e-9), it might be due to floating point noise.
                        # But if it doesn't round to half-integers, it might not be "integer/half-integer".
                        # However, if the error is extremely small, maybe we can return the floats?
                        # The prompt says "with integer/half-integer entries".
                        # So we should strictly enforce this.
                        # If we can't round, we discard this solution.
                        pass
                        
        except Exception:
            pass
            
        return None

    # Try ranks from 119 downwards
    # We can't try too many due to time.
    # Let's try 119, 118, 117, 116, 115.
    
    for R in range(rank_school - 1, rank_school - 6, -1):
        if time.time() > deadline:
            break
            
        result = try_rank(R)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            # If we found a valid lower rank, we can stop or try to go lower?
            # The task asks for "rank-minimal".
            # So we should keep trying to go lower.
            # But if we find 119, we should try 118, etc.
            # The loop does this naturally.
            pass

    # If we didn't find anything better, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
