import time
import numpy as np
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses Alternating Least Squares (ALS) to find a low-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Precompute indices for the target tensor T
    # T[a, b, d] = 1 if (a%m == b//p) and (a//m == d%n), else 0
    # We need to solve for U, V, W such that sum_c U[a,c]V[b,c]W[d,c] = T[a,b,d]
    
    def build_target_vector(a, V, W, rank):
        """
        For a fixed row index 'a' of U, build the target vector t of length (m*p)*(p*n)
        and the matrix A of shape ((m*p)*(p*n), rank) such that A @ u_a = t.
        Here u_a is U[a, :].
        The equation is sum_c U[a,c] * (V[b,c] * W[d,c]) = T[a,b,d]
        So A[b, d, c] = V[b,c] * W[d,c].
        """
        # Flatten b, d indices
        # b in [0, m*p), d in [0, p*n)
        # We create a matrix of shape (m*p*p*n, rank)
        # But we can be smarter. We only need to enforce constraints where T is 1 or 0.
        # Actually, we must enforce ALL constraints.
        
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        
        # We want to solve for U[a, :]
        # Let X be a matrix where X[idx, c] = V[b, c] * W[d, c]
        # where idx = b * (p*n) + d
        
        # This creates a dense matrix of size 1296 x rank.
        # V and W are fixed.
        
        # Efficient construction:
        # V is (36, R), W is (36, R)
        # We need outer product of each row of V and each row of W?
        # No, it's element-wise product of V[b,:] and W[d,:].
        
        # We can build this using broadcasting if we reshape.
        # But 1296 x 130 is small.
        
        target = np.zeros(m * p * p * n)
        A_mat = np.zeros((m * p * p * n, rank))
        
        # Precompute V and W as numpy arrays
        V_np = np.asarray(V)
        W_np = np.asarray(W)
        
        # Loop over b and d to fill A_mat and target
        # This is 36*36 = 1296 iterations.
        idx = 0
        a_i = a // m
        a_j = a % m
        
        for b in range(m * p):
            b_j = b // p
            for d in range(p * n):
                d_i = d % n
                
                # Determine target value
                if a_j == b_j and a_i == d_i:
                    target[idx] = 1.0
                else:
                    target[idx] = 0.0
                
                # Fill A_mat row
                # A_mat[idx, c] = V[b, c] * W[d, c]
                A_mat[idx, :] = V_np[b, :] * W_np[d, :]
                idx += 1
                
        return A_mat, target

    def solve_als(U, V, W, rank, iterations=20):
        """
        Perform ALS iterations to minimize error.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Update U
            for a in range(n * m):
                A_mat, target = build_target_vector(a, V, W, rank)
                # Solve A_mat @ u_a = target
                # Use least squares
                try:
                    u_a, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, target, rcond=None)
                    U[a, :] = u_a
                except np.linalg.LinAlgError:
                    pass
            
            # Update V
            # For fixed b, solve for V[b, :]
            # Equation: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
            # Rearranged: sum_c (U[a,c] * W[d,c]) * V[b,c] = T[a,b,d]
            # A_mat_V[a, d, c] = U[a,c] * W[d,c]
            # Target is T[a,b,d] for all a,d
            
            U_np = np.array(U)
            W_np = np.array(W)
            
            for b in range(m * p):
                # Build A_mat for V[b]
                # Size: (n*m * p*n, rank)
                # n*m = 16, p*n = 36. Total 576 rows.
                A_mat_v = np.zeros((n * m * p * n, rank))
                target_v = np.zeros(n * m * p * n)
                
                b_j = b // p
                idx = 0
                for a in range(n * m):
                    a_i = a // m
                    a_j = a % m
                    for d in range(p * n):
                        d_i = d % n
                        
                        if a_j == b_j and a_i == d_i:
                            target_v[idx] = 1.0
                        else:
                            target_v[idx] = 0.0
                        
                        A_mat_v[idx, :] = U_np[a, :] * W_np[d, :]
                        idx += 1
                
                try:
                    v_b, residuals, rank_ls, sv = np.linalg.lstsq(A_mat_v, target_v, rcond=None)
                    V[b, :] = v_b
                except np.linalg.LinAlgError:
                    pass
            
            # Update W
            # For fixed d, solve for W[d, :]
            # Equation: sum_c (U[a,c] * V[b,c]) * W[d,c] = T[a,b,d]
            
            U_np = np.array(U)
            V_np = np.array(V)
            
            for d in range(p * n):
                A_mat_w = np.zeros((n * m * m * p, rank))
                target_w = np.zeros(n * m * m * p)
                
                d_i = d % n
                idx = 0
                for a in range(n * m):
                    a_i = a // m
                    a_j = a % m
                    for b in range(m * p):
                        b_j = b // p
                        
                        if a_j == b_j and a_i == d_i:
                            target_w[idx] = 1.0
                        else:
                            target_w[idx] = 0.0
                        
                        A_mat_w[idx, :] = U_np[a, :] * V_np[b, :]
                        idx += 1
                
                try:
                    w_d, residuals, rank_ls, sv = np.linalg.lstsq(A_mat_w, target_w, rcond=None)
                    W[d, :] = w_d
                except np.linalg.LinAlgError:
                    pass
                    
        return U, V, W

    def check_error(U, V, W):
        """Calculate the max absolute error."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        max_err = 0.0
        # Sample some entries or check all?
        # Checking all is 16 * 36 * 36 = 20736 checks.
        # Each check is a dot product of length rank.
        # This is fast enough.
        
        for a in range(n * m):
            a_i = a // m
            a_j = a % m
            for b in range(m * p):
                b_j = b // p
                for d in range(p * n):
                    d_i = d % n
                    
                    if a_j == b_j and a_i == d_i:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    # Compute sum_c U[a,c] V[b,c] W[d,c]
                    val = np.dot(np.dot(U[a, :], V[b, :]), W[d, :])
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
                        
        return max_err

    # Try ranks from 130 down to 100
    # Start with 130 as it's a known good bound for 4x4x4, and 4x4x9 is larger.
    # Actually, 144 is schoolbook.
    # We will try 130, 125, 120, 115, 110.
    
    for rank in [130, 125, 120, 115, 110, 105]:
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + restart)
            # Initialize with small random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Run ALS
            U, V, W = solve_als(U, V, W, rank, iterations=30)
            
            # Check error
            err = check_error(U, V, W)
            
            if err < 1e-6:
                # Success! Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
