"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    def to_half_integer(x):
        """Round to nearest 0.5"""
        return np.round(2.0 * x) / 2.0
        
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
        
    def verify_strict(U, V, W, n, m, p):
        """Check if decomposition is exact with strict integer/half-integer check"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return verify(U, V, W, n, m, p, tol=1e-12)
        
    def try_find_block_decomposition(rank_block, time_limit, seed):
        """
        Try to find a rank_block decomposition for 2x3 * 3x3 -> 2x3.
        Returns (U_block, V_block, W_block) or None.
        U_block: (6, rank_block)
        V_block: (9, rank_block)  # 3x3 block of V
        W_block: (6, rank_block)  # 3x3 block of W
        """
        n_b, m_b, p_b = 2, 3, 3
        start = time.time()
        np.random.seed(seed)
        
        U = np.random.randn(n_b * m_b, rank_block) * 0.5
        V = np.random.randn(m_b * p_b, rank_block) * 0.5
        W = np.random.randn(p_b * n_b, rank_block) * 0.5
        
        def als_iteration(U, V, W, r):
            # Update U
            for i in range(n_b):
                for j in range(m_b):
                    A = np.zeros((p_b, r))
                    for k in range(p_b):
                        A[k, :] = V[j * p_b + k, :] * W[k * n_b + i, :]
                    try:
                        U[i * m_b + j, :] = np.linalg.lstsq(A, np.ones(p_b), rcond=1e-12)[0]
                    except:
                        pass
            # Update V
            for j in range(m_b):
                for k in range(p_b):
                    A = np.zeros((n_b, r))
                    for i in range(n_b):
                        A[i, :] = U[i * m_b + j, :] * W[k * n_b + i, :]
                    try:
                        V[j * p_b + k, :] = np.linalg.lstsq(A, np.ones(n_b), rcond=1e-12)[0]
                    except:
                        pass
            # Update W
            for k in range(p_b):
                for i in range(n_b):
                    A = np.zeros((m_b, r))
                    for j in range(m_b):
                        A[j, :] = U[i * m_b + j, :] * V[j * p_b + k, :]
                    try:
                        W[k * n_b + i, :] = np.linalg.lstsq(A, np.ones(m_b), rcond=1e-12)[0]
                    except:
                        pass
            return U, V, W
            
        for iteration in range(100):
            if time.time() - start > time_limit:
                break
            U, V, W = als_iteration(U, V, W, rank_block)
            
            # Check convergence
            if iteration > 5 and iteration % 10 == 0:
                residual = 0.0
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            s = np.sum(U[i * m_b + j, :] * V[j * p_b + k, :] * W[k * n_b + i, :])
                            residual += (s - 1.0) ** 2
                if residual < 1e-10:
                    break
                    
        # Snap to half-integers
        U_h = to_half_integer(U)
        V_h = to_half_integer(V)
        W_h = to_half_integer(W)
        
        # Verify
        if verify_strict(U_h, V_h, W_h, n_b, m_b, p_b):
            return U_h, V_h, W_h
            
        # Try a few more iterations if close
        if verify(U, V, W, n_b, m_b, p_b, tol=1e-8):
            U_close = to_half_integer(U)
            V_close = to_half_integer(V)
            W_close = to_half_integer(W)
            if verify_strict(U_close, V_close, W_close, n_b, m_b, p_b):
                return U_close, V_close, W_close
                
        return None

    best_result = None
    
    # Try to find a lower rank block decomposition
    # Schoolbook block rank is 18. Try 17, 16, 15.
    for r_block in [17, 16, 15]:
        if time.time() > deadline - 30:
            break
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            remaining = deadline - time.time()
            time_per_try = min(5.0, remaining / (5 - seed + 1))
            result = try_find_block_decomposition(r_block, time_per_try, seed)
            if result is not None:
                U_b, V_b, W_b = result
                # Construct full decomposition
                r_total = 3 * r_block
                U = np.zeros((n * m, r_total))
                V = np.zeros((m * p, r_total))
                W = np.zeros((p * n, r_total))
                
                for b in range(3):
                    col_start = b * r_block
                    col_end = (b + 1) * r_block
                    U[:, col_start:col_end] = U_b
                    # V_b is 9 x r_block (3x3). Full V is 27 x r_total (3x9).
                    # V[j, 3b+k, col] = V_b[j, k, col-b*r_block]
                    for j in range(m):
                        for k in range(3):
                            V[j * p + b * 3 + k, col_start:col_end] = V_b[j * 3 + k, :]
                    # W_b is 6 x r_block (3x2). Full W is 18 x r_total (9x2).
                    # W[3b+k, i, col] = W_b[k, i, col-b*r_block]
                    for k in range(3):
                        for i in range(n):
                            W[(b * 3 + k) * n + i, col_start:col_end] = W_b[k * n + i, :]
                
                if verify_strict(U, V, W, n, m, p):
                    best_result = (r_total, U, V, W)
                    break
        if best_result is not None:
            break
            
    if best_result is not None:
        r, U_h, V_h, W_h = best_result
        return {
            "U": U_h.tolist(),
            "V": V_h.tolist(),
            "W": W_h.tolist()
        }
        
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
