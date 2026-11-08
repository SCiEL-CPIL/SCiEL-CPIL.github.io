"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Schoolbook decomposition as baseline (rank = n*m*p = 144)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U,V,W satisfy the bilinear constraints."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.sum(W[k*n+i] * U[i*m+j1] * V[j2*p+k])
                        target = 1.0 if j1 == j2 else 0.0
                        max_err = max(max_err, abs(s - target))
        return max_err < tol
    
    def check_exact(U, V, W):
        """Check if the decomposition is exact with integer/half-integer entries."""
        # First check if all entries are integer or half-integer
        for mat in [U, V, W]:
            for val in mat.flat:
                if val != 0 and val != int(val) and abs(val*2 - round(val*2)) > 1e-9:
                    return False
        # Then check exactness
        return check_decomposition(U, V, W, tol=1e-9)
    
    def round_to_half(U, V, W):
        """Round entries to nearest half-integers and check validity."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if check_decomposition(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def try_optimize(rank, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        best_err = float('inf')
        
        for outer_iter in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Optimize U (fix V, W)
            for i in range(n):
                for j1 in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for j2 in range(m):
                            a_vec = V[j2*p+k] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            u_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i*m+j1] = u_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
            
            if time.time() > deadline - 15:
                break
            
            # Optimize V (fix U, W)
            for j2 in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for j1 in range(m):
                            a_vec = U[i*m+j1] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            v_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[j2*p+k] = v_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            
            if time.time() > deadline - 15:
                break
            
            # Optimize W (fix U, V)
            for i in range(n):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for j1 in range(m):
                        for j2 in range(m):
                            a_vec = U[i*m+j1] * V[j2*p+k]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    
                    if len(A_rows) > 0:
                        A_mat = np.array(A_rows)
                        b_vec = np.array(b_vals)
                        try:
                            w_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[k*n+i] = w_new
                        except:
                            pass
                    
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            
            # Check error
            err = np.max([abs(np.sum(W[k*n+i] * U[i*m+j1] * V[j2*p+k]) - (1.0 if j1==j2 else 0.0)) 
                         for i in range(n) for k in range(p) for j1 in range(m) for j2 in range(m)])
            
            if err < best_err:
                best_err = err
            
            if err < 1e-6:
                # Found a valid decomposition, try to round
                U_r, V_r, W_r, valid = round_to_half(U, V, W)
                if valid and rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    return True, U_r, V_r, W_r
                elif rank < best_rank:
                    # Even if rounding fails, keep track
                    pass
            
            if time.time() > deadline - 15:
                break
        
        return False, U, V, W
    
    # Try ranks from low to high
    # Lower bound: at least max(n*p, m*p, n*m) = max(36, 36, 16) = 36
    # But realistically, we expect higher. Let's try a range.
    
    # Try a few different ranks with multiple seeds
    ranks_to_try = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            break
        
        # Try a few seeds for each rank
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            
            success, U, V, W = try_optimize(rank, max_iters=50, seed=seed)
            
            if success:
                break
        
        if time.time() > deadline - 15:
            break
    
    # If we found a better result, return it
    if best_rank < rank_schoolbook:
        return best_result
    
    # Otherwise, return the schoolbook decomposition
    return best_result
# EVOLVE-BLOCK-END
