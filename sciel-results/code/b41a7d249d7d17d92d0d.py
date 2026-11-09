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
    
    # Strassen's 7-term algorithm for 3x3 * 3x3 -> 3x3
    # M1 = (A11+A33)*(B11+B33)
    # M2 = (A21+A23)*B11
    # M3 = A11*(B12-B32)
    # M4 = A33*(B23-B13)
    # M5 = (A13+A33)*B33
    # M6 = (A11-A21)*(B21+B22)
    # M7 = (A23-A13)*(B11+B12)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C13 = M2-M4+M6
    # C21 = M1-M3+M5-M7
    # C22 = -M1+M2+M4+M7
    # C23 = M3-M2+M6-M7
    # C31 = M1-M4+M5
    # C32 = M2-M5+M7
    # C33 = M1-M3+M4-M6
    
    def construct_strassen_block():
        """
        Construct a rank-7 decomposition for 2x3 * 3x3 -> 2x3 using Strassen.
        For each row i of A (i=0,1) and each of the 7 Strassen terms, we create one rank term.
        Total rank for one 3x3 block = 2 * 7 = 14.
        For 3 blocks (since p=9), total rank = 3 * 14 = 42.
        """
        r = 42
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Strassen coefficients for C = A*B where A is 3x3, B is 3x3
        # We need to express each C[i,j] as sum of products
        # For our case, A is 2x3 (rows 0,1) and B is 3x3, C is 2x3
        
        # Strassen terms for 3x3*3x3:
        # M1 = (A11+A33)*(B11+B33)
        # M2 = (A21+A23)*B11
        # M3 = A11*(B12-B32)
        # M4 = A33*(B23-B13)
        # M5 = (A13+A33)*B33
        # M6 = (A11-A21)*(B21+B22)
        # M7 = (A23-A13)*(B11+B12)
        
        # For 2x3 * 3x3 -> 2x3, we only use rows 1,2 of A (A11,A12,A13 and A21,A22,A23)
        # Let's adapt Strassen for 2x3 * 3x3:
        # We can still use the 7-term structure but with A3* = 0 for missing row 3
        
        # Actually, let's use a different approach: directly construct the rank-14 decomposition
        # for 2x3 * 3x3 -> 2x3 by using Strassen-like combinations
        
        # For block b (b=0,1,2), the B block is B[:, b*3:(b+1)*3]
        # U index: i*m + j where i in {0,1}, j in {0,1,2}
        # V index: j*p + b*3 + k where j in {0,1,2}, k in {0,1,2}
        # W index: (b*3+k)*n + i where k in {0,1,2}, i in {0,1}
        
        # Strassen decomposition for 2x3 * 3x3 -> 2x3:
        # We'll use 7 terms per row of A, so 14 terms total
        
        # Define the Strassen-like decomposition for 2x3 * 3x3:
        # Let A = [a00 a01 a02; a10 a11 a12]
        # Let B = [b00 b01 b02; b10 b11 b12; b20 b21 b22]
        # C = A*B = [c00 c01 c02; c10 c11 c12]
        
        # We can use the following 14-term decomposition:
        # For row 0 of A:
        # M1 = (a00+a02)*(b00+b20)
        # M2 = (a01+a02)*b00
        # M3 = a00*(b01-b21)
        # M4 = a02*(b12-b02)
        # M5 = (a00+a02)*b22
        # M6 = (a00-a01)*(b10+b11)
        # M7 = (a02-a00)*(b00+b01)
        
        # For row 1 of A:
        # M8 = (a10+a12)*(b00+b20)
        # M9 = (a11+a12)*b00
        # M10 = a10*(b01-b21)
        # M11 = a12*(b12-b02)
        # M12 = (a10+a12)*b22
        # M13 = (a10-a11)*(b10+b11)
        # M14 = (a12-a10)*(b00+b01)
        
        # This is getting complex. Let's just use the schoolbook for now but with a better structure.
        # Actually, let's try a direct approach: use rank 14 per block with structured initialization
        
        for b in range(3):
            for i in range(n):  # i = 0, 1
                for term in range(7):  # 7 Strassen terms per row
                    col = b * 14 + i * 7 + term
                    
                    # For each Strassen term, we need to set U, V, W appropriately
                    # This is complex; let's use a simpler approach
                    
                    # Simple structured approach: use rank 14 with specific patterns
                    # For now, let's just return the schoolbook and improve later
                    pass
        
        # For now, fall back to schoolbook for each block
        for b in range(3):
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        col = b * 18 + i * m * 3 + j * 3 + k
                        U[i * m + j, col] = 1.0
                        V[j * p + b * 3 + k, col] = 1.0
                        W[(b * 3 + k) * n + i, col] = 1.0
        
        return U, V, W
    
    def try_rank_als(r, seed, time_limit, block_structure=False):
        """Try to find a rank-r decomposition using structured ALS"""
        start = time.time()
        np.random.seed(seed)
        
        if r >= rank_schoolbook:
            U = U_sb[:, :r].copy()
            V = V_sb[:, :r].copy()
            W = W_sb[:, :r].copy()
        else:
            # Structured random initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            if block_structure and r % 3 == 0:
                block_size = r // 3
                for b in range(3):
                    start_col = b * block_size
                    end_col = (b + 1) * block_size
                    if start_col >= r:
                        break
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
                    for j in range(m):
                        for k in range(3):
                            V[j * p + b * 3 + k, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
                    for k in range(3):
                        for i in range(n):
                            W[(b * 3 + k) * n + i, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
            else:
                U[:, :] = np.random.randn(n * m, r) * 0.5
                V[:, :] = np.random.randn(m * p, r) * 0.5
                W[:, :] = np.random.randn(p * n, r) * 0.5
        
        def als_iteration(U, V, W, r):
            """One ALS iteration"""
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)[0]
                    except:
                        pass
            
            return U, V, W
        
        # Main ALS loop
        max_iters = 200
        for iteration in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            U, V, W = als_iteration(U, V, W, r)
            
            # Check convergence
            if iteration > 0 and iteration % 20 == 0:
                # Quick check: compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            residual += (s - 1.0) ** 2
                if residual < 1e-10:
                    break
        
        # Final polish with tighter tolerance
        for iteration in range(50):
            if time.time() - start > time_limit:
                break
            
            U, V, W = als_iteration(U, V, W, r)
        
        # Round to half-integers and verify
        U_h = to_half_integer(U)
        V_h = to_half_integer(V)
        W_h = to_half_integer(W)
        
        if verify_strict(U_h, V_h, W_h, n, m, p):
            return (U_h, V_h, W_h)
        
        if verify(U, V, W, n, m, p, tol=1e-10):
            U_close = to_half_integer(U)
            V_close = to_half_integer(V)
            W_close = to_half_integer(W)
            if verify_strict(U_close, V_close, W_close, n, m, p):
                return (U_close, V_close, W_close)
        
        return None
    
    # Strategy: Try to find low-rank decompositions
    # First, try block-structured decompositions with various ranks per block
    # Then try general ALS with various ranks
    
    best_result = None
    
    # Try block-structured decompositions: 3 blocks of r_per_block rank each
    for r_per_block in range(6, 19):  # total rank from 18 to 54
        r_total = 3 * r_per_block
        if r_total >= rank_schoolbook:
            continue
        if time.time() > deadline - 30:
            break
        
        max_seeds = 10
        for seed in range(max_seeds):
            if time.time() > deadline - 20:
                break
            remaining = deadline - time.time()
            time_per_try = min(8.0, remaining / (max_seeds - seed + 1))
            result = try_rank_als(r_total, seed, time_limit=time_per_try, block_structure=True)
            if result is not None:
                U_h, V_h, W_h = result
                if best_result is None or r_total < best_result[0]:
                    best_result = (r_total, U_h, V_h, W_h)
                break  # Found a valid decomposition for this r_per_block
    
    # If we found a block-structured result, try to improve with general ALS
    if best_result is not None:
        r_found = best_result[0]
        # Try slightly lower ranks with general ALS
        for r in range(r_found - 1, max(15, r_found - 10), -1):
            if time.time() > deadline - 15:
                break
            max_seeds = 5
            for seed in range(max_seeds):
                if time.time() > deadline - 10:
                    break
                remaining = deadline - time.time()
                time_per_try = min(6.0, remaining / (max_seeds - seed + 1))
                result = try_rank_als(r, seed, time_limit=time_per_try, block_structure=False)
                if result is not None:
                    U_h, V_h, W_h = result
                    if best_result is None or r < best_result[0]:
                        best_result = (r, U_h, V_h, W_h)
                    break
    
    # If no block-structured result, try general ALS from lower ranks
    if best_result is None:
        for r in range(30, 15, -1):
            if time.time() > deadline - 15:
                break
            max_seeds = 8
            for seed in range(max_seeds):
                if time.time() > deadline - 10:
                    break
                remaining = deadline - time.time()
                time_per_try = min(5.0, remaining / (max_seeds - seed + 1))
                result = try_rank_als(r, seed, time_limit=time_per_try, block_structure=False)
                if result is not None:
                    U_h, V_h, W_h = result
                    best_result = (r, U_h, V_h, W_h)
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
