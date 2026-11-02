import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # New strategy: Tensor network / block structure
    # We assume a block structure where the 3x3 middle matrix is factored
    # into smaller blocks, or we use a structured decomposition based on
    # the fact that 3 = 1 + 2.
    
    # Let's try a structured search for rank < 30.
    # Known lower bound is 15 (from max(nm, mp, pn)).
    # Strassen-like for 2x3x5?
    
    # Let's try to find a decomposition with rank 29 or lower.
    # We will use a smarter search: fix U and V to have a specific structure.
    
    # Structure:
    # U is 6xR, V is 15xR, W is 10xR.
    # We can try to make U and V sparse with small integers.
    
    # A different approach: Use the fact that the tensor is 2x3x5.
    # We can view it as a 2x3 matrix of 3x5 matrices? No.
    
    # Let's try a targeted search for rank 29, 28, etc.
    # We will use a randomized local search starting from a near-solution.
    
    def try_structured_rank(target_rank):
        # We will try to find a solution where U, V, W are sparse integers.
        # We initialize with a random sparse matrix and try to optimize.
        
        # Since exact verification is strict, we need to solve the system exactly.
        # For rank R, we have 30 equations and R * 10 unknowns (if W is free).
        # But U and V are also unknowns.
        
        # Let's fix U and V to be random sparse integer matrices and solve for W.
        # If W is integer/half-integer, we are done.
        
        # To improve odds, we can try different sparsity levels and value sets.
        
        best_solution = None
        
        # Use a fixed number of attempts to stay within time
        max_attempts = 5000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            
            attempts += 1
            
            # Generate random U and V
            # We use a "block" structure idea:
            # Partition the 3 middle columns into blocks of size 1 and 2?
            # Or just random sparse.
            
            # Let's try a "greedy" construction of U and V.
            # Initialize U, V as zeros.
            U = np.zeros((6, target_rank), dtype=int)
            V = np.zeros((15, target_rank), dtype=int)
            
            # We want U[i*m+j] and V[j*p+k] to overlap in a way that W can be solved.
            
            # Strategy: Pick random non-zero patterns for U and V.
            # Values in {-1, 1}.
            
            # Density
            density_u = 0.4
            density_v = 0.4
            
            mask_u = np.random.rand(6, target_rank) < density_u
            mask_v = np.random.rand(15, target_rank) < density_v
            
            U[mask_u] = np.random.choice([-1, 1], size=(6, target_rank))[mask_u]
            V[mask_v] = np.random.choice([-1, 1], size=(15, target_rank))[mask_v]
            
            # Solve for W
            # System: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # This is linear in W.
            
            A = np.zeros((30, target_rank * 10))
            b = np.ones(30)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :].astype(float)
                        v_row = V[j*p+k, :].astype(float)
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * target_rank
                        A[idx, start_idx:start_idx+target_rank] = coeffs
                        idx += 1
            
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(10, target_rank)
            
            # Check if W_sol is close to integers or half-integers
            # We want entries in {0, +/- 0.5, +/- 1, ...}
            
            # Check integers first
            W_int_check = W_sol
            if np.allclose(W_int_check, np.round(W_int_check), atol=1e-4):
                W_list = np.round(W_int_check).tolist()
                U_list = U.tolist()
                V_list = V.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Check half-integers
            W_half_check = W_sol * 2
            if np.allclose(W_half_check, np.round(W_half_check), atol=1e-4):
                W_list = (np.round(W_half_check) / 2).tolist()
                U_list = U.tolist()
                V_list = V.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Optionally, try to "clean" W by rounding and re-verifying?
            # If it's close, rounding might work.
            W_rounded = np.round(W_sol * 2) / 2
            if np.allclose(W_rounded, W_sol, atol=1e-3):
                W_list = W_rounded.tolist()
                U_list = U.tolist()
                V_list = V.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 29 down to 15?
    # Schoolbook is 30.
    # We want to beat 30.
    
    for r in range(29, 14, -1):
        if time.time() > deadline:
            break
        result = try_structured_rank(r)
        if result:
            return result
        
    return schoolbook()
