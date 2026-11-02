import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Rank-27 schoolbook (always valid)
    def _schoolbook():
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a candidate decomposition
    def _verify(U, V, W):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Build tensor: T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try rank-26: use Strassen on top-left 2x2 block
    # Partition 3x3 into:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] similarly
    # C = A*B:
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x1) = A11*B12 + A12*B22
    # C21 (1x2) = A21*B11 + A22*B21
    # C22 (1x1) = A21*B12 + A22*B22
    
    # For C11: use Strassen (rank 7) for A11*B11, plus A12*B21 (rank 2)
    # For C12: A11*B12 (rank 4) + A12*B22 (rank 1) = rank 5
    # For C21: A21*B11 (rank 4) + A22*B21 (rank 1) = rank 5
    # For C22: A21*B12 (rank 2) + A22*B22 (rank 1) = rank 3
    # Total: 7 + 2 + 5 + 5 + 3 = 22? No, that's not right because Strassen terms are shared.
    
    # Actually let me think more carefully. The standard approach:
    # Use Strassen's 7 multiplications for the 2x2 block product A11*B11.
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # But in our case, A22 is 1x1, A12 is 2x1, A21 is 1x2, etc.
    # This gets complicated. Let me just try a direct rank-26 construction.
    
    # Alternative: Try to find rank-26 by solving a system.
    # The tensor has 27 entries. We need rank r such that there exist U (9xr), V (9xr), W (9xr)
    # with sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    
    # For rank 26, we have 26*27 = 702 variables (minus gauge freedom).
    # This is a system of 27 polynomial equations in 702 unknowns.
    # We can try to set up a linear system by fixing some structure.
    
    # Let me try a specific rank-26 construction based on the idea that
    # one of the 27 schoolbook terms can be eliminated by combining others.
    
    # Actually, let me just try to numerically solve for rank-26.
    # Fix U and V to have a specific structure, then solve for W.
    
    # Strategy: Use the 26 schoolbook terms (all except one), and see if we can
    # adjust coefficients to make it work.
    
    # Let's try: remove the (i=2,j=2,k=2) term (the last one in schoolbook).
    # Then we need: sum_{r=1}^{26} U[8,r]*V[8,r]*W[8,r] = 1
    # But each term only contributes to one (i,j,k) entry in schoolbook.
    # So if we remove term 27, entry (2,2,2) gets 0, not 1.
    
    # We need to modify the decomposition so that the (2,2,2) entry is still 1.
    # This means some other terms must also contribute to (2,2,2).
    
    # This is getting complex. Let me try a numerical approach.
    
    # Set up: we want U, V, W of shape (9, 26) such that
    # for all (i,j,k): sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    
    # Let's try to use a parameterization where U and V are mostly identity-like.
    
    # Actually, let me just try random initialization and gradient descent.
    
    def _try_rank26():
        if time.time() > deadline - 30:
            return None
        
        np.random.seed(42)
        
        # Initialize U, V, W randomly
        U = np.random.randn(9, 26) * 0.1
        V = np.random.randn(9, 26) * 0.1
        W = np.random.randn(9, 26) * 0.1
        
        # We want to minimize sum_{i,j,k} (sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] - 1)^2
        # This is a non-convex optimization problem.
        
        # Let's use a simple iterative approach: alternate between fixing two matrices
        # and solving for the third.
        
        for iteration in range(200):
            if time.time() > deadline - 30:
                return None
            
            # Fix U and V, solve for W
            # For each (i,j,k), we need: sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = 1
            # This is a linear system in W for each (k,i) pair.
            # For fixed (k,i), the equation is: sum_j sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r] = ...
            # Wait, this isn't quite right. Let me restructure.
            
            # The constraint is: for each (i,j,k), T[i,j,k] = 1
            # T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
            
            # If we fix U and V, then for each (i,k), we have:
            # sum_j sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = ... 
            # No, that's not right either. The constraint is per (i,j,k).
            
            # Let me think of it differently. For fixed U and V, the equation for W is:
            # For each (i,j,k): sum_r (U[i*3+j, r] * V[j*3+k, r]) * W[k*3+i, r] = 1
            # Let A[i,j,k,r] = U[i*3+j, r] * V[j*3+k, r]
            # Then: sum_r A[i,j,k,r] * W[k*3+i, r] = 1
            # This is a linear system: for each (i,j,k), we have one equation.
            # Unknowns are W[k*3+i, r] for all k,i,r.
            # There are 9*26 = 234 unknowns and 27 equations.
            # So it's underdetermined. We can find a minimum-norm solution.
            
            # Build the linear system: M @ w_flat = b
            # where w_flat is W flattened (234 entries), b is ones (27 entries)
            # M has shape (27, 234)
            
            M = np.zeros((27, 234))
            b = np.ones(27)
            
            idx = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for r in range(26):
                            M[idx, k*3+i*26+r] = U[i*3+j, r] * V[j*3+k, r]
                        idx += 1
            
            # Solve M @ w_flat = b in least squares sense
            w_flat, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
            W_new = w_flat.reshape(9, 26)
            
            # Check residual
            residual = np.sum((M @ w_flat - b)**2)
            
            # Similarly, fix V and W, solve for U
            M2 = np.zeros((27, 234))
            idx = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for r in range(26):
                            M2[idx, i*3+j*26+r] = V[j*3+k, r] * W[k*3+i, r]
                        idx += 1
            u_flat, residuals2, rank2, sv2 = np.linalg.lstsq(M2, b, rcond=None)
            U_new = u_flat.reshape(9, 26)
            
            # Fix U_new and W, solve for V
            M3 = np.zeros((27, 234))
            idx = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for r in range(26):
                            M3[idx, j*3+k*26+r] = U_new[i*3+j, r] * W[k*3+i, r]
                        idx += 1
            v_flat, residuals3, rank3, sv3 = np.linalg.lstsq(M3, b, rcond=None)
            V_new = v_flat.reshape(9, 26)
            
            U, V, W = U_new, V_new, W_new
            
            # Check if we've converged
            if residual < 1e-12 and residuals2 < 1e-12 and residuals3 < 1e-12:
                # Verify
                Ua = U
                Va = V
                Wa = W
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = np.sum(Ua[i*3+j, :] * Va[j*3+k, :] * Wa[k*3+i, :])
                            if abs(val - 1.0) > 1e-9:
                                break
                else:
                    break
        
        # Final verification
        Ua = U
        Va = V
        Wa = W
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(Ua[i*3+j, :] * Va[j*3+k, :] * Wa[k*3+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return None
        
        # Convert to lists with rounded values
        U_list = [[float(U[i, r]) for r in range(26)] for i in range(9)]
        V_list = [[float(V[i, r]) for r in range(26)] for i in range(9)]
        W_list = [[float(W[i, r]) for r in range(26)] for i in range(9)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank-26
    result26 = _try_rank26()
    if result26 is not None:
        if _verify(result26["U"], result26["V"], result26["W"]):
            return result26
    
    # Fallback to rank-27
    return _schoolbook()
