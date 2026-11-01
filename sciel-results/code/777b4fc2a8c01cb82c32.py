import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130.0

    size_u = n * m   # 4
    size_v = m * p   # 18
    size_w = p * n   # 18
    n_tensors = n * m * p  # 36

    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 36."""
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2x2():
        """
        Return (U, V, W) for the 2x2x2 tensor T[i,j,k]=1 with rank 7.
        U: (4,7), V: (4,7), W: (4,7) for the local 2x2x2 problem.
        
        Using the standard Strassen decomposition:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B00)
        M4 = A11(B01-B11)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M3 - M5 - M7
        C01 = M4 + M5
        C10 = M2 + M3
        C11 = M1 - M2 + M4 + M7
        
        So W coefficients:
        C00: M1+1, M3+1, M5-1, M7-1
        C01: M4+1, M5+1
        C10: M2+1, M3+1
        C11: M1+1, M2-1, M4+1, M7+1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00 += 1
        W[3, 0] = 1  # C11 += 1
        
        # M2: (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10 += 1
        
        # M3: A00(B10-B00)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[0, 2] = -1 # B00
        W[0, 2] = 1  # C00 += 1
        W[2, 2] = 1  # C10 += 1
        
        # M4: A11(B01-B11)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[3, 3] = -1 # B11
        W[1, 3] = 1  # C01 += 1
        
        # M5: (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00 -= 1
        W[1, 4] = 1  # C01 += 1
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        # M6 doesn't appear in the C formula above... 
        # Let me recheck. The standard 7-term Strassen has:
        # C00 = M1 + M3 - M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        # So M6 is not used? That can't be right for 7 terms.
        
        # Actually, I think the issue is that different formulations of Strassen
        # use different sets of 7 products. Let me use a verified set.
        
        # Let me try a different approach: just verify numerically.
        
        # Actually, let me use a well-known rank-7 decomposition.
        # One standard set (from Alon, Boppana, Sidorenko):
        # r=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  -> T[0,0,0]
        # r=1: U=[0,0,1,0], V=[1,0,0,0], W=[0,0,1,0]  -> T[1,0,0]
        # r=2: U=[0,1,0,0], V=[1,0,0,0], W=[0,1,0,0]  -> T[0,1,0]
        # r=3: U=[0,0,0,1], V=[1,0,0,0], W=[0,0,0,1]  -> T[1,1,0]
        # This is just schoolbook for k=0 (4 terms).
        
        # Let me try a completely different approach: use the fact that
        # the tensor rank of 2x2x2 is 7, and find a valid decomposition
        # by solving the system of equations.
        
        # Actually, let me just hardcode a known valid decomposition.
        # From the literature, one rank-7 decomposition is:
        
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        # I'll use the following 7 terms (verified by expansion):
        # Term 0: (A00)(B00)(C00)
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 1: (A00)(B01)(C01)
        U[0,1]=1; V[1,1]=1; W[1,1]=1
        # Term 2: (A01)(B00)(C10)  -- wait, this is wrong indexing
        
        # Let me be more careful. T[i,j,k] = 1 means:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}
        
        # A rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Let me just use schoolbook (rank 4) for the 2x2x2 block as a fallback,
        # and focus on the full 2x2x9 problem.
        
        # For the full problem, let me try a block decomposition.
        # Split the 9 columns of B into groups.
        
        return None  # Signal to use a different approach

    # Let me try a different structural approach:
    # For 2x2 * 2x9, we can think of this as 2 independent 2x2 multiplications
    # for the first 2 columns (using Strassen, rank 7), and 7 independent
    # 2x1 multiplications for the remaining 7 columns (each needs rank 2, total 14).
    # Total: 7 + 14 = 21? No, that's not right either.
    
    # Actually, for each column k, we're computing C[:,k] = A * B[:,k]
    # where A is 2x2 and B[:,k] is 2x1.
    # The rank of 2x2x1 matrix multiplication is 4 (schoolbook: 4 mults).
    # So for 9 columns, schoolbook gives 4*9 = 36.
    
    # Can we do better? The total tensor is 2x2x9.
    # The rank of the 2x2x9 tensor is at most 36 (schoolbook).
    # Lower bound: the tensor has a 2x2x2 sub-tensor of rank 7,
    # and 7 more 2x2x1 tensors each of rank at least 2 (actually 4).
    # So rank >= 7 + 7*2 = 21? Or 7 + 7*4 = 35?
    
    # Let me try to construct a rank-35 decomposition:
    # Use Strassen (rank 7) for columns 0,1 and schoolbook (rank 4 each) for columns 2-8.
    # Total: 7 + 7*4 = 35.
    
    def build_hybrid_35():
        R = 35
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen part for k=0,1 (columns 0 and 1 of B)
        # We need 7 terms for the 2x2*2x2 subproblem.
        # Let me use a verified rank-7 decomposition.
        
        # I'll construct it by solving the equations.
        # For k in {0,1}, j in {0,1}, i in {0,1}:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # Let me try a specific decomposition.
        # I'll use the following 7 terms (each is a rank-1 tensor):
        
        strassen_U = [
            [1, 0, 0, 1],  # term 0: A00 + A11
            [0, 0, 1, 1],  # term 1: A10 + A11
            [1, 0, 0, 0],  # term 2: A00
            [0, 0, 0, 1],  # term 3: A11
            [1, 1, 0, 0],  # term 4: A00 + A01
            [0, 0, 1, -1], # term 5: A10 - A00
            [0, 1, 0, -1], # term 6: A01 - A11
        ]
        strassen_V = [
            [1, 0, 0, 1],  # term 0: B00 + B11
            [1, 0, 0, 0],  # term 1: B00
            [0, 0, 1, -1], # term 2: B10 - B00
            [0, 1, 0, -1], # term 3: B01 - B11
            [0, 0, 0, 1],  # term 4: B11
            [1, 1, 0, 0],  # term 5: B00 + B01
            [0, 0, 1, 1],  # term 6: B10 + B11
        ]
        strassen_W = [
            [1, 0, 0, 1],  # term 0: C00 + C11
            [0, 0, 1, 0],  # term 1: C10
            [1, 0, 1, 0],  # term 2: C00 + C10
            [0, 1, 0, 1],  # term 3: C01 + C11
            [-1, 1, 0, 0], # term 4: -C00 + C01
            [0, -1, 0, 0], # term 5: -C01  (need to verify)
            [-1, 0, 0, 1], # term 6: -C00 + C11 (need to verify)
        ]
        
        # Let me verify this decomposition numerically.
        # For each (i,j,k) in {0,1}^3, check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(strassen_U[r][i*2+j] * strassen_V[r][j*2+k] * strassen_W[r][k*2+i] for r in range(7))
                    if abs(val - 1.0) > 1e-8:
                        valid = False
        
        if not valid:
            # Try to fix by adjusting W coefficients
            # Let me use a different set
            pass
        
        # Since verification is failing, let me just use schoolbook for the first 2 columns too.
        # That gives rank 36, same as full schoolbook.
        
        # Let me try yet another approach: use the Kronecker product structure.
        # The 2x2x9 tensor can be seen as a 9-fold sum of 2x2x1 tensors.
        # Each 2x2x1 tensor has rank 4.
        # Total rank 36.
        
        # Is there a way to share factors across the 9 columns?
        # The U factor depends only on (i,j), not on k.
        # The V factor depends on (j,k), so it changes with k.
        # The W factor depends on (k,i), so it also changes with k.
        
        # For a given (i,j), U[i,j,:] is the same across all k.
        # But V[j,k,:] and W[k,i,:] change with k.
        
        # So we can't easily share factors across different k values
        # unless we find a structure in the V and W factors.
        
        # Let me try a completely different approach: use tensor decomposition
        # algorithms to find a low-rank decomposition.
        
        return None  # Signal to use schoolbook

    # Fallback: use schoolbook (rank 36)
    U, V, W = build_schoolbook()
    
    # Try to improve using optimization
    try:
        from scipy.optimize import least_squares
        
        def residuals(params, R):
            size_u = n * m
            size_v = m * p
            size_w = p * n
            
            U_flat = params[:R * size_u].reshape(R, size_u)
            V_flat = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
            W_flat = params[R * (size_u + size_v) :].reshape(R, size_w)
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        val = np.sum(U_flat[:, ui] * V_flat[:, vj] * W_flat[:, wk])
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
        
        # Try ranks from 25 to 35
        for R in range(25, 36):
            if time.time() > deadline - 15:
                break
            
            n_params = R * (size_u + size_v + size_w)
            np.random.seed(42 + R)
            x0 = np.random.randn(n_params) * 0.1
            
            try:
                result = least_squares(
                    residuals, 
                    x0, 
                    args=(R,), 
                    method='lm',
                    max_nfev=200
                )
                
                res = residuals(result.x, R)
                if np.max(np.abs(res)) < 1e-8:
                    U_flat = result.x[:R * size_u].reshape(R, size_u)
                    V_flat = result.x[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
                    W_flat = result.x[R * (size_u + size_v) :].reshape(R, size_w)
                    
                    U_new = U_flat.T
                    V_new = V_flat.T
                    W_new = W_flat.T
                    
                    if verify(U_new, V_new, W_new):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = R
                        break
            except Exception:
                pass
        
        U, V, W = best_U, best_V, best_W
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
