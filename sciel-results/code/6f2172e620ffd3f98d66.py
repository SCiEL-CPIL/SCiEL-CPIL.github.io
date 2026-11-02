import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 115
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition(n, m, p)
    
    # === STRASSEN-BASED APPROACH ===
    # M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{1,2,2} has rank 7 via Strassen (instead of 4 schoolbook)
    # M_{3,2,4} has rank 24 schoolbook
    # Total: 24 * 7 = 168 (worse than schoolbook 96)
    # 
    # Better: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} rank 2
    # M_{3,4,4} rank 48 schoolbook
    # Total: 48 * 2 = 96 (same as schoolbook)
    #
    # Alternative: Use Strassen on 2x2 blocks within the 4x8 dimension
    # Split B (4x8) into four 2x2 blocks along the 8-dimension
    # B = [[B00, B01], [B10, B11]] where each Bxy is 2x2
    # A (3x4) = [[A0, A1]] where A0, A1 are 3x2
    # C[i,k] for k in 0..7
    # 
    # Actually, let's think differently.
    # M_{3,4,8}: A is 3x4, B is 4x8, C is 3x8
    # 
    # Key insight: We can use the fact that for each row i of A (3 rows),
    # we compute A[i,:] * B which gives a 1x8 vector.
    # A[i,:] is 1x4, B is 4x8.
    # 
    # For 1x4 * 4x8 -> 1x8:
    # Schoolbook: 4*8 = 32 multiplications
    # 
    # Can we do better? Split the 4 into 2+2:
    # A[i,:] = [a0, a1, a2, a3]
    # B columns: b0, b1, b2, b3 (each 8-dim)
    # C[i,k] = a0*b0[k] + a1*b1[k] + a2*b2[k] + a3*b3[k]
    # 
    # This is 4 independent 1x1*1x8 products, total 32 mults.
    # 
    # Winograd for 1x4*4x8 doesn't help directly.
    # 
    # Let's try a different structure:
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # If we can find a decomposition of M_{3,4,4} with rank < 48,
    # then total rank < 96.
    #
    # M_{3,4,4}: 3x4 * 4x4 -> 3x4
    # Known: rank(M_{3,4,4}) <= 48 (schoolbook)
    # Can we do better?
    #
    # Try: M_{3,4,4} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{3,2,4}: 3x2 * 2x4 -> 3x4, rank 24 schoolbook
    # M_{1,2,2}: rank 4 schoolbook
    # Total: 24*4 = 96 (no savings)
    #
    # With Strassen: M_{1,2,2} rank 7
    # Total: 24*7 = 168 (worse)
    #
    # The issue is that Strassen increases rank for small blocks.
    #
    # Let's try yet another approach:
    # Direct construction with small integer entries.
    # 
    # For rank r, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    #
    # This is equivalent to: the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # should equal the identity-like tensor where T[i,j,k]=1 iff the (i,j,k) triple
    # corresponds to a valid multiplication term.
    #
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for ALL i,j,k
    #
    # This is a system of n*m*p = 96 equations in r* (n*m + m*p + p*n) = r*(12+32+24) = 68r unknowns.
    #
    # For r=96: 96 eqs, 6528 unknowns (very underdetermined)
    # For r=50: 96 eqs, 3400 unknowns
    # For r=40: 96 eqs, 2720 unknowns
    #
    # The constraint is that entries should be small integers/half-integers.
    #
    # Let's try a structured approach:
    # Fix W to have simple structure, solve for U and V.
    #
    # Alternative: Use the fact that the problem is equivalent to finding
    # a rank-r decomposition of the 3x4x8 all-ones tensor.
    #
    # The all-ones tensor of size a x b x c has rank min(a*b, a*c, b*c) in some cases?
    # No, that's not right either.
    #
    # Actually, the matrix multiplication tensor M_{n,m,p} is NOT the all-ones tensor.
    # It's the tensor where T[i,j,k] = 1 if the indices are valid, which for
    # standard matrix multiplication means all triples (i,j,k) with i<n, j<m, k<p.
    # So yes, it IS the all-ones tensor of size n x m x p!
    #
    # Wait, no. Let me re-examine.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (using the convention U[i*m+j,c], V[j*p+k,c], W[k*n+i,c])
    #
    # So yes, we need to decompose the all-ones tensor of size 3x4x8.
    #
    # The rank of the all-ones tensor a x b x c is known to be:
    # rank(1_{a x b x c}) = min over decompositions...
    #
    # Actually, the border rank and rank of the all-ones tensor:
    # For 1_{2x2x2}, rank = 2 (not 4!)
    # Because: 1 = (1/2)(1,1,1)(1,1,1)(1,1,1) + ... no that's not right.
    #
    # Let me think again. The all-ones tensor T[i,j,k] = 1.
    # We want T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # where u_c is a vector of length a, v_c of length b, w_c of length c.
    #
    # For a=b=c=2: Can we do rank 2?
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1
    #
    # With rank 2: sum_{c=0,1} u_c[i] * v_c[j] * w_c[k] = 1
    # This means the 2x2x2 all-ones tensor has rank 2.
    #
    # In general, 1_{a x b x c} has rank min(a, b, c) * max(a, b, c) / min(a,b,c)... no.
    #
    # Actually, I recall that the rank of the all-ones tensor a x b x c is:
    # If a <= b <= c, then rank = a * b? No...
    #
    # Let's think: 1_{a x b x c} = sum_{c'=1}^{r} u_{c'} ⊗ v_{c'} ⊗ w_{c'}
    # where u_{c'} in R^a, v_{c'} in R^b, w_{c'} in R^c.
    #
    # The matrix unfolding: T_{(a,b),c} is an (ab) x c matrix of all ones.
    # rank(T_{(a,b),c}) = 1.
    # But tensor rank >= max of matrix ranks = 1. That's a weak bound.
    #
    # Actually, for the all-ones tensor, I believe:
    # rank(1_{a x b x c}) = min(a*b, a*c, b*c) when one dimension is 1?
    # No, for 1_{1 x b x c}, it's just a matrix of all ones, rank 1.
    #
    # For 1_{2 x 2 x 2}: Let's verify rank 2.
    # u_0 = [1, 0], v_0 = [1, 0], w_0 = [1, 0]
    # u_1 = [0, 1], v_1 = [1, 0], w_1 = [1, 0]
    # sum = u_0⊗v_0⊗w_0 + u_1⊗v_1⊗w_1
    # T[0,0,0] = 1*1*1 + 0*1*1 = 1 ✓
    # T[0,0,1] = 1*1*0 + 0*1*0 = 0 ✗ (should be 1)
    #
    # So that doesn't work. Let me try:
    # u_0 = [1, 1], v_0 = [1, 0], w_0 = [1, 0]
    # u_1 = [0, 0], v_1 = [1, 1], w_1 = [0, 1]
    # T[0,0,0] = 1*1*1 + 0*1*0 = 1
    # T[0,0,1] = 1*1*0 + 0*1*1 = 0 ✗
    #
    # Hmm. Let me try rank 3 for 2x2x2:
    # Actually, I think the rank of 1_{2x2x2} is 2.
    # u_0 = [1, 1], v_0 = [1, 1], w_0 = [1, 1]
    # u_1 = [1, -1], v_1 = [1, 1], w_1 = [1, -1]
    # sum = (1,1)⊗(1,1)⊗(1,1) + (1,-1)⊗(1,1)⊗(1,-1)
    # T[0,0,0] = 1*1*1 + 1*1*1 = 2 ✗
    #
    # Scale: (1/2)(1,1)⊗(1,1)⊗(1,1) + (1/2)(1,-1)⊗(1,1)⊗(1,-1)
    # T[0,0,0] = (1/2)(1) + (1/2)(1) = 1 ✓
    # T[0,0,1] = (1/2)(1) + (1/2)(-1) = 0 ✗
    #
    # This is getting complicated. Let me just use the known result:
    # The rank of the all-ones tensor 1_{a x b x c} is min(a, b, c) * max(a, b, c) / min(a,b,c) ... 
    #
    # Actually, I found it: rank(1_{a x b x c}) = min(a*b, a*c, b*c) is NOT correct.
    #
    # For our case: 1_{3 x 4 x 8}
    # 
    # Let me just try to find a low-rank decomposition numerically.
    #
    # Approach: Use the fact that we can set up a least-squares problem.
    # For a given rank r, we want to find U (12 x r), V (32 x r), W (24 x r)
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    #
    # This is a bilinear problem. We can use alternating minimization:
    # Fix U and V, solve for W (linear system).
    # Fix U and W, solve for V.
    # Fix V and W, solve for U.
    #
    # Let's implement this.
    
    def alternating_minimization(n, m, p, target_rank, max_iters=200, seed_base=42, tol=1e-8):
        """Alternating minimization for rank decomposition."""
        for attempt in range(10):
            if time.time() - start_time > deadline - 10:
                return None
            
            seed = seed_base + attempt * 1000 + target_rank
            np.random.seed(seed)
            
            # Initialize U, V, W with small random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for it in range(max_iters):
                if time.time() - start_time > deadline - 10:
                    return None
                
                # Fix U and V, solve for W
                # For each (i,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
                # This gives m equations for r unknowns W[k*n+i,:]
                # Let M[i,k,c,j] = U[i*m+j,c]*V[j*p+k,c]
                # Then sum_c M[i,k,c,j]*W[k*n+i,c] = 1 for all j
                # This is a linear system: M[i,k,:] @ W[k*n+i,:] = ones(m)
                
                W_new = np.zeros((p*n, target_rank))
                valid = True
                for i in range(n):
                    for k in range(p):
                        # Build the m x r system
                        A_sys = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Solve A_sys @ W_row = ones(m)
                        try:
                            W_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=tol)
                            W_new[k*n+i, :] = W_row
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                W = W_new
                
                # Fix V and W, solve for U
                U_new = np.zeros((n*m, target_rank))
                valid = True
                for i in range(n):
                    for j in range(m):
                        A_sys = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        try:
                            U_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(p), rcond=tol)
                            U_new[i*m+j, :] = U_row
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                U = U_new
                
                # Fix U and W, solve for V
                V_new = np.zeros((m*p, target_rank))
                valid = True
                for j in range(m):
                    for k in range(p):
                        A_sys = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            V_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(n), rcond=tol)
                            V_new[j*p+k, :] = V_row
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                V = V_new
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (s - 1.0)**2
                if err < 1e-12:
                    break
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
            
            # Try with integer rounding
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something...
    # For all-ones tensor 3x4x8, rank is at least min(3*4, 3*8, 4*8) = 12? No.
    # Actually, rank >= max(n, m, p) = 8? No.
    # 
    # Let's just try a range of ranks.
    
    for target_rank in range(12, best_rank + 1):
        if time.time() - start_time > deadline - 10:
            break
        
        result = alternating_minimization(n, m, p, target_rank, max_iters=100, max_attempts=5)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the minimal rank, stop
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
