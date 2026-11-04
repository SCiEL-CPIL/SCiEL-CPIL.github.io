import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook rank
    R0 = n * m * p  # 96

    def build_schoolbook():
        U0 = np.zeros((n * m, R0))
        V0 = np.zeros((m * p, R0))
        W0 = np.zeros((p * n, R0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col] = 1.0
                    V0[j * p + k, col] = 1.0
                    W0[k * n + i, col] = 1.0
                    col += 1
        return U0, V0, W0

    U0, V0, W0 = build_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0

    def compute_tensor(U, V, W):
        """Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        N = n * m * p
        I, J, K = np.indices((n, m, p))
        u_idx = (I * m + J).flatten()
        v_idx = (J * p + K).flatten()
        w_idx = (K * n + I).flatten()
        U_rows = U[u_idx, :]
        V_rows = V[v_idx, :]
        W_rows = W[w_idx, :]
        return np.sum(U_rows * V_rows * W_rows, axis=1)

    def verify(U, V, W, atol=1e-7):
        T = compute_tensor(U, V, W)
        return np.allclose(T, 1.0, atol=atol)

    # Try to find lower rank decompositions using structured random search + refinement
    # Key insight: for matrix multiplication tensors, we can use the "partition" structure.
    # Partition the inner dimension m=4 into two halves of size 2, and use Strassen-like
    # bilinear forms. But instead of hardcoding, we solve for the optimal coefficients.

    # Approach: For each candidate rank R < 96, try to find U, V, W of shape
    # (n*m, R), (m*p, R), (p*n, R) such that the tensor equation holds.
    #
    # We use a "block-structured" parameterization:
    # - Split m into m1=2, m2=2
    # - Split p into p1=4, p2=4  
    # - Split n into n1=1, n2=2
    #
    # For the n2=2, m1=2, p1=4 block: use 2x2 Strassen structure (rank 7 per 2x2x2 sub-block)
    # For the n1=1 blocks: schoolbook (rank = m*p per row block)

    # Let's try a more systematic approach: enumerate possible block structures
    # and for each, solve the linear system for the coefficients.

    # Block structure: n -> [1, 2], m -> [2, 2], p -> [4, 4]
    # Block (n=1, m=2, p=4): 1x2x4 tensor, schoolbook rank = 8
    # Block (n=1, m=2, p=4): 1x2x4 tensor, schoolbook rank = 8
    # Block (n=2, m=2, p=4): 2x2x4 tensor
    #   Split p into [2,2]: two 2x2x2 blocks, each Strassen rank 7 -> total 14
    # Block (n=2, m=2, p=4): 2x2x4 tensor
    #   Split p into [2,2]: two 2x2x2 blocks, each Strassen rank 7 -> total 14
    # Total: 8 + 8 + 14 + 14 = 44? That seems too good. Let me reconsider.

    # Actually the tensor is T[i,j,k] where i in 0..2, j in 0..3, k in 0..7.
    # A block decomposition must cover ALL (i,j,k) tuples.
    # If we partition i into {0} and {1,2}, j into {0,1} and {2,3}, k into {0,1,2,3} and {4,5,6,7}:
    #   Block (i={0}, j={0,1}, k={0..3}): 1x2x4, rank 8
    #   Block (i={0}, j={0,1}, k={4..7}): 1x2x4, rank 8
    #   Block (i={0}, j={2,3}, k={0..3}): 1x2x4, rank 8
    #   Block (i={0}, j={2,3}, k={4..7}): 1x2x4, rank 8
    #   Block (i={1,2}, j={0,1}, k={0..3}): 2x2x4, can use Strassen on p-dim: 2*(2x2x2) = 2*7=14
    #   Block (i={1,2}, j={0,1}, k={4..7}): 2x2x4, rank 14
    #   Block (i={1,2}, j={2,3}, k={0..3}): 2x2x4, rank 14
    #   Block (i={1,2}, j={2,3}, k={4..7}): 2x2x4, rank 14
    # Total: 4*8 + 4*14 = 32 + 56 = 88

    # This matches the earlier analysis. Let me implement this carefully.
    # For 2x2x2 Strassen, I need to get the coefficients RIGHT.
    # Let me derive them numerically.

    def derive_strassen_2x2x2():
        """
        Find coefficients for 2x2x2 matrix multiplication tensor.
        T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
        
        We want rank-7 decomposition. The standard Strassen gives:
        For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B:
        C[0,0] = a*e + b*g
        C[0,1] = a*f + b*h
        C[1,0] = c*e + d*g
        C[1,1] = c*f + d*h
        
        In tensor form, T[i,j,k] corresponds to coefficient of A[i,j]*B[j,k] in C[i,k].
        So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul).
        
        We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        
        Standard Strassen 7 terms:
        M1 = (a+d)(e+h)  -> U contributes to (0,0) and (1,1), V to (0,0) and (1,1), W to (0,0) and (1,1)
        M2 = (c+d)(e+f)  -> U: (1,0),(1,1), V: (0,0),(0,1), W: (0,1),(1,0)
        M3 = a(f-h)      -> U: (0,0), V: (0,1),(0,0), W: (1,0),(1,1)
        M4 = d(g-e)      -> U: (1,1), V: (1,0),(0,0), W: (0,0),(0,1)
        M5 = (a+b)h      -> U: (0,0),(0,1), V: (1,1), W: (0,0),(1,0)
        M6 = (c+d)f      -> U: (1,0),(1,1), V: (0,1), W: (1,1),(0,1)
        M7 = (a+b)g      -> U: (0,0),(0,1), V: (1,0), W: (1,0),(0,1)
        
        Wait, I need to be more careful. Let me use the standard formulation.
        
        Actually, let me just solve it numerically. I'll set up the system where
        each of the 7 terms has a specific pattern of nonzero entries in U, V, W,
        and solve for the scalar coefficients.
        """
        # Standard Strassen factorization for 2x2x2:
        # Term 1: M1 = (a+d)(e+h)
        #   U: U[0,0]+=1, U[1,1]+=1
        #   V: V[0,0]+=1, V[1,1]+=1
        #   W: W[0,0]+=1, W[1,1]+=1  (but W index is (k,i))
        #   Actually W[k,i]: for C[0,0] and C[1,1]
        #   C[0,0] gets M1, C[1,1] gets M1
        #   So W[0,0]+=1 (k=0,i=0), W[1,1]+=1 (k=1,i=1)
        
        # Let me use a different approach: just hardcode the well-known Strassen
        # and verify it numerically.
        
        # The 7 terms for 2x2 matmul (C = A*B):
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*(B[0,0]+B[0,1])
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]+A[1,1])*B[0,1]
        # M7 = (A[0,0]+A[0,1])*B[1,0]
        
        # C[0,0] = M1 - M5 + M7 + M4
        # C[0,1] = M3 + M5
        # C[1,0] = M6 - M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Let me verify with A=I, B=I:
        # a=1,b=0,c=0,d=1, e=1,f=0,g=0,h=1
        # M1 = (1+1)(1+1) = 4
        # M2 = (0+1)(1+0) = 1
        # M3 = 1*(0-1) = -1
        # M4 = 1*(0-1) = -1
        # M5 = (1+0)*1 = 1
        # M6 = (0+1)*0 = 0
        # M7 = (1+0)*0 = 0
        # C[0,0] = 4 - 1 + 0 + (-1) = 2. Should be 1. WRONG.
        
        # Let me try another source. From the original Strassen paper:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e+f)  
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c+d)f
        # M7 = (a+b)g
        # C11 = M1 - M2 + M3 + M6
        # C10 = M6 - M4
        # C01 = M5 + M3
        # C00 = M1 - M5 + M7 + M4
        
        # Same as before. Let me try yet another formulation.
        
        # From https://en.wikipedia.org/wiki/Strassen_algorithm:
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e + f)
        # M3 = a(f - h)
        # M4 = d(g - e)
        # M5 = (a + b)h
        # M6 = (c + d)f
        # M7 = (a + b)g
        # C11 = M1 - M2 + M3 + M6
        # C10 = M6 - M4
        # C01 = M5 + M3
        # C00 = M1 - M5 + M7 + M4
        
        # Test with A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # C should be [[19,22],[43,50]]
        # M1 = (1+4)(5+8) = 5*13 = 65
        # M2 = (3+4)(5+6) = 7*11 = 77
        # M3 = 1*(6-8) = -2
        # M4 = 4*(7-5) = 8
        # M5 = (1+2)*8 = 24
        # M6 = (3+4)*6 = 42
        # M7 = (1+2)*7 = 21
        # C11 = 65 - 77 + (-2) + 42 = 28. Should be 50. WRONG.
        
        # There must be a different correct set. Let me look for the actual correct one.
        # 
        # After research, the correct Strassen formulas are:
        # M1 = (a + d) * (e + h)
        # M2 = (c + d) * (e + f)
        # M3 = a * (f - h)
        # M4 = d * (g - e)
        # M5 = (a + b) * h
        # M6 = (c + d) * f
        # M7 = (a + b) * g
        #
        # C[0,0] = M1 - M5 + M7 + M4
        # C[0,1] = M3 + M5
        # C[1,0] = M6 - M4
        # C[1,1] = M1 - M2 + M3 + M6
        #
        # These keep giving wrong answers. Let me try a completely different approach:
        # I'll solve for the coefficients numerically using the constraint that
        # the decomposition must be exact.
        
        # For 2x2x2, the tensor has 8 entries, all equal to 1.
        # A rank-r decomposition gives r terms, each with U (4 entries), V (4 entries), W (4 entries).
        # Total unknowns: 12r. Constraints: 8 equations (but they're nonlinear).
        
        # For r=7, let's use the known structure. I'll parameterize each term by
        # which entries of U, V, W are nonzero, then solve for the values.
        
        # Actually, let me just use a verified implementation. I'll use the one from
        # the NumPy/SciPy ecosystem or derive it by solving a linear system.
        
        # KEY INSIGHT: The tensor T[i,j,k] for 2x2x2 matmul is:
        # T[0,0,0]=1, T[0,0,1]=0, T[0,1,0]=0, T[0,1,1]=1
        # T[1,0,0]=0, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=0
        # Wait no. T[i,j,k] is the coefficient such that C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]
        # For standard matmul, T[i,j,k] = 1 for ALL i,j,k. It's the all-ones tensor!
        # Because C[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) triple contributes.
        
        # So we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k in {0,1}
        
        # Let me just solve this numerically for rank 7.
        # I'll use a random initialization and gradient descent, or better,
        # I'll use the fact that this is a well-known problem and the answer is:
        
        # After careful derivation, here's a correct rank-7 decomposition:
        # Using the basis of linear forms:
        
        # Term 1: U=(1,0,0,1) V=(1,0,0,1) W=(1,0,0,1)  [M1=(a+d)(e+h), contributes to C00,C11]
        # Term 2: U=(0,0,1,1) V=(1,1,0,0) W=(0,1,1,0)  [M2=(c+d)(e+f)]
        # Term 3: U=(1,0,0,0) V=(0,1,0,1) W=(0,1,1,0)  [M3=a(f-h)]  -- need to check W indexing
        # 
        # This is getting too error-prone from memory. Let me use a different strategy:
        # I'll implement the block decomposition using ONLY schoolbook for sub-blocks,
        # but with a smarter partition that still gives rank < 96.
        
        # Partition: n=3 -> [3], m=4 -> [4], p=8 -> [4,4]
        # Two blocks: 3x4x4 and 3x4x4, each schoolbook rank 48. Total 96. No improvement.
        
        # Partition: n=3 -> [1,2], m=4 -> [4], p=8 -> [8]
        # Two blocks: 1x4x8 (rank 32) and 2x4x8 (rank 64). Total 96. No improvement.
        
        # The ONLY way to beat schoolbook is to use Strassen-type savings on 2x2x2 sub-blocks.
        # Let me just hardcode a VERIFIED Strassen decomposition by testing it.
        
        # I'll generate the Strassen decomposition and verify it before using.
        
        # Correct Strassen (finally, after checking multiple sources):
        # For C = A * B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
        #
        # p1 = (a + d) * (e + h)
        # p2 = (c + d) * (e + f)
        # p3 = a * (f - h)
        # p4 = d * (g - e)
        # p5 = (a + b) * h
        # p6 = (c + d) * f
        # p7 = (a + b) * g
        #
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p6 - p4
        # C[1,1] = p1 - p2 + p3 + p6
        
        # Let me verify ONE more time with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
        # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        # p1 = (1+4)*(5+8) = 5*13 = 65
        # p2 = (3+4)*(5+6) = 7*11 = 77
        # p3 = 1*(6-8) = -2
        # p4 = 4*(7-5) = 8
        # p5 = (1+2)*8 = 24
        # p6 = (3+4)*6 = 42
        # p7 = (1+2)*7 = 21
        # C[0,0] = 65 + 8 - 24 + 21 = 70. Expected 19. STILL WRONG.
        
        # I give up trying to remember. Let me SOLVE it.
        # The 7 linear forms for A are:
        # L1 = a + d, L2 = c + d, L3 = a, L4 = d, L5 = a + b, L6 = c + d, L7 = a + b
        # The 7 linear forms for B are:
        # R1 = e + h, R2 = e + f, R3 = f - h, R4 = g - e, R5 = h, R6 = f, R7 = g
        #
        # We need: for each (i,j,k), sum_c coeff[i,j,k,c] * L_c * R_c = T[i,j,k] = 1
        # where coeff is determined by which entries of U,V,W are nonzero.
        
        # Actually, in the tensor decomposition:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        #
        # For term c, U[i,j,c] is a linear form in A, V[j,k,c] in B, W[k,i,c] in C.
        # The product U[i,j,c]*V[j,k,c] must equal a specific bilinear form, and
        # W[k,i,c] is the coefficient that routes it to C[k,i].
        
        # For Strassen, each term c has:
        # U_c is a vector in R^4 (for the 4 entries of A)
        # V_c is a vector in R^4 (for the 4 entries of B)  
        # W_c is a vector in R^4 (for the 4 entries of C, indexed as (k,i))
        
        # And the constraint is: for all i,j,k: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
        # With n=m=p=2.
        
        # Let me just brute-force solve this. I'll fix the support pattern (which entries
        # are nonzero) to match the known Strassen structure, then solve for the values.
        
        # Strassen support pattern (from the original paper, 1969):
        # Term 1: U={0,0:1, 1,1:1}, V={0,0:1, 1,1:1}, W={0,0:1, 1,1:1}
        # Term 2: U={1,0:1, 1,1:1}, V={0,0:1, 0,1:1}, W={0,1:1, 1,0:1}
        # Term 3: U={0,0:1}, V={0,1:1, 0,0:-1}, W={1,0:1, 1,1:1}
        # Term 4: U={1,1:1}, V={1,0:1, 0,0:-1}, W={0,0:1, 0,1:1}
        # Term 5: U={0,0:1, 0,1:1}, V={1,1:1}, W={0,0:1, 1,0:1}
        # Term 6: U={1,0:1, 1,1:1}, V={0,1:1}, W={1,1:1, 0,1:1}
        # Term 7: U={0,0:1, 0,1:1}, V={1,0:1}, W={1,0:1, 0,1:1}
        
        # With all coefficients = 1, let's check if this works.
        # T[0,0,0] = sum_c U[0,0,c]*V[0,0,c]*W[0,0,c]
        # Term 1: 1*1*1 = 1
        # Term 2: 0*1*0 = 0
        # Term 3: 1*0*0 = 0
        # Term 4: 0*0*1 = 0
        # Term 5: 1*0*1 = 0
        # Term 6: 0*0*0 = 0
        # Term 7: 1*0*0 = 0
        # Sum = 1. Correct!
        
        # T[0,0,1] = sum_c U[0,0,c]*V[0,1,c]*W[1,0,c]
        # Term 1: 1*0*0 = 0
        # Term 2: 0*1*1 = 0
        # Term 3: 1*1*1 = 1
        # Term 4: 0*(-1)*0 = 0
        # Term 5: 1*0*1 = 0
        # Term 6: 0*1*0 = 0
        # Term 7: 1*0*1 = 0
        # Sum = 1. But T[0,0,1] should be 1. Correct!
        
        # T[0,1,0] = sum_c U[0,1,c]*V[1,0,c]*W[0,0,c]
        # Term 1: 0*0*1 = 0
        # Term 2: 0*0*0 = 0
        # Term 3: 0*0*0 = 0
        # Term 4: 0*1*1 = 0
        # Term 5: 1*0*1 = 0
        # Term 6: 0*0*0 = 0
        # Term 7: 1*1*0 = 0
        # Sum = 0. But T[0,1,0] should be 1. WRONG!
        
        # The support pattern is wrong. Let me try a different approach.
        # I'll just use schoolbook for everything. Rank 96 is safe.
        # But let me try one more thing: use the "Laderman" algorithm for 3x3x3.
        # Laderman's algorithm uses 23 multiplications for 3x3x3.
        # For 3x4x8, I can embed the 3x3x3 part and use schoolbook for the rest.
        
        # Actually, let me try a completely different approach:
        # Use the fact that we can OVERLAP blocks. Instead of a strict partition,
        # we can have overlapping linear forms that cover the tensor.
        
        # For now, let me just return schoolbook. It's guaranteed correct.
        return None  # Signal that we couldn't find a better decomposition

    # Try to find a better decomposition using random search
    # We'll try ranks from 88 down to 80
    for R in range(88, max(79, R0 - 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Try random initialization + alternating minimization
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(R * 1000 + attempt)
            
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Alternating minimization: fix two, solve for the third
            # This is a non-convex problem, so it may not converge
            # But for small dimensions, it might work
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), U[i*m+j,:] should satisfy
                # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                # This is linear in U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j,:]
                        # For each k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        A_mat = np.zeros((p, R))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_vec = np.ones(p)
                        # Solve A_mat @ U[i*m+j,:] = b_vec
                        try:
                            U[i * m + j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k,:] should satisfy
                # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, R))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_vec = np.ones(n)
                        try:
                            V[j * p + k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i,:] should satisfy
                # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, R))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_vec = np.ones(m)
                        try:
                            W[k * n + i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Check convergence
                if verify(U, V, W, atol=1e-5):
                    if verify(U, V, W, atol=1e-9):
                        best_U, best_V, best_W = U, V, W
                        best_rank = R
                        break
                    else:
                        # Close but not exact, keep going
                        pass
            
            if best_rank < R0:
                break
        
        if best_rank < R0:
            break

    # Return the best decomposition found
    if best_rank < R0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
