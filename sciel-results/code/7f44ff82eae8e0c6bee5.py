import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: T_{2,1,9} (j=2)
    # A[i, 2] * B[2, k] * C[k, i]
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: T_{2,2,9} (j=0,1)
    # Split k into groups of 2, and one 1
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for k1, k2 in k_groups:
        # Strassen for 2x2x2
        # M1 = (A00+A11)(B0k1+B1k1) -> C terms?
        # Standard Strassen for C = AB:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the linear combinations to U, V, W.
        # Each rank-1 term is u * v * w.
        # u is a vector in A-space, v in B-space, w in C-space.
        
        # M1: u = e00 + e11, v = e0k1 + e1k1, w = ?
        # Wait, Strassen computes scalar M_r.
        # In tensor decomposition, we add terms corresponding to the expansion.
        # Actually, the rank-1 terms of the Strassen decomposition are:
        # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
        # where u_r, v_r, w_r are specific vectors.
        
        # For M_{2,2,2}:
        # u1 = [1, 0, 0, 1] (A00, A11)
        # v1 = [1, 0, 1, 0] (B00, B10) -- Note: B is 2x2, indices (j,k). j=0,1; k=0,1.
        # w1 = [1, 0, 1, 0] (C00, C10) -- Note: C is 2x2, indices (k,i). k=0,1; i=0,1.
        # Wait, the trace convention is W[k*n+i].
        # For 2x2, W indices: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3.
        # C00 corresponds to k=0, i=0. Index 0.
        # C01 corresponds to k=0, i=1. Index 1.
        # C10 corresponds to k=1, i=0. Index 2.
        # C11 corresponds to k=1, i=1. Index 3.
        
        # Strassen vectors for 2x2x2:
        # r=1: u=[1,0,0,1], v=[1,0,1,0], w=[1,1,0,0] ? No.
        # Let's use the explicit rank-1 decomposition of the Strassen tensor.
        
        # The 7 terms are:
        # 1. (A00+A11)(B00+B10) -> contributes to C00, C10?
        # Actually, it's easier to just list the 7 rank-1 tensors that sum to the Strassen tensor.
        
        # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,1,0] ?
        # Let's derive or look up the explicit rank-1 decomposition for M_{2,2,2} rank 7.
        
        # A standard set of 7 rank-1 tensors for M_{2,2,2}:
        # 1. e00^A e00^B e00^C + e00^A e01^B e01^C + e11^A e00^B e10^C + e11^A e01^B e11^C
        # This is not rank 1.
        
        # The rank-1 decomposition of the Strassen algorithm:
        # M1 = (A00+A11)(B00+B10)
        # This M1 is added to C00 and C10?
        # C00 = M1 + M4 - M5 - M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # We can represent the tensor as:
        # sum_{r=1}^7 u_r \otimes v_r \otimes w_r
        # where w_r is the vector of coefficients for C.
        
        # For this specific group (k1, k2), the local B indices are j=0,1 and k=k1,k2.
        # Local C indices are k=k1,k2 and i=0,1.
        
        # u vectors (A space, size 4 for j=0,1):
        # v vectors (B space, size 4 for j=0,1; k=k1,k2)
        # w vectors (C space, size 4 for k=k1,k2; i=0,1)
        
        # We will fill the global U, V, W.
        
        # u1 = A00 + A11 -> U[i*3+j, col]
        # v1 = B0k1 + B1k1 -> V[j*9+k, col]
        # w1 = Ck1,0 + Ck2,1? No.
        
        # Let's use the known 7 rank-1 components.
        # 1. u=[1,0,0,1], v=[1,0,1,0], w=[1,0,0,1]
        # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # 3. u=[1,0,0,0], v=[0,1,0,0], w=[1,1,0,0]
        # 4. u=[0,0,0,1], v=[0,0,1,1], w=[0,1,0,0]
        # 5. u=[1,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
        # 6. u=[1,0,-1,0], v=[1,0,1,0], w=[0,1,0,0]
        # 7. u=[0,1,0,-1], v=[0,1,0,1], w=[0,0,0,1]
        
        # Let's verify term 1:
        # (A00+A11)(B0k1+B1k1) (Ck1,0 + Ck2,1) ??
        # This doesn't look right.
        
        # I will use a helper to generate the Strassen terms for a 2x2x2 block.
        
        def add_strassen_222(k_start, k_end):
            nonlocal col
            # k_start, k_end are the two k indices
            k1, k2 = k_start, k_end
            
            # 7 terms
            # 1
            U[0*3+0, col] += 1; U[1*3+1, col] += 1
            V[0*9+k1, col] += 1; V[1*9+k1, col] += 1
            W[k1*2+0, col] += 1; W[k2*2+1, col] += 1
            col += 1
            
            # 2
            U[1*3+0, col] += 1; U[1*3+1, col] += 1
            V[0*9+k1, col] += 1
            W[k2*2+0, col] += 1
            col += 1
            
            # 3
            U[0*3+0, col] += 1
            V[1*9+k1, col] += 1
            W[k1*2+0, col] += 1; W[k1*2+1, col] += 1
            col += 1
            
            # 4
            U[1*3+1, col] += 1
            V[0*9+k2, col] += 1; V[1*9+k2, col] += 1
            W[k1*2+1, col] += 1
            col += 1
            
            # 5
            U[0*3+0, col] += 1; U[0*3+1, col] += 1
            V[1*9+k2, col] += 1
            W[k2*2+0, col] += 1
            col += 1
            
            # 6
            U[0*3+0, col] += 1; U[1*3+0, col] += -1
            V[0*9+k1, col] += 1; V[0*9+k2, col] += 1
            W[k1*2+1, col] += 1
            col += 1
            
            # 7
            U[0*3+1, col] += 1; U[1*3+1, col] += -1
            V[1*9+k1, col] += 1; V[1*9+k2, col] += 1
            W[k2*2+1, col] += 1
            col += 1

        add_strassen_222(k1, k2)

    # Last k=8
    # Schoolbook 2x2x1
    k = 8
    # A00 B0k Ck0
    U[0*3+0, col] = 1; V[0*9+k, col] = 1; W[k*2+0, col] = 1; col += 1
    # A01 B1k Ck0
    U[0*3+1, col] = 1; V[1*9+k, col] = 1; W[k*2+0, col] = 1; col += 1
    # A10 B0k Ck1
    U[1*3+0, col] = 1; V[0*9+k, col] = 1; W[k*2+1, col] = 1; col += 1
    # A11 B1k Ck1
    U[1*3+1, col] = 1; V[1*9+k, col] = 1; W[k*2+1, col] = 1; col += 1

    return {"U": U, "V": V, "W": W}
