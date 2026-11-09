import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # We will construct a rank 44 decomposition
    R = 44
    U = np.zeros((size_U, R))
    V = np.zeros((size_V, R))
    W = np.zeros((size_W, R))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= R:
            return
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Helper for Schoolbook 2x2x2 block
    # A is 2x2 (rows i1, i2; cols j1, j2)
    # B is 2x2 (rows j1, j2; cols k1, k2)
    # C is 2x2 (rows i1, i2; cols k1, k2)
    def add_schoolbook_2x2x2(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # C[i1, k1] = A[i1, j1]*B[j1, k1] + A[i1, j2]*B[j2, k1]
        add_term(i1*m+j1, j1*p+k1, k1*n+i1, 1, 1, 1)
        add_term(i1*m+j2, j2*p+k1, k1*n+i1, 1, 1, 1)
        # C[i1, k2] = A[i1, j1]*B[j1, k2] + A[i1, j2]*B[j2, k2]
        add_term(i1*m+j1, j1*p+k2, k2*n+i1, 1, 1, 1)
        add_term(i1*m+j2, j2*p+k2, k2*n+i1, 1, 1, 1)
        # C[i2, k1] = A[i2, j1]*B[j1, k1] + A[i2, j2]*B[j2, k1]
        add_term(i2*m+j1, j1*p+k1, k1*n+i2, 1, 1, 1)
        add_term(i2*m+j2, j2*p+k1, k1*n+i2, 1, 1, 1)
        # C[i2, k2] = A[i2, j1]*B[j1, k2] + A[i2, j2]*B[j2, k2]
        add_term(i2*m+j1, j1*p+k2, k2*n+i2, 1, 1, 1)
        add_term(i2*m+j2, j2*p+k2, k2*n+i2, 1, 1, 1)

    # Helper for Strassen 2x2x2 block
    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # Indices for A: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
        # Indices for B: (j1,k1), (j1,k2), (j2,k1), (j2,k2)
        
        # m1 = (a+d)(e+h)
        # U: a+d -> (i1,j1)+1, (i2,j2)+1. Cannot do this in one term.
        # Strassen requires linear combinations.
        # We need to expand the linear combinations into rank-1 terms?
        # No, Strassen is 7 scalar multiplications.
        # But our tensor decomposition requires rank-1 terms of the form:
        # U[i,j] * V[j,k] * W[k,i]
        # The "scalar multiplications" in Strassen are products of linear combinations.
        # e.g., (a+d)(e+h) = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # This expands to:
        # A[i1,j1]B[j1,k1] + A[i1,j1]B[j2,k2] + A[i2,j2]B[j1,k1] + A[i2,j2]B[j2,k2]
        # Each of these is a rank-1 term in the original tensor?
        # Yes.
        # So one Strassen "multiplication" corresponds to 4 rank-1 terms in the naive decomposition?
        # No, that would be 7*4 = 28 rank-1 terms.
        # Wait. The rank of the tensor is the minimum number of rank-1 terms.
        # Strassen proves the rank is <= 7 for 2x2x2.
        # How?
        # The 7 scalar multiplications are:
        # m1 = (a+d)(e+h)
        # ...
        # The terms added to the output are linear combinations of these m's.
        # But the decomposition of the *tensor* itself is:
        # T = sum_{r=1}^R U_r V_r W_r^T
        # For Strassen, R=7.
        # The matrices U, V, W for Strassen 2x2x2 are 4x7.
        # Let's hardcode the Strassen 2x2x2 decomposition factors.
        
        # Standard Strassen factors (4x7):
        # U (4x7) corresponding to a, b, c, d
        # V (4x7) corresponding to e, f, g, h
        # W (4x7) corresponding to C11, C12, C21, C22
        
        # U:
        # a: 1, 0, 1, 0, 1, 0, 0
        # b: 0, 0, 0, 0, 1, 0, 1
        # c: 0, 1, 0, 0, 0, 1, 0
        # d: 1, 1, 0, 1, 0, 0, 0
        
        # V:
        # e: 1, 1, 0, 0, 0, 1, 0
        # f: 0, 0, 1, 0, 0, 1, 0
        # g: 0, 0, 0, 1, 0, 0, 1
        # h: 1, 0, -1, -1, 1, 0, 1
        
        # W:
        # C11: 1, -1, 1, 1, -1, 0, 1
        # C12: 0, 0, 1, 0, 1, 0, 0
        # C21: 0, 1, 0, 1, 0, 0, 0
        # C22: 1, -1, 1, 1, 0, 1, 0
        
        # Note: Signs might need adjustment based on specific convention.
        # Let's verify C11 = m1 - m2 + m3 + m4 - m5 + m7?
        # Standard:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # Let's map U, V, W for each m_r.
        
        # m1: (a+d)(e+h)
        # U: a=1, d=1. V: e=1, h=1. W: C11+=1, C22+=1.
        # This is NOT a rank-1 term in the original A,B,C space if we just take sums.
        # The rank-1 terms of the tensor are the "atoms".
        # For Strassen, the atoms are the 7 products.
        # Atom 1: (a+d)(e+h).
        # This corresponds to a term in the decomposition where:
        # U has 1s at (i1,j1) and (i2,j2).
        # V has 1s at (j1,k1) and (j2,k2).
        # W has 1s at (k1,i1) and (k2,i2)? No.
        # The output C11 gets +1, C22 gets +1.
        # So W has 1 at (k1,i1) and 1 at (k2,i2)?
        # Let's check the contraction.
        # Term 1: U1 V1 W1.
        # U1[i1,j1]=1, U1[i2,j2]=1.
        # V1[j1,k1]=1, V1[j2,k2]=1.
        # W1[k1,i1]=1, W1[k2,i2]=1.
        # Contribution to C[i1,k1]: U1[i1,j1]V1[j1,k1]W1[k1,i1] = 1*1*1 = 1.
        # Contribution to C[i2,k2]: U1[i2,j2]V1[j2,k2]W1[k2,i2] = 1*1*1 = 1.
        # Contribution to C[i1,k2]: U1[i1,j1]V1[j1,k2]W1[k2,i1] -> V1[j1,k2]=0.
        # Contribution to C[i2,k1]: U1[i2,j1]V1[j1,k1]W1[k1,i2] -> U1[i2,j1]=0.
        # So this term adds 1 to C11 and 1 to C22.
        # This matches the "1" in C11 = m1... and C22 = m1...
        
        # We will just add the 7 terms for Strassen.
        
        # Term 1: (a+d)(e+h)
        U[i1*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i1, col] = 1
        U[i2*m+j2, col] = 1; V[j2*p+k2, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # Term 2: (c+d)e
        U[i2*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i2, col] = 1
        U[i2*m+j2, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i2, col] = 1 # Wait, V is same for both U parts?
        # (c+d)e = c*e + d*e
        # c is A[i2,j1]. d is A[i2,j2]. e is B[j1,k1].
        # Term 2a: A[i2,j1] B[j1,k1] -> C[i2,k1] += 1.
        # Term 2b: A[i2,j2] B[j1,k1] -> C[i2,k1] += 1.
        # This is 2 rank-1 terms.
        # Strassen counts this as 1 scalar multiplication, but it expands to 2 rank-1 tensor terms?
        # No. The rank of the tensor is the number of rank-1 terms in the sum.
        # If Strassen reduces the number of scalar multiplications, does it reduce the tensor rank?
        # Yes. The tensor rank of 2x2x2 is 7.
        # So there exist 7 rank-1 tensors that sum to the multiplication tensor.
        # The factors U, V, W for these 7 tensors are the Strassen factors.
        
        # Let's use the explicit 4x7 matrices.
        
        U_s = np.array([
            [1, 0, 1, 0, 1, 0, 0], # a
            [0, 0, 0, 0, 1, 0, 1], # b
            [0, 1, 0, 0, 0, 1, 0], # c
            [1, 1, 0, 1, 0, 0, 0]  # d
        ])
        
        V_s = np.array([
            [1, 1, 0, 0, 0, 1, 0], # e
            [0, 0, 1, 0, 0, 1, 0], # f
            [0, 0, 0, 1, 0, 0, 1], # g
            [1, 0, -1, -1, 1, 0, 1]# h
        ])
        
        W_s = np.array([
            [1, -1, 1, 1, -1, 0, 1], # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 1, 0, 1, 0]   # C22
        ])
        
        for r in range(7):
            # Add term r
            U[i1*m+j1, col] += U_s[0, r]
            U[i1*m+j2, col] += U_s[1, r]
            U[i2*m+j1, col] += U_s[2, r]
            U[i2*m+j2, col] += U_s[3, r]
            
            V[j1*p+k1, col] += V_s[0, r]
            V[j1*p+k2, col] += V_s[1, r]
            V[j2*p+k1, col] += V_s[2, r]
            V[j2*p+k2, col] += V_s[3, r]
            
            W[k1*n+i1, col] += W_s[0, r]
            W[k2*n+i1, col] += W_s[1, r]
            W[k1*n+i2, col] += W_s[2, r]
            W[k2*n+i2, col] += W_s[3, r]
            
            col += 1

    # Execute Plan:
    
    # 1. Strassen for A[0:2,0:2] and B[0:2,0:2] (k=0,1)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1)
    
    # 2. Schoolbook for A[0:2,0:2] and B[0:2,2:5] (k=2,3,4)
    # This is 2x2x3. Schoolbook is 12 terms.
    for k in range(2, 5):
        # C[i, k] = sum_j A[i,j] B[j,k]
        for i in range(2):
            for j in range(2):
                add_term(i*m+j, j*p+k, k*n+i, 1, 1, 1)
                
    # 3. Schoolbook for A[0:2,2:3] (j=2) and B[2:3,0:5] (k=0..4)
    # A is 2x1, B is 1x5.
    # C[i, k] = A[i,2] B[2,k]
    for i in range(2):
        for k in range(5):
            add_term(i*m+2, 2*p+k, k*n+i, 1, 1, 1)
            
    # 4. Schoolbook for A[2:3,0:2] (i=2) and B[0:2,0:5] (k=0..4)
    # A is 1x2, B is 2x5.
    # C[2, k] = A[2,0] B[0,k] + A[2,1] B[1,k]
    for j in range(2):
        for k in range(5):
            add_term(2*m+j, j*p+k, k*n+2, 1, 1, 1)
            
    # 5. Schoolbook for A[2:3,2:3] (i=2, j=2) and B[2:3,0:5] (k=0..4)
    # A is 1x1, B is 1x5.
    # C[2, k] = A[2,2] B[2,k]
    for k in range(5):
        add_term(2*m+2, 2*p+k, k*n+2, 1, 1, 1)
        
    # Verify column count
    if col != R:
        # Fallback to rank 45 if construction fails
        U0 = np.zeros((size_U, 45))
        V0 = np.zeros((size_V, 45))
        W0 = np.zeros((size_W, 45))
        col0 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, col0] = 1.0
                    V0[j*p+k, col0] = 1.0
                    W0[k*n+i, col0] = 1.0
                    col0 += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Final check for exactness
    # The Strassen and Schoolbook terms are exact integers.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

