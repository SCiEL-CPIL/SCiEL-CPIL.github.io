import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the n x m * m x p multiplication."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj, col] = 1.0
                    V[jj * p + kk, col] = 1.0
                    W[kk * n + ii, col] = 1.0
                    col += 1
        return U, V, W

    # Try to construct a rank-32 or lower decomposition
    # Strategy: Use Strassen for 2x2x2 sub-blocks and handle the remaining
    # dimension carefully to potentially save rank through shared terms.
    
    # Approach: Split A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # Split B = [B1; B2] where B1 is 1x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x1 * 1x6 = 2x6, needs 12 rank terms (each output element is a product)
    # A2*B2: 2x2 * 2x6, split B2 into three 2x2 blocks, each Strassen rank 7 = 21
    # Total naive: 33
    #
    # Can we save rank by combining? The key is that A1*B1 produces terms like
    # a_i * b_k for i in {0,1}, k in {0,...,5}
    # And A2*B2 produces terms involving a_{i,1}, a_{i,2} and b_{j,k} for j in {1,2}
    #
    # Let's try a different decomposition:
    # View the problem as computing 6 independent 2x2 multiplications plus corrections
    # 
    # Actually, let me try: split B into 3 blocks of 2 columns
    # C[:, 0:2] = A * B[:, 0:2]
    # C[:, 2:4] = A * B[:, 2:4]  
    # C[:, 4:6] = A * B[:, 4:6]
    #
    # Each A * B[:, k:k+2] is a 2x3 * 3x2 = 2x2 multiplication
    # If we can do each 2x3x2 in rank r, total = 3r
    #
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # C[0,0] = a0*b0 + a1*b2 + a2*b4
    # C[0,1] = a0*b1 + a1*b3 + a2*b5
    # C[1,0] = a3*b0 + a4*b2 + a5*b4
    # C[1,1] = a3*b1 + a4*b3 + a5*b5
    #
    # This is 4 independent bilinear forms, each with 3 terms = 12 total products
    # But can we share? Let's see if rank < 12 is possible for 2x3x2
    #
    # The tensor for 2x3x2 has 4*3*3 = 36 nonzero entries in its 2x3x2x3x2 representation
    # Actually it's a 2x3x2 tensor (for the output) with coefficients from 2x3 and 3x2 inputs
    #
    # Let me just try to find the best known or constructible rank for 2x3x2
    # and see if 3*r < 33, i.e., r < 11
    
    # For now, let's try the rank-33 construction and see if we can shave off 1
    # by finding a linear dependence
    
    # Construct rank-33 first
    U = np.zeros((n * m, 33))
    V = np.zeros((m * p, 33))
    W = np.zeros((p * n, 33))
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] - 12 terms
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:] using Strassen for each 2x2 block of B
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
    
    # Verify rank-33
    if verify_decomposition(U, V, W, n, m, p):
        # Try to see if we can reduce by 1
        # Check if any column is a linear combination of others
        # For now, return rank 33
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U_school, V_school, W_school = schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
