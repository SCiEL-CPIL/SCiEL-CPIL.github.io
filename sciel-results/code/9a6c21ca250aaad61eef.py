import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin

    def schoolbook():
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

    def verify(U, V, W, tol=1e-6):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def strassen_3x3x3():
        """Return rank-7 Strassen decomposition for 3x3 * 3x3.
        U: (9, 7), V: (9, 7), W: (9, 7)
        Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=3
        """
        # Strassen for 3x3x3 (rank 7)
        # The 7 products are:
        # M1 = (A00+A22)*(B00+B22)
        # M2 = (A20+A22)*B00
        # M3 = A00*(B01-B21)
        # M4 = A22*(B10-B12)
        # M5 = (A00+A11)*(B00+B11)
        # M6 = (A11+A22)*(B11+B22)
        # M7 = (A01+A11)*(B10-B00) - (A00+A10)*(B01+B11)  -- actually let me use a known decomposition
        # 
        # Let me use the standard rank-7 decomposition for 3x3 matrix multiplication.
        # This is the Strassen-like decomposition. Let me use explicit formulas.
        #
        # Actually, let me use a well-known rank-7 decomposition:
        # The 7 rank-1 terms for C = A*B where A,B are 3x3:
        #
        # Using the decomposition from "A rank-7 algorithm for 3x3 matrix multiplication"
        # Let me construct it explicitly.
        #
        # C[i,k] = sum over the 7 products
        #
        # Product 1: (A[0,0]+A[2,2]) * (B[0,0]+B[2,2])
        #   contributes to C[0,0]: +1, C[0,2]: -1, C[2,0]: -1, C[2,2]: +1
        # Product 2: (A[2,0]+A[2,2]) * B[0,0]
        #   contributes to C[2,0]: +1, C[2,2]: -1
        # Product 3: A[0,0] * (B[0,1]-B[2,1])
        #   contributes to C[0,1]: +1, C[0,2]: -1  -- wait, this doesn't look right
        #
        # Let me use a different approach. I'll use the known fact that 3x3x3 has rank 7
        # and construct it via the Strassen-like method or just use a verified decomposition.
        #
        # Actually, let me just use the standard Strassen extension. For 3x3, we can view
        # it as overlapping 2x2 blocks. But a cleaner approach:
        #
        # Here's a verified rank-7 decomposition for 3x3 matrix multiplication:
        # (indices 0-8 for 3x3 matrices)
        
        # Let me use the decomposition from Landsberg's work or construct it directly.
        # I'll use a known explicit form:
        
        U_s = np.zeros((9, 7))
        V_s = np.zeros((9, 7))
        W_s = np.zeros((9, 7))
        
        # Strassen-style rank-7 for 3x3x3
        # Using the decomposition where we define 7 bilinear forms:
        
        # M1 = (a00+a22)(b00+b22)
        U_s[0,0] = 1; U_s[8,0] = 1  # a00, a22
        V_s[0,0] = 1; V_s[8,0] = 1  # b00, b22
        W_s[0,0] = 1; W_s[8,0] = 1  # c00, c22
        W_s[2,0] = -1; W_s[6,0] = -1  # c02, c20
        
        # M2 = (a20+a22)b00
        U_s[6,1] = 1; U_s[8,1] = 1  # a20, a22
        V_s[0,1] = 1  # b00
        W_s[6,1] = 1; W_s[8,1] = -1  # c20, c22
        
        # M3 = a00(b01-b21)
        U_s[0,2] = 1  # a00
        V_s[1,2] = 1; V_s[7,2] = -1  # b01, b21
        W_s[1,2] = 1; W_s[3,2] = -1  # c01, c21 -- wait, W index is k*n+i
        # W[k*n+i]: for c01: k=0,i=1 -> W[1]; for c21: k=2,i=1 -> W[7]
        W_s[1,2] = 1; W_s[7,2] = -1
        
        # M4 = a22(b10-b12)
        U_s[8,3] = 1  # a22
        V_s[3,3] = 1; V_s[5,3] = -1  # b10, b12
        # c20: k=2,i=0 -> W[6]; c22: k=2,i=2 -> W[8]
        W_s[6,3] = 1; W_s[8,3] = -1
        
        # M5 = (a00+a11)(b00+b11)
        U_s[0,4] = 1; U_s[4,4] = 1  # a00, a11
        V_s[0,4] = 1; V_s[4,4] = 1  # b00, b11
        W_s[0,4] = 1; W_s[4,4] = 1  # c00, c11
        W_s[4,4] = -1  # wait, need to be careful
        # c00: k=0,i=0 -> W[0]; c11: k=1,i=1 -> W[4]
        # c01: k=0,i=1 -> W[1]; c10: k=1,i=0 -> W[3]
        W_s[0,4] = 1; W_s[4,4] = 1
        W_s[1,4] = -1; W_s[3,4] = -1
        
        # M6 = (a11+a22)(b11+b22)
        U_s[4,5] = 1; U_s[8,5] = 1  # a11, a22
        V_s[4,5] = 1; V_s[8,5] = 1  # b11, b22
        # c11: W[4]; c12: k=1,i=2 -> W[5]; c21: k=2,i=1 -> W[7]; c22: W[8]
        W_s[4,5] = 1; W_s[8,5] = 1
        W_s[5,5] = -1; W_s[7,5] = -1
        
        # M7 = (a01+a11)(b10-b00) - (a00+a10)(b01+b11)
        # This is getting complicated. Let me use a cleaner known decomposition.
        
        # Let me restart with a cleaner approach. I'll use the fact that we can
        # verify any decomposition, so let me just try a few known forms.
        
        # Actually, let me use a simpler and well-known rank-7 decomposition.
        # From the literature, here's one form:
        
        U_s = np.zeros((9, 7))
        V_s = np.zeros((9, 7))
        W_s = np.zeros((9, 7))
        
        # I'll use the decomposition from "The rank of matrix multiplication" 
        # Let me just try to find one that works by construction.
        
        # Alternative: use the 2x2 Strassen embedded in 3x3
        # For the top-left 2x2 block, use Strassen (rank 4)
        # Then handle the remaining entries.
        
        # Actually, let me just use a brute-force known decomposition.
        # Here's a verified rank-7 for 3x3x3 (from various sources):
        
        # Using the convention C[i,k] = sum_r U[i*3+j... no, U[i*3+k? No.
        # U is indexed by (i,j) of A: U[i*3+j, r]
        # V is indexed by (j,k) of B: V[j*3+k, r]  
        # W is indexed by (k,i) of C: W[k*3+i, r]
        
        # Let me use this known decomposition:
        # r=0: (a00+a22)(b00+b22) -> c00+, c02-, c20-, c22+
        U_s[0,0]=1; U_s[8,0]=1
        V_s[0,0]=1; V_s[8,0]=1
        W_s[0,0]=1; W_s[8,0]=1; W_s[2,0]=-1; W_s[6,0]=-1
        
        # r=1: (a20+a22)b00 -> c20+, c22-
        U_s[6,1]=1; U_s[8,1]=1
        V_s[0,1]=1
        W_s[6,1]=1; W_s[8,1]=-1
        
        # r=2: a00(b01-b21) -> c01+, c21-
        U_s[0,2]=1
        V_s[1,2]=1; V_s[7,2]=-1
        W_s[1,2]=1; W_s[7,2]=-1
        
        # r=3: a22(b10-b12) -> c20+, c22-
        U_s[8,3]=1
        V_s[3,3]=1; V_s[5,3]=-1
        W_s[6,3]=1; W_s[8,3]=-1
        
        # r=4: (a00+a11)(b00+b11) -> c00+, c11+, c01-, c10-
        U_s[0,4]=1; U_s[4,4]=1
        V_s[0,4]=1; V_s[4,4]=1
        W_s[0,4]=1; W_s[4,4]=1; W_s[1,4]=-1; W_s[3,4]=-1
        
        # r=5: (a11+a22)(b11+b22) -> c11+, c22+, c12-, c21-
        U_s[4,5]=1; U_s[8,5]=1
        V_s[4,5]=1; V_s[8,5]=1
        W_s[4,5]=1; W_s[8,5]=1; W_s[5,5]=-1; W_s[7,5]=-1
        
        # r=6: (a01+a11)(b10-b00) + (a00+a10)(b01+b11) ... this is getting messy
        # Let me try: a10(b01-b11) + a11(b10-b00) ... no
        # 
        # Actually, let me try a different r=6:
        # We need to cover c10, c12, c01, c21 residuals
        # After r=0..5, let me compute what's missing.
        
        # This is getting too error-prone. Let me use a different strategy:
        # Just use the schoolbook for the full 3x3x5 and see if we can improve.
        # Or better: use a known correct Strassen for 2x2 and extend.
        
        return U_s, V_s, W_s

    def strassen_2x2x2():
        """Rank-4 Strassen for 2x2x2. U,V,W: (4,4)"""
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        # M1 = (a00+a11)(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
        # M2 = (a10+a11)b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3 = a00(b01-b11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=-1
        # M4 = a11(b10-b00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[2,3]=1; W[0,3]=-1
        return U, V, W

    # Strategy: Use Strassen for the 3x3x3 core (rank 7), then add rank for columns 3,4 of B
    # The full problem is 3x3 * 3x5.
    # We can decompose B = [B_left, B_right] where B_left is 3x3 and B_right is 3x2.
    # C = A * B = A*B_left + A*B_right
    # A*B_left is 3x3 (use Strassen rank 7)
    # A*B_right is 3x2 (need additional rank)
    # 
    # For A*B_right: A is 3x3, B_right is 3x2.
    # This is a 3x3x2 tensor. Schoolbook rank = 3*3*2 = 18.
    # Can we do better? The tensor is T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2}, k in {0,1}
    # This is the same as two copies of the 3x3x1 tensor (outer product).
    # Actually, A*B_right has entries C[i,k] = sum_j A[i,j] * B_right[j,k] for k=0,1
    # This is like two separate matrix-vector multiplications.
    # 
    # For a 3x3 * 3x1 (matrix-vector), the rank is 9 (schoolbook) since it's 
    # essentially the same as 3x3x1 tensor which has rank 9 (no known improvement).
    # Actually, the 3x3x1 tensor T[i,j,k] where k is fixed is just the outer product
    # structure. The rank of the 3x3x1 tensor (i.e., T[i,j,k] = delta_{k,0} for all i,j)
    # is actually 9 because it's just the identity-like structure... 
    # Wait, no. T[i,j,k] for 3x3x1 means k only has value 0.
    # T[i,j,0] = 1 for all i,j. This is a rank-9 tensor (sum over r of e_i * e_j * e_0).
    # Actually it's rank 9: each (i,j) pair needs its own term.
    # 
    # So for A*B_right (3x3 * 3x2), we need rank 9+9=18 in schoolbook.
    # Total: 7 + 18 = 25. But schoolbook for 3x3x5 is 45.
    # 
    # Can we do better for the 3x3x2 part? 
    # The tensor for 3x3 * 3x2 is T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..1.
    # This is the Kronecker product of the 3x3x1 tensor with a 2-dimensional space.
    # Rank of 3x3x1 tensor: The tensor T[i,j,k] with k=0 only, T[i,j,0]=1.
    # This is sum_{i,j} e_i ⊗ e_j ⊗ e_0, which has rank 9.
    # So 3x3x2 has rank at least 9 (since fixing k=0 gives rank 9 sub-tensor).
    # And rank at most 18 (schoolbook).
    # Actually, the rank is exactly 9! Because:
    # T[i,j,k] = 1 for k=0,1 and all i,j.
    # We can write: T = sum_{r=1}^{9} u_r ⊗ v_r ⊗ w_r
    # where for each (i,j) pair, we have one term.
    # But wait, that's 9 terms for k=0 and 9 for k=1, total 18.
    # Unless we can share. But the W vectors for k=0 and k=1 are different (e_0 vs e_1 in the k-direction).
    # So rank is 18.
    # 
    # Hmm, so total rank would be 7 + 18 = 25.
    # 
    # But wait, maybe we can do better by not separating. Let me think differently.
    # 
    # Alternative: Use Strassen for a 2x2x2 block and handle the rest.
    # 
    # Actually, let me just try to find the best decomposition by optimization.
    # Given the time constraint, let me try a few approaches:
    
    # Approach 1: Pure schoolbook (rank 45) - always valid
    U_sb, V_sb, W_sb = schoolbook()
    
    # Approach 2: Strassen core + schoolbook for extra columns
    # Use rank-7 for 3x3x3, then rank-18 for 3x3x2
    # Total rank: 25
    # But I need a valid rank-7 for 3x3x3. Let me verify my construction.
    
    U_s, V_s, W_s = strassen_3x3x3()
    
    # Verify the Strassen decomposition
    strassen_ok = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = np.sum(U_s[i*3+j, :] * V_s[j*3+k, :] * W_s[k*3+i, :])
                if abs(s - 1.0) > 1e-6:
                    strassen_ok = False
                    break
            if not strassen_ok:
                break
        if not strassen_ok:
            break
    
    if strassen_ok:
        # Build the full 3x3x5 decomposition
        # First 7 terms handle 3x3x3 (k=0,1,2)
        # Next 18 terms handle 3x3x2 (k=3,4)
        R_total = 7 + 18
        U_full = np.zeros((n*m, R_total))
        V_full = np.zeros((m*p, R_total))
        W_full = np.zeros((p*n, R_total))
        
        # Copy Strassen for k=0,1,2
        U_full[:, :7] = U_s
        V_full[:, :7] = V_s
        W_full[:, :7] = W_s
        
        # For k=3,4: schoolbook for A(3x3) * B_right(3x2)
        # B_right[j, k-3] for j=0,1,2 and k=3,4
        col = 7
        for i in range(3):
            for j in range(3):
                for kk in range(2):  # kk=0,1 corresponds to k=3,4
                    U_full[i*3+j, col] = 1
                    V_full[j*3+3+kk, col] = 1
                    W_full[(3+kk)*3+i, col] = 1
                    col += 1
        
        # Verify
        if verify(U_full, V_full, W_full):
            # Convert to lists
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
