import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    # Target: Rank 56 construction using Karatsuba on m and Strassen on n,p blocks
    # A is 2x4, B is 4x8.
    # Split A into A0 (2x2), A1 (2x2) along columns.
    # Split B into B0 (2x8), B1 (2x8) along rows.
    # C = A0*B0 + A1*B1.
    # Karatsuba: C = A0*B0 + A1*B1 + (A0+A1)*(B0+B1) - A0*B0 - A1*B1 ? No.
    # Karatsuba for scalar: xy = (x+y)(y+x) - x^2 - y^2? No.
    # Karatsuba: (a+b)(c+d) = ac + ad + bc + bd.
    # To compute ac+bd with 3 mults: M1 = ac, M2 = bd, M3 = (a+b)(c+d). Then ad+bc = M3 - M1 - M2.
    # So total 3 mults for 2 scalar products? No, that's for 1 product of 2-digit numbers.
    # Here we have 2 matrix products: A0B0 and A1B1.
    # If we treat A0, A1 as "digits" and B0, B1 as "digits", we want A0B0 + A1B1.
    # This is not the standard Karatsuba form (which computes full product).
    # Standard Karatsuba computes A0B0 + A0B1 + A1B0 + A1B1.
    # We only need the diagonal terms A0B0 + A1B1.
    # We can compute:
    # M1 = A0 B0
    # M2 = A1 B1
    # M3 = (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1
    # Then A0B1 + A1B0 = M3 - M1 - M2.
    # This gives us the off-diagonal terms. But we don't need them.
    # So Karatsuba doesn't directly help reduce the rank of A0B0 + A1B1 below 2 * rank(A0B0).
    # Unless we can share the computation.
    
    # Let's look at the block sizes:
    # A0, A1 are 2x2.
    # B0, B1 are 2x8.
    # Product A0B0 is 2x8.
    # Rank of 2x2 * 2x8 multiplication?
    # This is equivalent to multiplying two 2x2 matrices if we view 2x8 as two 2x4 blocks? No.
    # 2x2 * 2x8 is just two independent 2x2 * 2x4? No.
    # It's a linear map from (2x2) x (2x8) to (2x8).
    # Let's decompose 2x2 * 2x8 using Strassen on the 2x2 part?
    # A0 = [[a,b],[c,d]], B0 = [[B00, B01, ...], [B10, B11, ...]] where B0j are 1x8 vectors.
    # A0 B0 = [[a B00 + b B10, ...], [c B00 + d B10, ...]]
    # Each entry of the result is a linear combination of entries of A0 and B0.
    # The tensor for 2x2 * 2x8 has size 2x2 x 2x8.
    # We can use the rank-7 Strassen decomposition for the 2x2 core, applied to each column of B?
    # Strassen computes 2x2 * 2x2 in 7 mults.
    # If B is 2x8, it's 4 independent 2x2 * 2x2 multiplications (columns 0-1, 2-3, etc).
    # So rank would be 7 * 4 = 28 for A0B0.
    # Similarly for A1B1: rank 28.
    # Total rank 56.
    
    # This matches the inspiration score 1.1429 (64/56).
    # Can we do better?
    # Maybe we can share the Strassen intermediates between A0B0 and A1B1?
    # A0 and A1 are different. B0 and B1 are different.
    # Unlikely to share directly without extra overhead.
    
    # Let's construct the rank 56 solution.
    
    def strassen_2x2(U, V, W):
        # Returns U, V, W factors for 2x2 matrix multiplication (rank 7)
        # Indices: U[i,j,t], V[j,k,t], W[k,i,t]
        # i,j,k in 0..1
        # t in 0..6
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # We need to map these to the tensor decomposition form.
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6
        
        # U factors (coeffs of A):
        # M1: A00, A11 -> U[0,0,0]=1, U[1,1,0]=1
        # M2: A10, A11 -> U[1,0,1]=1, U[1,1,1]=1
        # M3: A00 -> U[0,0,2]=1
        # M4: A11 -> U[1,1,3]=1
        # M5: A00, A01 -> U[0,0,4]=1, U[0,1,4]=1
        # M6: A10, A00 -> U[1,0,5]=1, U[0,0,5]=-1
        # M7: A01, A11 -> U[0,1,6]=1, U[1,1,6]=-1
        
        # V factors (coeffs of B):
        # M1: B00, B11 -> V[0,0,0]=1, V[1,1,0]=1
        # M2: B00 -> V[0,0,1]=1
        # M3: B10, B11 -> V[1,0,2]=1, V[1,1,2]=-1
        # M4: B01, B00 -> V[0,1,3]=1, V[0,0,3]=-1
        # M5: B11 -> V[1,1,4]=1
        # M6: B00, B10 -> V[0,0,5]=1, V[1,0,5]=1
        # M7: B01, B11 -> V[0,1,6]=1, V[1,1,6]=1
        
        # W factors (coeffs of C):
        # M1: C00, C11 -> W[0,0,0]=1, W[1,1,0]=1
        # M2: C10 -> W[1,0,1]=1
        # M3: C01, C11 -> W[0,1,2]=1, W[1,1,2]=-1
        # M4: C00, C10 -> W[0,0,3]=-1, W[1,0,3]=1
        # M5: C01 -> W[0,1,4]=1
        # M6: C11 -> W[1,1,5]=1
        # M7: C00 -> W[0,0,6]=1
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Flatten indices: i*2+j, j*2+k, k*2+i
        
        # M1
        U_s[0*2+0, 0] = 1; U_s[1*2+1, 0] = 1
        V_s[0*2+0, 0] = 1; V_s[1*2+1, 0] = 1
        W_s[0*2+0, 0] = 1; W_s[1*2+1, 0] = 1
        
        # M2
        U_s[1*2+0, 1] = 1; U_s[1*2+1, 1] = 1
        V_s[0*2+0, 1] = 1
        W_s[1*2+0, 1] = 1
        
        # M3
        U_s[0*2+0, 2] = 1
        V_s[1*2+0, 2] = 1; V_s[1*2+1, 2] = -1
        W_s[0*2+1, 2] = 1; W_s[1*2+1, 2] = -1
        
        # M4
        U_s[1*2+1, 3] = 1
        V_s[0*2+1, 3] = 1; V_s[0*2+0, 3] = -1
        W_s[0*2+0, 3] = -1; W_s[1*2+0, 3] = 1
        
        # M5
        U_s[0*2+0, 4] = 1; U_s[0*2+1, 4] = 1
        V_s[1*2+1, 4] = 1
        W_s[0*2+1, 4] = 1
        
        # M6
        U_s[1*2+0, 5] = 1; U_s[0*2+0, 5] = -1
        V_s[0*2+0, 5] = 1; V_s[1*2+0, 5] = 1
        W_s[1*2+1, 5] = 1
        
        # M7
        U_s[0*2+1, 6] = 1; U_s[1*2+1, 6] = -1
        V_s[0*2+1, 6] = 1; V_s[1*2+1, 6] = 1
        W_s[0*2+0, 6] = 1
        
        return U_s, V_s, W_s

    U_s, V_s, W_s = strassen_2x2(None, None, None)
    
    # Now build the 2x4 * 4x8 tensor.
    # A is 2x4. Split into A0 (2x2, cols 0-1) and A1 (2x2, cols 2-3).
    # B is 4x8. Split into B0 (2x8, rows 0-1) and B1 (2x8, rows 2-3).
    # C = A0 B0 + A1 B1.
    
    # We compute A0 B0 using Strassen on the 2x2 A0 and "Strassen-like" on the 2x8 B0?
    # No, B0 is 2x8. We can view B0 as 4 columns of 2x2 matrices? No, B0 is 2 rows, 8 cols.
    # It is 4 blocks of 2x2 matrices if we group columns (0,1), (2,3), etc.
    # Let B0 = [B0_0, B0_1, B0_2, B0_3] where each B0_k is 2x2.
    # Then A0 B0 = [A0 B0_0, A0 B0_1, A0 B0_2, A0 B0_3].
    # Each A0 B0_k is a 2x2 * 2x2 multiplication.
    # We can use the same Strassen decomposition for each block k.
    # The U factors are the same for all k (depend only on A0).
    # The V factors are different for each k (depend on B0_k).
    # The W factors are different for each k (depend on output position).
    
    # Total rank for A0 B0: 4 blocks * 7 rank = 28.
    # Total rank for A1 B1: 4 blocks * 7 rank = 28.
    # Total rank = 56.
    
    best_rank = 56
    best_U = np.zeros((U_dim, best_rank))
    best_V = np.zeros((V_dim, best_rank))
    best_W = np.zeros((W_dim, best_rank))
    
    # Indices mapping:
    # A: i in 0..1, j in 0..3.
    # A0: j in 0..1. A1: j in 2..3.
    # B: j in 0..3, k in 0..7.
    # B0: j in 0..1. B1: j in 2..3.
    # C: i in 0..1, k in 0..7.
    
    # We iterate over the 4 blocks of B (cols 0-1, 2-3, 4-5, 6-7).
    # Let block index b in 0..3.
    # B block b corresponds to columns k = 2*b, 2*b+1.
    # B0 block b: rows 0,1.
    # B1 block b: rows 2,3.
    
    # For each block b, we have two products: A0 * B0_block and A1 * B1_block.
    # We use the Strassen factors U_s, V_s, W_s.
    
    for b in range(4):
        # Offset for this block in the rank dimension
        # We allocate 14 ranks per block (7 for A0, 7 for A1)
        base_rank = b * 14
        
        # --- A0 * B0_block ---
        # A0 indices: i in 0..1, j in 0..1.
        # B0_block indices: j in 0..1 (local), k in 2*b, 2*b+1.
        # C indices: i in 0..1, k in 2*b, 2*b+1.
        
        for t in range(7):
            r = base_rank + t
            
            # U factors
            # U_s is 4x7 (i*2+j, t).
            # Map to best_U (i*4+j, r).
            # A0 j_local is 0,1. Global j is 0,1.
            for i in range(2):
                for j_local in range(2):
                    val = U_s[i*2+j_local, t]
                    if val != 0:
                        best_U[i*4 + j_local, r] += val
            
            # V factors
            # V_s is 4x7 (j*2+k_local, t).
            # Map to best_V (j*8+k, r).
            # B0 j_local is 0,1. Global j is 0,1.
            # B0 k_local is 0,1. Global k is 2*b, 2*b+1.
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local*2+k_local, t]
                    if val != 0:
                        g_j = j_local
                        g_k = 2*b + k_local
                        best_V[g_j*8 + g_k, r] += val
            
            # W factors
            # W_s is 4x7 (k*2+i, t). Note: W index in Strassen was k*2+i?
            # In my strassen_2x2, W_s index was k*2+i.
            # C indices: i in 0..1, k in 2*b, 2*b+1.
            # Trace convention: W[k*n+i, r] = W[k*2+i, r].
            for k_local in range(2):
                for i in range(2):
                    val = W_s[k_local*2+i, t]
                    if val != 0:
                        g_k = 2*b + k_local
                        best_W[g_k*2 + i, r] += val
        
        # --- A1 * B1_block ---
        # A1 indices: i in 0..1, j in 2..3.
        # B1_block indices: j in 2..3 (local 0,1), k in 2*b, 2*b+1.
        # C indices: i in 0..1, k in 2*b, 2*b+1.
        
        for t in range(7):
            r = base_rank + 7 + t
            
            # U factors
            # A1 j_local 0,1 -> Global j 2,3.
            for i in range(2):
                for j_local in range(2):
                    val = U_s[i*2+j_local, t]
                    if val != 0:
                        g_j = 2 + j_local
                        best_U[i*4 + g_j, r] += val
            
            # V factors
            # B1 j_local 0,1 -> Global j 2,3.
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local*2+k_local, t]
                    if val != 0:
                        g_j = 2 + j_local
                        g_k = 2*b + k_local
                        best_V[g_j*8 + g_k, r] += val
            
            # W factors
            # Same C indices as before.
            for k_local in range(2):
                for i in range(2):
                    val = W_s[k_local*2+i, t]
                    if val != 0:
                        g_k = 2*b + k_local
                        best_W[g_k*2 + i, r] += val
    
    # Verify
    def verify_decomp(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook if verification fails
        best_rank = n * m * p
        best_U = np.zeros((U_dim, best_rank))
        best_V = np.zeros((V_dim, best_rank))
        best_W = np.zeros((W_dim, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
