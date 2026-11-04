import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    
    # --- Strassen 2x2x2 decomposition ---
    # U2[i*2+j, r] = coeff of A[i,j] in term r
    # V2[j*2+k, r] = coeff of B[j,k] in term r
    # W2[k*2+i, r] = coeff of C[i,k] in term r  (trace convention: W[k*n+i, r])
    
    U2 = np.array([
        # A00, A01, A10, A11
        [ 1,  0,  0,  0],  # M1: A00
        [ 0,  0,  0,  0],  # M2: (A10+A11) -> wait, let me redo this properly
    ], dtype=float)
    
    # Let me define Strassen terms explicitly:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U2 shape (4, 7): rows are (i,j) for A, cols are terms 1..7
    U2 = np.zeros((4, 7))
    # M1: A00 + A11
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11
    # M2: A10 + A11
    U2[2, 1] = 1  # A10
    U2[3, 1] = 1  # A11
    # M3: A00
    U2[0, 2] = 1  # A00
    # M4: A11
    U2[3, 3] = 1  # A11
    # M5: A00 + A01
    U2[0, 4] = 1  # A00
    U2[1, 4] = 1  # A01
    # M6: A10 - A00
    U2[2, 5] = 1  # A10
    U2[0, 5] = -1 # A00
    # M7: A01 - A11
    U2[1, 6] = 1  # A01
    U2[3, 6] = -1 # A11
    
    # V2 shape (4, 7): rows are (j,k) for B, cols are terms 1..7
    V2 = np.zeros((4, 7))
    # M1: B00 + B11
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11
    # M2: B00
    V2[0, 1] = 1  # B00
    # M3: B01 - B11
    V2[1, 2] = 1  # B01
    V2[3, 2] = -1 # B11
    # M4: B10 - B00
    V2[2, 3] = 1  # B10
    V2[0, 3] = -1 # B00
    # M5: B11
    V2[3, 4] = 1  # B11
    # M6: B00 + B01
    V2[0, 5] = 1  # B00
    V2[1, 5] = 1  # B01
    # M7: B10 + B11
    V2[2, 6] = 1  # B10
    V2[3, 6] = 1  # B11
    
    # W2 shape (4, 7): rows are (k,i) for C (trace convention W[k*n+i]), cols are terms 1..7
    # C00 = M1 + M4 - M5 + M7  -> W[0*2+0, r] = W[0, r]
    # C01 = M3 + M5             -> W[1*2+0, r] = W[2, r]
    # C10 = M2 + M4             -> W[0*2+1, r] = W[1, r]
    # C11 = M1 - M2 + M3 + M6   -> W[1*2+1, r] = W[3, r]
    W2 = np.zeros((4, 7))
    # C00 (index 0): M1 + M4 - M5 + M7
    W2[0, 0] = 1   # M1
    W2[0, 3] = 1   # M4
    W2[0, 4] = -1  # M5
    W2[0, 6] = 1   # M7
    # C01 (index 2): M3 + M5
    W2[2, 2] = 1   # M3
    W2[2, 4] = 1   # M5
    # C10 (index 1): M2 + M4
    W2[1, 1] = 1   # M2
    W2[1, 3] = 1   # M4
    # C11 (index 3): M1 - M2 + M3 + M6
    W2[3, 0] = 1   # M1
    W2[3, 1] = -1  # M2
    W2[3, 2] = 1   # M3
    W2[3, 5] = 1   # M6
    
    # Verify 2x2 Strassen
    # T[i,j,k] = sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r]
    # For 2x2: i,j,k in {0,1}
    # Check T[0,0,0] should be 1 (A00 * B00 * C00)
    # Actually T[i*m+j, j*p+k, k*n+i] = 1 for matrix mult
    # For 2x2: T[i*2+j, j*2+k, k*2+i] = 1
    # T[0,0,0]: i=0,j=0,k=0 -> U2[0,0]*V2[0,0]*W2[0,0] = 1*1*1 = 1 ✓
    # T[0,0,1]: i=0,j=0,k=1 -> U2[0,0]*V2[1,0]*W2[2,0] = 1*0*0 = 0... 
    # Wait, T[i*2+j, j*2+k, k*2+i]: for i=0,j=0,k=1: T[0, 1, 2]
    # That's A00 * B01 * C01. U2[0,0]*V2[1,0]*W2[2,0] = 1*0*0 = 0. But it should be 1!
    # Hmm, let me recheck. T[a,b,c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for some i,j,k.
    # For i=0,j=0,k=0: a=0, b=0, c=0. T[0,0,0]=1.
    # For i=0,j=0,k=1: a=0, b=1, c=2. T[0,1,2]=1.
    # The decomposition: T[a,b,c] = sum_r U2[a,r]*V2[b,r]*W2[c,r]
    # T[0,1,2] = sum_r U2[0,r]*V2[1,r]*W2[2,r]
    # r=0: U2[0,0]*V2[1,0]*W2[2,0] = 1*0*0 = 0
    # r=1: U2[0,1]*V2[1,1]*W2[2,1] = 0*0*0 = 0
    # r=2: U2[0,2]*V2[1,2]*W2[2,2] = 1*1*1 = 1
    # r=3: U2[0,3]*V2[1,3]*W2[2,3] = 0*0*0 = 0
    # r=4: U2[0,4]*V2[1,4]*W2[2,4] = 1*0*1 = 0
    # r=5: U2[0,5]*V2[1,5]*W2[2,5] = -1*1*0 = 0
    # r=6: U2[0,6]*V2[1,6]*W2[2,6] = 0*0*0 = 0
    # Sum = 1. ✓
    
    # Now build 4x4x4 via Kronecker product
    # 4x4 matrices as 2x2 blocks of 2x2 submatrices.
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # A[i,j] where i=2*i1+i2, j=2*j1+j2
    # B[j,k] where j=2*j1+j2, k=2*k1+k2
    # C[i,k] where i=2*i1+i2, k=2*k1+k2
    #
    # Kronecker: U4[i*4+j, r1*7+r2] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]  -- NO
    # The Kronecker product structure for tensor decomposition:
    # If T_2 has rank 7 with factors U2, V2, W2, then T_4 (4x4x4) has rank 49
    # with factors that are Kronecker products, but we need to be careful about indexing.
    #
    # For 4x4: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # A[i,j] = A[2*i1+i2, 2*j1+j2]
    # The "outer" indices are (i1, j1) and "inner" are (i2, j2)
    # U2 for outer: U2[i1*2+j1, r1] -- but U2 is for 2x2, so row index is i1*2+j1
    # Wait, U2 has shape (4,7) where row = i*2+j for 2x2.
    # For 4x4, the outer 2x2 block structure:
    # U4[row, r1*7+r2] where row = i*4+j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # = (i1*2 + j1)*4 + (i2*2 + j2)
    # So U4[row, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # Similarly for V and W.
    
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for r1 in range(7):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    # U4 index: row = i*4+j = (2*i1+i2)*4 + (2*j1+j2)
                                    row_u = (2*i1 + i2) * 4 + (2*j1 + j2)
                                    U4[row_u, r] = U2[i1*2 + j1, r1] * U2[i2*2 + j2, r2]
                                    
                                    # V4 index: row = j*4+k = (2*j1+j2)*4 + (2*k1+k2)
                                    row_v = (2*j1 + j2) * 4 + (2*k1 + k2)
                                    V4[row_v, r] = V2[j1*2 + k1, r1] * V2[j2*2 + k2, r2]
                                    
                                    # W4 index: row = k*4+i = (2*k1+k2)*4 + (2*i1+i2)
                                    row_w = (2*k1 + k2) * 4 + (2*i1 + i2)
                                    W4[row_w, r] = W2[k1*2 + i1, r1] * W2[k2*2 + i2, r2]
    
    # Verify the decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U4, V4, W4)
    
    # Build the true tensor
    T_true = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T_true[i*4+j, j*4+k, k*4+i] = 1.0
    
    max_diff = np.max(np.abs(T_decomp - T_true))
    
    if max_diff < 1e-9:
        # Valid decomposition
        if time.time() - start_time > 110:
            # Fallback (shouldn't happen, but just in case)
            n, m, p = 4, 4, 4
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col] = 1
                        V_fb[j * p + k][col] = 1
                        W_fb[k * n + i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        
        return {
            "U": U4.tolist(),
            "V": V4.tolist(),
            "W": W4.tolist()
        }
    else:
        # Fallback: naive rank-64 decomposition
        n, m, p = 4, 4, 4
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
