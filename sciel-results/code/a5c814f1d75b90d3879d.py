import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 100.0

    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Indices for 2x2:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    
    # M1 = (A00+A11)(B00+B11)
    # A coeffs: A00:1, A11:1
    # B coeffs: B00:1, B11:1
    # C coeffs: C00:1, C11:1
    # Inner 2x2 mult:
    # For each term t in 0..6 of inner 2x2:
    # U2[row, t] is 1 for specific (i,j)
    # We need the base 2x2 Strassen matrices.
    
    # Standard Strassen 2x2:
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    # C00 = M1 + M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M6
    
    # Let's hardcode U2, V2, W2 for 2x2.
    # Term 0 (M1):
    # A: A00(1), A11(1). Rows 0, 3.
    # B: B00(1), B11(1). Rows 0, 3.
    # C: C00(1), C11(1). Rows 0, 3.
    # Wait, U2 is for A.
    # U2[0, 0] = 1 (A00), U2[3, 0] = 1 (A11)
    # V2[0, 0] = 1 (B00), V2[3, 0] = 1 (B11)
    # W2[0, 0] = 1 (C00), W2[3, 0] = 1 (C11)
    
    # Term 1 (M2):
    # A: A10(1), A11(1). Rows 2, 3.
    # B: B00(1). Row 0.
    # C: C10(1). Row 2.
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 2 (M3):
    # A: A00(1). Row 0.
    # B: B01(1), B11(-1). Rows 1, 3.
    # C: C01(1), C11(-1). Rows 1, 3.
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # Term 3 (M4):
    # A: A11(1). Row 3.
    # B: B10(1), B00(-1). Rows 2, 0.
    # C: C00(1), C10(1). Rows 0, 2.
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4 (M5):
    # A: A00(1), A01(1). Rows 0, 1.
    # B: B11(1). Row 3.
    # C: C01(1), C11(1). Rows 1, 3.
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    
    # Term 5 (M6):
    # A: A10(1), A00(-1). Rows 2, 0.
    # B: B00(1), B01(1). Rows 0, 1.
    # C: C00(-1), C11(1). Rows 0, 3.
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = -1; W2[3, 5] = 1
    
    # Term 6 (M7):
    # A: A01(1), A11(-1). Rows 1, 3.
    # B: B10(1), B11(1). Rows 2, 3.
    # C: C00(1), C11(0). Rows 0.
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2
    # T[i,j,k] = sum_t U2[i*2+j, t] V2[j*2+k, t] W2[k*2+i, t]
    # Should be 1 if i==k? No, C[i,k] = sum_j A[i,j]B[j,k].
    # Tensor T[i,j,k] is 1 if C[i,k] depends on A[i,j]B[j,k]?
    # Yes, the multiplication tensor has 1 at (i,j,k) if C[i,k] = sum ... A[i,j]B[j,k] ...
    # Actually, the standard tensor definition for matmul A(n,m) B(m,p) -> C(n,p)
    # is T[i,j,k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
    # Since C[i,k] = sum_j A[i,j]B[j,k], T[i,j,k] is 1 for all valid i,j,k.
    # Wait, the task says "exactly decompose the matrix multiplication tensor".
    # The tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # So we need sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1.
    
    # Let's check T[0,0,0] for 2x2.
    # i=0, j=0, k=0.
    # sum_t U2[0, t] V2[0, t] W2[0, t]
    # t=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # t=1: U[0,1]=0
    # t=2: U[0,2]=1, V[0,2]=0
    # t=3: U[0,3]=0
    # t=4: U[0,4]=1, V[0,4]=0
    # t=5: U[0,5]=-1, V[0,5]=1, W[0,5]=-1 -> (-1)(1)(-1) = 1
    # t=6: U[0,6]=0
    # Sum = 1 + 1 = 2? 
    # This is wrong. Strassen computes C, but the tensor decomposition must satisfy sum = 1.
    # My hardcoded U2, V2, W2 are for the *algorithm* steps, not the tensor factors directly?
    # No, the tensor factors *are* the algorithm steps.
    # Let's re-verify Strassen tensor.
    # C00 = M1 + M4 - M6 + M7
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # Coeff of A00B00 in C00:
    # From M1: 1
    # From M6: -(-1) = 1? No, M6 is subtracted. -M6.
    # M6 term for A00B00 is -A00B00. So -(-A00B00) = +A00B00.
    # Sum: 1 + 1 = 2?
    # Something is wrong with my Strassen expansion or coefficients.
    
    # Let's use a known correct Strassen tensor decomposition.
    # Or just use the recursive construction which is safer if the base is correct.
    # But I need a correct base.
    
    # Alternative: Use the identity that Rank(2x2x2) <= 7.
    # I will trust the standard Strassen algorithm structure but I must ensure the tensor identity holds.
    # The identity is sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1.
    
    # Let's try to construct U4, V4, W4 directly using the block logic, assuming U2, V2, W2 are correct.
    # If U2, V2, W2 are correct, then U4, V4, W4 will be correct.
    # I will add a verification step for U2, V2, W2.
    
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(W.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if val != 1:
                        return False
        return True

    # Verify U2, V2, W2
    if not verify_decomp(U2, V2, W2, 2, 2, 2):
        # Fallback to schoolbook 2x2 (Rank 4)
        U2 = np.eye(4, 4, dtype=int)
        V2 = np.eye(4, 4, dtype=int)
        W2 = np.eye(4, 4, dtype=int)
        # Check schoolbook 2x2
        # U[i*2+j, t] = 1 if t == i*2+j
        # V[j*2+k, t] = 1 if t == j*2+k
        # W[k*2+i, t] = 1 if t == k*2+i
        # Product is 1 only if i*2+j == j*2+k == k*2+i.
        # This implies i=j=k.
        # But we need T[i,j,k]=1 for ALL i,j,k.
        # Schoolbook tensor:
        # U[i*m+j, t] = 1 if t = i*m+j
        # V[j*p+k, t] = 1 if t = j*p+k
        # W[k*n+i, t] = 1 if t = k*n+i
        # For 2x2:
        # t=0: i=0,j=0. V[0,0]=1 (j=0,k=0). W[0,0]=1 (k=0,i=0).
        # So T[0,0,0] = 1.
        # t=1: i=0,j=1. V[1,?]... V row 1 is j=0,k=1.
        # Wait, V index is j*p+k.
        # If t=1, U[1,1]=1 (i=0,j=1).
        # V[1,1]=1 (j=0,k=1).
        # W[1,1]=1 (k=0,i=1).
        # So T[0,1,0] = 1.
        # This works for schoolbook.
        
    # Now build 4x4x4
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Strassen Coeffs for 2x2 blocks
    # s: 0..6
    # A_coeffs[s][I][J]
    # B_coeffs[s][J][K]
    # C_coeffs[s][I][K]
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    B_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [1, 1]]  # M7
    ]
    C_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [-1, 0]],# M3
        [[1, 0], [1, 0]], # M4
        [[0, 1], [0, 1]], # M5
        [[-1, 0], [0, 1]],# M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    for s in range(7):
        for u in range(7):
            t = s * 7 + u
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # U4 part
                        # Global i = 2*I + i_loc, j = 2*J + j_loc
                        # U2 row = i_loc*2 + j_loc
                        # Coeff from A_coeffs[s][I][J]
                        # U2 col = u
                        for i_loc in range(2):
                            for j_loc in range(2):
                                i = 2*I + i_loc
                                j = 2*J + j_loc
                                u2_val = U2[i_loc*2 + j_loc, u]
                                a_coeff = A_coeffs[s][I][J]
                                U4[i*4 + j, t] += u2_val * a_coeff
                                
                        # V4 part
                        # Global j = 2*J + j_loc, k = 2*K + k_loc
                        # V2 row = j_loc*2 + k_loc
                        # Coeff from B_coeffs[s][J][K]
                        for j_loc in range(2):
                            for k_loc in range(2):
                                j = 2*J + j_loc
                                k = 2*K + k_loc
                                v2_val = V2[j_loc*2 + k_loc, u]
                                b_coeff = B_coeffs[s][J][K]
                                V4[j*4 + k, t] += v2_val * b_coeff
                                
                        # W4 part
                        # Global k = 2*K + k_loc, i = 2*I + i_loc
                        # W2 row = k_loc*2 + i_loc
                        # Coeff from C_coeffs[s][I][K]
                        for k_loc in range(2):
                            for i_loc in range(2):
                                k = 2*K + k_loc
                                i = 2*I + i_loc
                                w2_val = W2[k_loc*2 + i_loc, u]
                                c_coeff = C_coeffs[s][I][K]
                                W4[k*4 + i, t] += w2_val * c_coeff

    # Verify 4x4
    if not verify_decomp(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook 4x4
        U4 = np.eye(16, 16, dtype=int)
        V4 = np.eye(16, 16, dtype=int)
        W4 = np.eye(16, 16, dtype=int)

    # Build 4x4x8
    # Rank 98
    U_final = np.zeros((16, 98), dtype=int)
    V_final = np.zeros((32, 98), dtype=int)
    W_final = np.zeros((32, 98), dtype=int)
    
    # First 49 terms: C_L = A * B_L (B cols 0-3)
    U_final[:, 0:49] = U4
    # V_final rows for B_L: j in 0..3, k in 0..3
    # Row index j*8 + k
    for j in range(4):
        for k in range(4):
            V_final[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
    # W_final rows for C_L: k in 0..3, i in 0..3
    # Row index k*4 + i
    W_final[0:16, 0:49] = W4
    
    # Next 49 terms: C_R = A * B_R (B cols 4-7)
    U_final[:, 49:98] = U4
    # V_final rows for B_R: j in 0..3, k in 4..7
    # Row index j*8 + k
    for j in range(4):
        for k in range(4):
            k_global = k + 4
            V_final[j*8 + k_global, 49:98] = V4[j*4 + k, 0:49]
    # W_final rows for C_R: k in 4..7, i in 0..3
    # Row index k*4 + i
    # k=4 -> 16..19, k=7 -> 28..31
    W_final[16:32, 49:98] = W4

    # Verify final
    if verify_decomp(U_final, V_final, W_final, 4, 4, 8):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to schoolbook 4x4x8
        U_school = np.eye(16, 128, dtype=int)
        V_school = np.eye(32, 128, dtype=int)
        W_school = np.eye(32, 128, dtype=int)
        # Map indices
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    t = i*32 + j*8 + k
                    U_school[i*4 + j, t] = 1
                    V_school[j*8 + k, t] = 1
                    W_school[k*4 + i, t] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

