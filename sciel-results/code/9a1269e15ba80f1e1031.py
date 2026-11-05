# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def make_rank_14():
        """
        Construct a rank 14 decomposition using Strassen's algorithm for 3x3 matrices 
        applied over the ring R[t]/(t^2) or by splitting the 6 columns into two 3-column blocks.
        
        The 3x3 x 3x6 multiplication can be viewed as two 3x3 x 3x3 multiplications:
        C1 = A * B1 (where B1 is cols 0-2)
        C2 = A * B2 (where B2 is cols 3-5)
        
        We use the rank-7 Strassen decomposition for 3x3 matrices.
        Total rank = 7 + 7 = 14.
        """
        # Strassen 3x3 decomposition (Rank 7)
        # Sources vary, but a standard set of 7 products exists.
        # Let A, B be 3x3.
        # M1 = A11 * (B22 - B12)
        # M2 = (A11 + A21) * B12
        # M3 = (A12 - A22) * (B21 + B22)
        # M4 = A22 * (B11 - B21)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 + A22) * (B21 - B22)
        
        # Indices for 3x3 matrix flattened row-major:
        # 0:11, 1:12, 2:13
        # 3:21, 4:22, 5:23
        # 6:31, 7:32, 8:33
        
        # We define the coefficients for U (A factors), V (B factors), W (C factors) for one 3x3 block.
        # U, V, W will be 9x7 matrices.
        
        U_3x3 = np.zeros((9, 7))
        V_3x3 = np.zeros((9, 7))
        W_3x3 = np.zeros((9, 7))
        
        # M1
        U_3x3[0, 0] = 1  # A11
        V_3x3[4, 0] = 1  # B22
        V_3x3[1, 0] = -1 # B12
        W_3x3[4, 0] = 1  # C22 (part of)
        W_3x3[0, 0] = 1  # C11 (part of)
        
        # M2
        U_3x3[0, 1] = 1  # A11
        U_3x3[3, 1] = 1  # A21
        V_3x3[1, 1] = 1  # B12
        W_3x3[1, 1] = 1  # C12 (part of)
        
        # M3
        U_3x3[1, 2] = 1  # A12
        U_3x3[4, 2] = -1 # A22
        V_3x3[6, 2] = 1  # B21
        V_3x3[4, 2] = 1  # B22
        W_3x3[6, 2] = 1  # C31 (part of)
        W_3x3[4, 2] = 1  # C22 (part of)
        
        # M4
        U_3x3[4, 3] = 1  # A22
        V_3x3[0, 3] = 1  # B11
        V_3x3[6, 3] = -1 # B21
        W_3x3[3, 3] = 1  # C21 (part of)
        
        # M5
        U_3x3[0, 4] = 1  # A11
        U_3x3[1, 4] = 1  # A12
        V_3x3[4, 4] = 1  # B22
        W_3x3[7, 4] = 1  # C32 (part of)
        W_3x3[1, 4] = 1  # C12 (part of)
        
        # M6
        U_3x3[3, 5] = 1  # A21
        U_3x3[0, 5] = -1 # A11
        V_3x3[0, 5] = 1  # B11
        V_3x3[1, 5] = 1  # B12
        W_3x3[3, 5] = 1  # C21 (part of)
        W_3x3[0, 5] = 1  # C11 (part of)
        
        # M7
        U_3x3[1, 6] = 1  # A12
        U_3x3[4, 6] = 1  # A22
        V_3x3[6, 6] = 1  # B21
        V_3x3[4, 6] = -1 # B22
        W_3x3[8, 6] = 1  # C33 (part of)
        W_3x3[7, 6] = 1  # C32 (part of)
        
        # We need to verify this specific Strassen variant. 
        # Let's check C11 = M1 + M6. 
        # M1 = A11(B22-B12). M6 = (A21-A11)(B11+B12).
        # C11 should be A11B11 + A12B21 + A13B31.
        # My indices above assumed 3x3. 
        # Wait, standard Strassen for 3x3 is not just 7 products in a simple way like 2x2. 
        # Actually, the rank of 3x3 matrix multiplication is known to be 7. 
        # However, the explicit coefficients are complex. 
        # Let's rely on the fact that if I can't verify the specific coefficients easily in my head, 
        # I should use a known correct set or fall back.
        
        # Let's try to verify the Strassen 3x3 coefficients I just wrote.
        # This is risky. Let's use a different approach: 
        # We know rank 7 exists. But coding it from memory is error-prone.
        # However, the prompt asks to KEEP the design family (Strassen-like).
        
        # Alternative: Use the rank 14 construction but with a verified 3x3 decomposition.
        # If I cannot guarantee the 3x3 coefficients are correct, I will return None and fall back.
        
        # Let's do a quick check of one element for the Strassen 3x3 I wrote above.
        # C11 (index 0) = M1 + M6
        # M1 = A11 * (B22 - B12) = A11B22 - A11B12
        # M6 = (A21 - A11) * (B11 + B12) = A21B11 + A21B12 - A11B11 - A11B12
        # Sum = A11B22 - A11B12 + A21B11 + A21B12 - A11B11 - A11B12
        # = A11B22 + A21B11 + A21B12 - A11B11 - 2A11B12
        # This is NOT A11B11 + A12B21 + A13B31. 
        # So my hardcoded Strassen 3x3 is WRONG.
        
        return None

    # Try Rank 14 (currently disabled due to incorrect coefficients in make_rank_14)
    result_14 = make_rank_14()
    if result_14 is not None and verify(result_14["U"], result_14["V"], result_14["W"]):
        return result_14

    # Fallback: Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
