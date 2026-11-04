import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 17 is the known minimal rank for 2x3x3 matrix multiplication.
    # We provide a deterministic, exact decomposition.
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices mapping:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Helper to set values
    def set_U(i, j, r, val): U[i * m + j, r] = val
    def set_V(j, k, r, val): V[j * p + k, r] = val
    def set_W(k, i, r, val): W[k * n + i, r] = val

    # Strassen-like decomposition for the 2x2x2 core (rows 0,1 of A; cols 0,1 of B; rows 0,1 of C)
    # This uses 7 multiplications.
    
    # M0: (A00 + A11)(B00 + B11)
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M1: (A10 + A11)B00
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    # M2: A00(B01 - B11)
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    # M3: A11(B10 - B00)
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    # M4: (A00 + A01)B11
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    # M5: (A10 - A00)(B00 + B01)
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    # M6: (A11 - A01)(B10 + B11)
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # Now we need to handle the 3rd column of B (k=2) and 3rd row of A (j=2).
    # The tensor is 2x3x3.
    # We have covered C[0:2, 0:2] using 7 terms.
    # We need C[0:2, 2] and C[?, 0:2] is done. Wait, C is 2x3.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The 7 terms above compute C[0,0], C[0,1], C[1,0], C[1,1] correctly.
    # We need to add terms for:
    # 1. C[0, 2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # 2. C[1, 2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # However, the Strassen terms above might have introduced cross terms or we can just add direct terms for the remaining dependencies.
    # Actually, the standard rank 17 construction for 2x3x3 is:
    # Use 7 for the 2x2x2 part.
    # The remaining part involves the 3rd column of B and 3rd row of A.
    # It turns out we can share some multiplications or just add 10 more? No, 7+10=17.
    
    # Let's look at the current program's hardcoded values for indices 7-16.
    # It seems to use direct terms for the remaining parts.
    
    # Term 7: A[0,0] * B[0,2] -> contributes to C[0,2]
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # Term 8: A[0,1] * B[1,2] -> contributes to C[0,2]
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # Term 9: A[1,0] * B[0,2] -> contributes to C[1,2]
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # Term 10: A[1,1] * B[1,2] -> contributes to C[1,2]
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)
    
    # Term 11: A[0,2] * B[2,0] -> contributes to C[0,0]
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # Term 12: A[0,2] * B[2,1] -> contributes to C[0,1]
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    # Term 13: A[0,2] * B[2,2] -> contributes to C[0,2]
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    # Term 14: A[1,2] * B[2,0] -> contributes to C[1,0]
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    # Term 15: A[1,2] * B[2,1] -> contributes to C[1,1]
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    # Term 16: A[1,2] * B[2,2] -> contributes to C[1,2]
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
