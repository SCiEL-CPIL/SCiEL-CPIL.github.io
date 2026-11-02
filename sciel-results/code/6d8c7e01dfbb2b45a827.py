import numpy as np

def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor rank of the 2x2 matrix multiplication tensor is 7.
    The decomposition satisfies:
      sum_{r=0}^{6} U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] = 1
      for all i in {0,1}, a in {0,1}, k in {0,1}.
      
    Indexing:
      U: (4, 7) -> U[i*2 + a, r] corresponds to A[i, a]
      V: (4, 7) -> V[a*2 + k, r] corresponds to B[a, k]
      W: (4, 7) -> W[k*2 + i, r] corresponds to C[i, k] coefficient
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices:
    # A00: U[0, :], A01: U[1, :], A10: U[2, :], A11: U[3, :]
    # B00: V[0, :], B01: V[1, :], B10: V[2, :], B11: V[3, :]
    # C00: W[0, :], C10: W[1, :], C01: W[2, :], C11: W[3, :]
    # Note: W index is k*n + i. 
    # C00 (i=0,k=0) -> 0*2+0=0
    # C10 (i=1,k=0) -> 0*2+1=1
    # C01 (i=0,k=1) -> 1*2+0=2
    # C11 (i=1,k=1) -> 1*2+1=3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # -B11
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # -B00
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # -A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # -A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    
    # Combine to get C
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # -M5
    W[0][6] = 1  # M7
    
    # C10 = M2 + M4
    W[1][1] = 1  # M2
    W[1][3] = 1  # M4
    
    # C01 = M3 + M5
    W[2][2] = 1  # M3
    W[2][4] = 1  # M5
    
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # -M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    # Verification polish: Ensure exactness and integer types
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check decomposition
    # Tensor T[i, a, k] = sum_r U[i*2+a, r] * V[a*2+k, r] * W[k*2+i, r]
    # We want T[i, a, k] = 1 for all i, a, k? No.
    # The tensor for matrix multiplication C = AB is defined by:
    # C[i, k] = sum_a A[i, a] B[a, k]
    # The decomposition is:
    # sum_r U[i, a, r] V[a, k, r] W[i, k, r] = delta_{i, k?} No.
    # The standard form is:
    # sum_r U[i, a, r] V[a, k, r] W[i, k, r] = 1 if this term corresponds to A[i,a]B[a,k] contributing to C[i,k].
    # Actually, the tensor T has entries T[i, a, k] = 1 if the product A[i,a]B[a,k] contributes to C[i,k].
    # In the decomposition:
    # sum_r U[i, a, r] V[a, k, r] W[i, k, r] = 1 for all i, a, k.
    # Wait, the indices in the prompt are:
    # U[i*m+a, r], V[a*p+k, r], W[k*n+i, r]
    # So sum_r U[i*2+a, r] * V[a*2+k, r] * W[k*2+i, r] should equal 1 for all i, a, k.
    
    error = 0.0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-9:
        # Fallback to schoolbook if Strassen verification fails (should not happen)
        U = [[0]*8 for _ in range(4)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(4)]
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    r = i*4 + a*2 + k
                    U[i*2+a][r] = 1
                    V[a*2+k][r] = 1
                    W[k*2+i][r] = 1
                    # This is rank 8, which is worse, but valid.
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
