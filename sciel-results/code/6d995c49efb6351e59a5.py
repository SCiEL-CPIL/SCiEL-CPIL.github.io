# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's rank-7 decomposition
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices:
    # U row i*m+j: A[i,j]
    # V row j*p+k: B[j,k]
    # W row k*n+i: C[i,k] (Note: Task says W[k*n+i, r] is the output factor. 
    # The tensor decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This matches the standard definition where W corresponds to the output C[i,k] but indexed as k*n+i.
    # Let's verify the indexing in the current code.
    # Current code:
    # W row index: k*n+i -> C00: k=0,i=0 -> 0; C01: k=1,i=0 -> 2; C10: k=0,i=1 -> 1; C11: k=1,i=1 -> 3
    # This means W[0] corresponds to C00, W[1] to C10, W[2] to C01, W[3] to C11.
    # Let's check the Strassen coefficients in the current code against this.
    # M1 = (A00+A11)(B00+B11)
    # C00 gets +1, C11 gets +1.
    # Current code M1 W: {0: 1, 1: 0, 2: 0, 3: 1}.
    # W[0]=1 (C00), W[3]=1 (C11). Correct.
    # M2 = (A10+A11)B00
    # C10 gets +1, C11 gets -1.
    # Current code M2 W: {0: 0, 1: 1, 2: 0, 3: -1}.
    # W[1]=1 (C10), W[3]=-1 (C11). Correct.
    
    # The current code is correct. I will just rewrite it to be a direct list assignment for clarity and speed.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
