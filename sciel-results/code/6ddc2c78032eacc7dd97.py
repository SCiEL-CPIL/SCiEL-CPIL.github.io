"""Matrix multiplication 2x3x3 with rank-minimal decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    
    Strategy: Use Strassen's 7-multiplication algorithm for the 2x2x2 core,
    then handle the remaining 10 terms naively.
    
    W indexing convention: W[k*n + i, r] where n=2
    So C[i,k] maps to W[k*2 + i, r]
    C[0,0]->W[0], C[0,1]->W[1], C[0,2]->W[2]
    C[1,0]->W[3], C[1,1]->W[4], C[1,2]->W[5]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 6 rows
    
    # U indices: A[i,j] -> U[i*m + j] = U[i*3 + j]
    # V indices: B[j,k] -> V[j*p + k] = V[j*3 + k]
    # W indices: C[i,k] -> W[k*n + i] = W[k*2 + i]
    
    # --- Strassen part for A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2] ---
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[3], B[1,1]=V[4]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[3], C[1,1]=W[4]
    
    # M1 = A[0,0] * (B[0,1] + B[1,1])
    c = 0
    U[0][c] = 1
    V[1][c] = 1
    V[4][c] = 1
    # C[0,0] += M1
    W[0][c] = 1
    # C[1,1] += M1
    W[4][c] = 1
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    c = 1
    U[0][c] = 1
    U[3][c] = 1
    V[4][c] = 1
    # C[1,0] += M2
    W[3][c] = 1
    # C[1,1] -= M2
    W[4][c] = -1
    
    # M3 = (A[0,1] + A[1,1]) * B[0,0]
    c = 2
    U[1][c] = 1
    U[4][c] = 1
    V[0][c] = 1
    # C[0,1] += M3
    W[1][c] = 1
    # C[1,1] += M3
    W[4][c] = 1
    
    # M4 = A[1,1] * (B[0,0] - B[0,1])
    c = 3
    U[4][c] = 1
    V[0][c] = 1
    V[1][c] = -1
    # C[0,0] += M4
    W[0][c] = 1
    
    # M5 = (A[0,0] + A[0,1]) * (B[1,0] - B[1,1])
    c = 4
    U[0][c] = 1
    U[1][c] = 1
    V[3][c] = 1
    V[4][c] = -1
    # C[0,0] -= M5
    W[0][c] = -1
    # C[0,1] += M5
    W[1][c] = 1
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    c = 5
    U[3][c] = 1
    U[0][c] = -1
    V[0][c] = 1
    V[3][c] = 1
    # C[1,0] += M6
    W[3][c] = 1
    # C[1,1] += M6
    W[4][c] = 1
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[0,0])
    c = 6
    U[1][c] = 1
    U[4][c] = -1
    V[4][c] = 1
    V[0][c] = 1
    # C[0,0] += M7
    W[0][c] = 1
    
    # --- Naive part for remaining terms ---
    # Terms involving A[:,2] (column 2 of A) with B rows 0,1,2
    # A[0,2]=U[2], A[1,2]=U[5]
    
    # A[0,2]*B[0,0] -> C[0,0]
    c = 7
    U[2][c] = 1
    V[0][c] = 1
    W[0][c] = 1
    
    # A[0,2]*B[0,1] -> C[0,1]
    c = 8
    U[2][c] = 1
    V[1][c] = 1
    W[1][c] = 1
    
    # A[0,2]*B[1,0] -> C[0,0] ... wait, B[1,0] is row 1, col 0
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # A[0,2]*B[1,k] contributes to C[0,k] for k=0,1,2
    # A[0,2]*B[1,0] -> C[0,0]
    c = 9
    U[2][c] = 1
    V[3][c] = 1
    W[0][c] = 1
    
    # A[0,2]*B[1,1] -> C[0,1]
    c = 10
    U[2][c] = 1
    V[4][c] = 1
    W[1][c] = 1
    
    # A[1,2]*B[0,0] -> C[1,0]
    c = 11
    U[5][c] = 1
    V[0][c] = 1
    W[3][c] = 1
    
    # A[1,2]*B[0,1] -> C[1,1]
    c = 12
    U[5][c] = 1
    V[1][c] = 1
    W[4][c] = 1
    
    # A[1,2]*B[1,0] -> C[1,0]
    c = 13
    U[5][c] = 1
    V[3][c] = 1
    W[3][c] = 1
    
    # A[1,2]*B[1,1] -> C[1,1]
    c = 14
    U[5][c] = 1
    V[4][c] = 1
    W[4][c] = 1
    
    # A[0,2]*B[2,0] -> C[0,0]
    c = 15
    U[2][c] = 1
    V[6][c] = 1
    W[0][c] = 1
    
    # A[0,2]*B[2,1] -> C[0,1]
    c = 16
    U[2][c] = 1
    V[7][c] = 1
    W[1][c] = 1
    
    # Wait, I'm missing several terms. Let me recount.
    # Total terms needed beyond Strassen:
    # A[0:2, 2] @ B[2, 0:3] : 2*3 = 6 terms
    # A[0:2, 0:2] @ B[0:2, 2] : 2*2 = 4 terms  
    # A[0:2, 2] @ B[2, 2] : 1*1 = 1 term... wait
    # Actually: C[0:2, 2] = A[0:2, 0:2]@B[0:2,2] + A[0:2,2]@B[2,2]
    # That's 2*2 + 2*1 = 6 terms for the last column
    # C[0:2, 0:2] also has A[0:2,2]@B[2,0:2] = 2*2 = 4 terms
    # Total extra: 4 + 6 = 10 terms. 7 + 10 = 17. Good.
    
    # I already have 7 Strassen + 10 naive above, but let me verify I have all 10.
    # Cols 7-16 above give 10 terms. Let me check they cover:
    # A[0,2]*B[0,0], A[0,2]*B[0,1], A[0,2]*B[1,0], A[0,2]*B[1,1],
    # A[1,2]*B[0,0], A[1,2]*B[0,1], A[1,2]*B[1,0], A[1,2]*B[1,1],
    # A[0,2]*B[2,0], A[0,2]*B[2,1]
    # Missing: A[1,2]*B[2,0], A[1,2]*B[2,1], A[0:2, 0:2]*B[0:2, 2] terms
    # I'm missing terms! Let me redo this more carefully.
    
    # Let me restart the naive part properly.
    
    # Reinitialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (columns 0-6)
    # M1 = A[0,0] * (B[0,1] + B[1,1])
    c = 0
    U[0][c] = 1
    V[1][c] = 1
    V[4][c] = 1
    W[0][c] = 1   # C[0,0]
    W[4][c] = 1   # C[1,1]
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    c = 1
    U[0][c] = 1
    U[3][c] = 1
    V[4][c] = 1
    W[3][c] = 1   # C[1,0]
    W[4][c] = -1  # C[1,1]
    
    # M3 = (A[0,1] + A[1,1]) * B[0,0]
    c = 2
    U[1][c] = 1
    U[4][c] = 1
    V[0][c] = 1
    W[1][c] = 1   # C[0,1]
    W[4][c] = 1   # C[1,1]
    
    # M4 = A[1,1] * (B[0,0] - B[0,1])
    c = 3
    U[4][c] = 1
    V[0][c] = 1
    V[1][c] = -1
    W[0][c] = 1   # C[0,0]
    
    # M5 = (A[0,0] + A[0,1]) * (B[1,0] - B[1,1])
    c = 4
    U[0][c] = 1
    U[1][c] = 1
    V[3][c] = 1
    V[4][c] = -1
    W[0][c] = -1  # C[0,0]
    W[1][c] = 1   # C[0,1]
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    c = 5
    U[3][c] = 1
    U[0][c] = -1
    V[0][c] = 1
    V[3][c] = 1
    W[3][c] = 1   # C[1,0]
    W[4][c] = 1   # C[1,1]
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[0,0])
    c = 6
    U[1][c] = 1
    U[4][c] = -1
    V[4][c] = 1
    V[0][c] = 1
    W[0][c] = 1   # C[0,0]
    
    # Naive part: all terms NOT in the 2x2x2 core
    # Core is A[0:2,0:2] @ B[0:2,0:2]
    # Remaining:
    # 1. A[0:2, 2] @ B[0:2, 0:2] -> C[0:2, 0:2] (4 terms)
    # 2. A[0:2, 2] @ B[0:2, 2] -> C[0:2, 2] (4 terms)
    # 3. A[0:2, 2] @ B[2, 0:2] -> C[0:2, 0:2] (4 terms)
    # 4. A[0:2, 2] @ B[2, 2] -> C[0:2, 2] (2 terms)
    # 5. A[0:2, 0:2] @ B[0:2, 2] -> C[0:2, 2] (4 terms)
    # Wait, that's 18 extra. Total would be 25. That's wrong.
    
    # Let me think again. The full product is:
    # C[i,k] = sum_j A[i,j] * B[j,k] for j=0,1,2
    # = sum_{j=0,1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    # 
    # The Strassen part handles sum_{j=0,1} A[i,j]*B[j,k] for i=0,1 and k=0,1
    # 
    # Remaining terms:
    # For k=0,1: A[i,2]*B[2,k] for i=0,1 -> 4 terms
    # For k=2: sum_{j=0,1} A[i,j]*B[j,2] for i=0,1 -> 4 terms
    # For k=2: A[i,2]*B[2,2] for i=0,1 -> 2 terms
    # Total remaining: 4 + 4 + 2 = 10 terms
    # Total rank: 7 + 10 = 17. Correct!
    
    # Naive term 1: A[0,2]*B[2,0] -> C[0,0]
    c = 7
    U[2][c] = 1
    V[6][c] = 1
    W[0][c] = 1
    
    # Naive term 2: A[0,2]*B[2,1] -> C[0,1]
    c = 8
    U[2][c] = 1
    V[7][c] = 1
    W[1][c] = 1
    
    # Naive term 3: A[1,2]*B[2,0] -> C[1,0]
    c = 9
    U[5][c] = 1
    V[6][c] = 1
    W[3][c] = 1
    
    # Naive term 4: A[1,2]*B[2,1] -> C[1,1]
    c = 10
    U[5][c] = 1
    V[7][c] = 1
    W[4][c] = 1
    
    # Naive term 5: A[0,0]*B[0,2] -> C[0,2]
    c = 11
    U[0][c] = 1
    V[2][c] = 1
    W[2][c] = 1
    
    # Naive term 6: A[0,1]*B[1,2] -> C[0,2]
    c = 12
    U[1][c] = 1
    V[5][c] = 1
    W[2][c] = 1
    
    # Naive term 7: A[1,0]*B[0,2] -> C[1,2]
    c = 13
    U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1
    
    # Naive term 8: A[1,1]*B[1,2] -> C[1,2]
    c = 14
    U[4][c] = 1
    V[5][c] = 1
    W[5][c] = 1
    
    # Naive term 9: A[0,2]*B[2,2] -> C[0,2]
    c = 15
    U[2][c] = 1
    V[8][c] = 1
    W[2][c] = 1
    
    # Naive term 10: A[1,2]*B[2,2] -> C[1,2]
    c = 16
    U[5][c] = 1
    V[8][c] = 1
    W[5][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
