"""Program for matrix multiplication 2x3x9 with reduced rank."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # For 2xm x mxp, rank = 2m + 2p - 4
    rank = 2 * m + 2 * p - 4  # 2*3 + 2*9 - 4 = 6 + 18 - 4 = 20
    
    # U: shape (n*m, rank) = (6, 20)
    # V: shape (m*p, rank) = (27, 20)  
    # W: shape (p*n, rank) = (18, 20)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construction for 2xm x mxp with rank 2m+2p-4:
    # Terms 0..(2m-1): correspond to (i, j) pairs, i in {0,1}, j in {0..m-1}
    #   These contribute to V at column k=0 only
    # Terms 2m..(2m+2p-5): correspond to (j, k) pairs, j in {0..m-1}, k in {1..p-1}
    #   Wait, this doesn't quite work. Let me think differently.
    
    # Actually, let me use a verified construction:
    # For each output entry C[i,k] (i=0,1; k=0..p-1):
    #   C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # A decomposition with rank 2m + 2p - 4:
    # Use 2m terms for the "A-side" and 2p-4 terms for "shared B-side"
    
    # Let me just use a simpler but still valid construction:
    # Rank = n*m*p is always valid. Let me try to find a better one.
    
    # For now, let me use a construction I can verify:
    # We'll use the fact that we can combine terms.
    
    # Actually, let me just implement a known result:
    # For 2xm x mxp, use rank 2m + 2p - 4
    
    # Term indices:
    # t = 0, 1: for i=0, j=0,1; k=0
    # t = 2, 3: for i=1, j=0,1; k=0  
    # t = 4, 5: for i=0,1; j=2; k=0
    # t = 6..19: for j=0,1,2; k=1..8 (but that's 3*8=24, too many)
    
    # Let me just do schoolbook to be safe, then optimize in future iterations.
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
# EVOLVE-BLOCK-END
