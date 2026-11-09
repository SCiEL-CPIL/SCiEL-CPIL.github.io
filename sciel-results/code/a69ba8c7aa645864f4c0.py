DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = 2, 4, 4
    # Try to find a decomposition with rank < 32
    # Use a structured approach
    
    rank = 24
    U = [[0]*rank for _ in range(n*m)]  # 8 x rank
    V = [[0]*rank for _ in range(m*p)]  # 16 x rank
    W = [[0]*rank for _ in range(p*n)]  # 8 x rank
    
    # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # We'll use a decomposition that groups j values
    
    # Idea: for each k, we need to compute 2 dot products of length 4
    # Use a 4-dimensional bilinear algorithm
    
    # Let's use the following: for each k, use 6 factors (like 2x2 Strassen generalized)
    # Actually, let's just try a direct but compressed approach
    
    # For now, let me try a specific known decomposition
    # (2,4,4) can be done in rank 20? Let me try.
    
    # I'll use a heuristic: assign random-ish but structured values and verify
    # Actually, let me just return the schoolbook for now but with a note
    # No, let me try to actually find a better one.
    
    # Try rank 20:
    # For each of the 4 k-values, we have 2 outputs (i=0,1)
    # Each output is a sum of 4 products
    # Total 8 "terms" but they share structure
    
    # Let me use the following decomposition:
    # For j-pairs (0,1) and (2,3), use Strassen-like on 2x2 blocks
    
    # Group j into {0,1} and {2,3}
    # For group {0,1}: 2x2 (A) x 2x4 (B) -> 2x4 (C contribution)
    # For group {2,3}: 2x2 (A) x 2x4 (B) -> 2x4 (C contribution)
    
    # 2x2 x 2x4 multiplication:
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01,b02,b03],[b10,b11,b12,b13]]
    # C = [[c00,c01,c02,c03],[c10,c11,c12,c13]]
    # c0k = a00*b0k + a01*b1k
    # c1k = a10*b0k + a11*b1k
    
    # This is 2x2 x 2x4. Rank? At most 2*2*4=16, but can be less.
    # Use: for each k, it's a 2x2 x 2x1, which has rank... 
    # 2x2 x 2x1: C[0,k]=a00*b0k+a01*b1k, C[1,k]=a10*b0k+a11*b1k
    # This is 4 products, rank 4? Or can we do better?
    # Actually for 2x2 x 2x1, the rank is 4 (I think it's optimal for this small case)
    # So 2x2 x 2x4 has rank at most 4*4=16, but we can share across k.
    
    # Hmm, let's just do it directly: for each k, 4 factors for the 2x2x2x1 part
    # So for group {0,1} and all 4 k's: 4*4=16 factors
    # For group {2,3} and all 4 k's: 4*4=16 factors
    # Total: 32. Same as schoolbook.
    
    # I need a smarter approach. Let me just return schoolbook for now.
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
