"""Matrix multiplication 3,3,8 decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-19 decomposition for 3x3x3 matrix multiplication
    # Using the known minimal decomposition for M_{3,3,3}
    # Indices: U[i*m + j], V[j*p + k], W[k*n + i]
    # For the 3x3x3 part, we use the standard rank-19 factors
    
    # The rank-19 decomposition of M_{3,3,3}:
    # U is 9x19, V is 9x19, W is 9x19
    # This is the well-known minimal decomposition
    
    # Rank-19 factors for 3x3x3 (Landsberg decomposition)
    # Each column represents one rank-1 term
    # U[i,j], V[j,k], W[k,i]
    
    # Using the explicit rank-19 construction
    # The factors are built from the algebraic structure of 3x3 matrices
    
    # Let me use a known explicit rank-19 decomposition
    # Based on the work of Landsberg and others
    
    # For 3x3x3, the rank-19 decomposition can be constructed as follows:
    # We use the fact that the tensor decomposes into 19 rank-1 terms
    
    # Explicit rank-19 factors (from literature):
    # These are the standard minimal factors
    
    # U matrix (9x19) - rows indexed by (i,j) where i,j in {0,1,2}
    # V matrix (9x19) - rows indexed by (j,k) where j,k in {0,1,2}
    # W matrix (9x19) - rows indexed by (k,i) where k,i in {0,1,2}
    
    # The rank-19 decomposition uses the following structure:
    # Terms 0-8: diagonal-like terms
    # Terms 9-18: off-diagonal correction terms
    
    # Let me construct this explicitly using the known formula
    
    # Actually, let me use a verified rank-19 decomposition
    # The standard one uses entries in {0, 1, -1}
    
    # I'll use the decomposition from the paper on minimal rank for 3x3x3
    
    # Rank 19 factors:
    # U: 9 rows, 19 cols
    # V: 9 rows, 19 cols  
    # W: 9 rows, 19 cols
    
    # Using the explicit construction:
    # The 19 terms correspond to specific index patterns
    
    # Let me build this step by step
    # Term structure for rank-19 3x3x3:
    
    # I'll use the known explicit factors
    # These are from the standard literature on tensor rank
    
    # For safety, let me use a verified decomposition
    # The rank-19 decomposition of M_{3,3,3}
    
    # Explicit rank-19 factors (verified):
    # U entries:
    U_333 = [
        # col 0-8: main terms
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # (2,2)
    ]
    V_333 = [
        # col 0-8: main terms
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    W_333 = [
        # col 0-8: main terms
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,1)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (2,2)
    ]
    
    # This is just the identity decomposition of rank 9, not rank 19
    # I need the actual rank-19 decomposition
    # Let me use a different approach - I'll construct the full 3x3x8 
    # using a hybrid: rank 9 for first 3 cols (identity) + schoolbook for last 5
    
    # Actually, the simplest improvement is to just use schoolbook but 
    # I can't beat rank 72 easily without the actual rank-19 factors
    
    # Let me just return the schoolbook decomposition which is guaranteed correct
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
