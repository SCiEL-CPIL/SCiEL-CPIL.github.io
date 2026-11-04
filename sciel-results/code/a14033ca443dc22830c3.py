"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Split p=7 into p1=4 and p2=3
    # Use optimal decompositions for M_{3,4,4} (rank 18) and M_{3,4,3} (rank 15)
    # Total rank = 33
    
    n, m = 3, 4
    p1, p2 = 4, 3  # p = p1 + p2 = 7
    
    # Known optimal rank-18 decomposition for M_{3,4,4}
    # This is based on the fact that M_{3,4,4} can be decomposed with rank 18
    # using a structured approach
    
    # For M_{n,m,p}, the tensor has entries T[i,j,k] = delta(i,j,k)
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if i,j,k form valid triple, else 0
    
    # Let's use a simple but valid decomposition first, then optimize
    # Actually, let's use the known result that M_{3,4,4} has rank 18
    
    # Construct rank-18 decomposition for M_{3,4,4}
    def construct_m_3_4_4():
        n, m, p = 3, 4, 4
        rank = 18
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Use a structured decomposition
        # For each i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
        # We need to express the identity tensor as sum of rank-1 terms
        
        # One approach: use the fact that we can group terms
        # Let's use a basis approach
        for r in range(rank):
            # Determine which (i,j,k) triple this rank-1 term corresponds to
            # We need a bijection from {0,...,17} to a subset of {0,1,2}x{0,1,2,3}x{0,1,2,3}
            # that covers all 48 required triples with 18 terms (impossible directly)
            # So we need linear combinations
            
            # Actually, the minimal rank for M_{3,4,4} is 18, so there exists such a decomposition
            # Let me use a known construction
            
            # For M_{n,m,p} with small dimensions, we can use explicit constructions
            # Here's one approach using block structure
            pass
        
        # Let me use a simpler valid decomposition (not necessarily minimal)
        # and then try to reduce
        rank = n * m * p  # 48, schoolbook
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Actually, let me just use the schoolbook for now and verify correctness
    # The key insight is that we can split the p dimension
    
    # For M_{3,4,7}, split into M_{3,4,4} and M_{3,4,3}
    # The tensor for M_{3,4,7} has T[i,j,k] = 1 if k < 7, else 0
    # We can write this as T1[i,j,k] + T2[i,j,k] where:
    # T1[i,j,k] = 1 if k < 4 (this is M_{3,4,4} embedded in first 4 positions)
    # T2[i,j,k] = 1 if k >= 4 (this is M_{3,4,3} embedded in last 3 positions)
    
    # For T1: use rank-18 decomposition of M_{3,4,4}
    # For T2: use rank-15 decomposition of M_{3,4,3}
    # Total: 18 + 15 = 33
    
    # Let me construct explicit decompositions
    
    # M_{3,4,3} has known optimal rank 15
    def get_m_3_4_3_decomp():
        n, m, p = 3, 4, 3
        rank = 15
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Known optimal rank-15 decomposition for M_{3,4,3}
        # This can be constructed using the fact that M_{3,4,3} = M_{3,3,3} + M_{1,4,3}
        # M_{3,3,3} has rank 23 (not helpful) or use other methods
        
        # Actually, for M_{3,4,3}, one can use:
        # Split j dimension: j in {0,1,2,3}
        # Use rank-12 for j in {0,1,2} (M_{3,3,3} part) and rank-3 for j=3
        # But M_{3,3,3} has rank 23, so this doesn't work directly
        
        # Let me use a valid decomposition (not necessarily minimal)
        rank = n * m * p  # 36
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # M_{3,4,4} - use schoolbook for now
    def get_m_3_4_4_decomp():
        n, m, p = 3, 4, 4
        rank = n * m * p  # 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Get decompositions
    U1, V1, W1 = get_m_3_4_4_decomp()
    U2, V2, W2 = get_m_3_4_3_decomp()
    
    r1 = len(U1[0])  # 48
    r2 = len(U2[0])  # 36
    
    # Combine: total rank = r1 + r2
    total_rank = r1 + r2
    
    # U: shape (n*m, total_rank) = (12, 84)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]  # (28, 84)
    W = [[0]*total_rank for _ in range(p*n)]  # (21, 84)
    
    # Fill U, V, W for first part (k in 0..3)
    for i in range(n):
        for j in range(m):
            for r in range(r1):
                U[i*m+j][r] = U1[i*m+j][r]
    
    for j in range(m):
        for k in range(p1):
            for r in range(r1):
                V[j*p+k][r] = V1[j*p1+k][r]
    
    for k in range(p1):
        for i in range(n):
            for r in range(r1):
                W[k*n+i][r] = W1[k*n+i][r]
    
    # Fill U, V, W for second part (k in 4..6)
    offset = r1
    for i in range(n):
        for j in range(m):
            for r in range(r2):
                U[i*m+j][offset+r] = U2[i*m+j][r]
    
    for j in range(m):
        for k in range(p2):
            for r in range(r2):
                V[j*p + p1 + k][offset+r] = V2[j*p2+k][r]
    
    for k in range(p2):
        for i in range(n):
            for r in range(r2):
                W[(p1+k)*n+i][offset+r] = W2[k*n+i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
