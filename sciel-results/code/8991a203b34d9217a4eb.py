"""Kronecker product decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x4x8 tensor as Kronecker product of 3x2x4 and 2x2x4
    # 3x2x4: rank 12 (schoolbook)
    # 2x2x4: rank 14 (Strassen-like for 2x2x4)
    # Total rank: 12 * 14 = 168
    # 
    # Actually, let me use a better approach:
    # 3x4x8 = 3x(2x2)x(4x2) = (3x2x4) ⊗ (2x2x2)
    # 3x2x4: rank 12 (schoolbook)
    # 2x2x2: rank 7 (Strassen)
    # Total rank: 12 * 7 = 84
    #
    # This is better than the current 98.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 3x2x4 factors (schoolbook, rank 12)
    # A is 3x2, B is 2x4, C is 3x4
    # U1: 6x12, V1: 8x12, W1: 12x12
    U1 = [[0] * 12 for _ in range(6)]
    V1 = [[0] * 12 for _ in range(8)]
    W1 = [[0] * 12 for _ in range(12)]
    
    col1 = 0
    for i in range(3):
        for j in range(2):
            for k in range(4):
                U1[i * 2 + j][col1] = 1
                V1[j * 4 + k][col1] = 1
                W1[k * 3 + i][col1] = 1
                col1 += 1
    
    # 2x2x2 factors (Strassen, rank 7)
    # A is 2x2, B is 2x2, C is 2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    # Add more columns for Strassen
    U2 = [
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, -1]
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0]
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ]
    
    # Now combine via Kronecker product
    # Global U: (3*2)*(2*2) = 6*4 = 24 rows, 84 cols
    # Global V: (2*2)*(2*4) = 4*8 = 32 rows, 84 cols
    # Global W: (4*2)*(2*3) = 8*6 = 48 rows, 84 cols
    #
    # Wait, dimensions:
    # n=3, m=4, p=8
    # U has n*m = 12 rows
    # V has m*p = 32 rows
    # W has p*n = 24 rows
    #
    # The decomposition: 3x4x8 = (3x2x4) ⊗ (1x2x2)
    # 3x2x4: A is 3x2, B is 2x4, C is 3x4
    # 1x2x2: A is 1x2, B is 2x2, C is 1x2
    #
    # Kronecker: A = A1 ⊗ A2 where A1 is 3x2, A2 is 1x2 -> A is 3x4
    # B = B1 ⊗ B2 where B1 is 2x4, B2 is 2x2 -> B is 4x8
    # C = C1 ⊗ C2 where C1 is 3x4, C2 is 1x2 -> C is 3x8
    #
    # So we need:
    # T1: 3x2x4 with rank r1
    # T2: 1x2x2 with rank r2
    # Total rank = r1 * r2
    #
    # T2: 1x2x2 - this is just 2 independent products, rank 2
    # T1: 3x2x4 - schoolbook rank 12
    # Total: 12 * 2 = 24? No, that's too low.
    #
    # Let me reconsider. The Kronecker product of tensors:
    # If T1 has rank r1 and T2 has rank r2, then T1⊗T2 has rank r1*r2.
    #
    # 3x4x8 = (3x2) x (2x4) x (4x2)
    # = (3x2x4) ⊗ (1x2x2)
    #
    # 3x2x4: rank 12 (schoolbook)
    # 1x2x2: rank 2 (just two scalar products)
    # Total: 24
    #
    # But wait, 1x2x2 means A is 1x2, B is 2x2, C is 1x2
    # This is computing c_k = sum_j a_j * b_jk for k=0,1
    # That's just 2 independent dot products, rank 2.
    #
    # So total rank = 12 * 2 = 24? That seems too good.
    #
    # Let me verify: 3x4x8 tensor has 3*4*8 = 96 entries.
    # Rank 24 would be a huge improvement over 98.
    #
    # Actually, I think I'm making an error. Let me be more careful.
    #
    # The matrix multiplication tensor for (n,m,p) has indices (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p), and the tensor entry is 1 if the
    # output C[i,k] receives contribution from A[i,j]*B[j,k].
    #
    # For Kronecker: if we write n = n1*n2, m = m1*m2, p = p1*p2
    # Then the (n,m,p) tensor = (n1,m1,p1) tensor ⊗ (n2,m2,p2) tensor
    #
    # 3x4x8: n=3, m=4, p=8
    # 3 = 3*1, 4 = 2*2, 8 = 4*2
    # So T1 = 3x2x4, T2 = 1x2x2
    #
    # T2 = 1x2x2: A is 1x2, B is 2x2, C is 1x2
    # Tensor entries: T2[i,j,k] = 1 if i=0 and the product A[0,j]*B[j,k] contributes to C[0,k]
    # So T2[0,j,k] = 1 for all j,k. This is just the outer product of ones.
    # Rank of T2 = 2 (since it's a 1x2x2 tensor with all entries 1, it factors as
    # sum over j of e_j ⊗ e_j ⊗ (sum over k of e_k) ... no.
    #
    # Actually T2[0,j,k] = 1 for all j in {0,1}, k in {0,1}
    # This can be written as sum_{j=0}^{1} e_0 ⊗ e_j ⊗ (e_0 + e_1)
    # So rank 2.
    #
    # T1 = 3x2x4: schoolbook rank 12.
    #
    # Total rank = 12 * 2 = 24.
    #
    # Let me construct this properly.
    
    # Reset
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # T1: 3x2x4, schoolbook, rank 12
    # T2: 1x2x2, rank 2
    
    # For each pair (t1, t2) where t1 in [0,12), t2 in [0,2):
    # col = t1 * 2 + t2
    
    # T1 factors (schoolbook for 3x2x4)
    # U1: 6x12, V1: 8x12, W1: 12x12
    U1 = [[0] * 12 for _ in range(6)]
    V1 = [[0] * 12 for _ in range(8)]
    W1 = [[0] * 12 for _ in range(12)]
    
    col1 = 0
    for i1 in range(3):
        for j1 in range(2):
            for k1 in range(4):
                U1[i1 * 2 + j1][col1] = 1
                V1[j1 * 4 + k1][col1] = 1
                W1[k1 * 3 + i1][col1] = 1
                col1 += 1
    
    # T2 factors for 1x2x2, rank 2
    # T2[0,j,k] = 1 for all j,k
    # Decomposition: sum_{j=0}^{1} e_0 ⊗ e_j ⊗ (sum_k e_k)
    # U2: 2x2 (n2*m2 = 1*2 = 2 rows)
    # V2: 4x2 (m2*p2 = 2*2 = 4 rows)
    # W2: 2x2 (p2*n2 = 2*1 = 2 rows)
    
    U2 = [[1, 0],
          [0, 1]]
    
    V2 = [[1, 0],
          [0, 0],
          [0, 1],
          [0, 0]]
    
    W2 = [[1, 1],
          [1, 1]]
    
    # Now combine
    # Global index mapping:
    # i = i1 * n2 + i2, where n2=1, so i = i1
    # j = j1 * m2 + j2, where m2=2, so j = j1 * 2 + j2
    # k = k1 * p2 + k2, where p2=2, so k = k1 * 2 + k2
    
    for t1 in range(12):
        for t2 in range(2):
            col = t1 * 2 + t2
            
            # U contribution
            for i1 in range(3):
                for j1 in range(2):
                    if U1[i1 * 2 + j1][t1] != 0:
                        for i2 in range(1):
                            for j2 in range(2):
                                if U2[i2 * 2 + j2][t2] != 0:
                                    global_i = i1 * 1 + i2
                                    global_j = j1 * 2 + j2
                                    U[global_i * m + global_j][col] += U1[i1 * 2 + j1][t1] * U2[i2 * 2 + j2][t2]
            
            # V contribution
            for j1 in range(2):
                for k1 in range(4):
                    if V1[j1 * 4 + k1][t1] != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                if V2[j2 * 2 + k2][t2] != 0:
                                    global_j = j1 * 2 + j2
                                    global_k = k1 * 2 + k2
                                    V[global_j * p + global_k][col] += V1[j1 * 4 + k1][t1] * V2[j2 * 2 + k2][t2]
            
            # W contribution
            for k1 in range(4):
                for i1 in range(3):
                    if W1[k1 * 3 + i1][t1] != 0:
                        for k2 in range(2):
                            for i2 in range(1):
                                if W2[k2 * 1 + i2][t2] != 0:
                                    global_k = k1 * 2 + k2
                                    global_i = i1 * 1 + i2
                                    W[global_k * n + global_i][col] += W1[k1 * 3 + i1][t1] * W2[k2 * 1 + i2][t2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
