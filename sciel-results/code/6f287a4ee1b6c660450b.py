"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition:
    # m=4 -> 2 blocks of size 2
    # p=8 -> 4 blocks of size 2
    # n=3 -> 1 block of size 2, 1 block of size 1
    
    # Total rank = 2 (m_blocks) * 4 (p_blocks) * (7 (strassen) + 4 (schoolbook 1x2x2)) = 88
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_A = [
        [1, 0, 0, 0, 0, 0, 0], # M1: A00
        [0, 0, 0, 0, 0, 0, 0], # M1: A11
        [0, 1, 0, 0, 0, 1, 0], # M2: A10, M6: A10
        [0, 1, 0, 0, 0, -1, 0],# M2: A11, M6: -A00? No.
        # Let's map carefully.
        # U indices for 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # M1: A00+A11 -> U[0]+=1, U[3]+=1
        # M2: A10+A11 -> U[2]+=1, U[3]+=1
        # M3: A00 -> U[0]+=1
        # M4: A11 -> U[3]+=1
        # M5: A00+A01 -> U[0]+=1, U[1]+=1
        # M6: A10-A00 -> U[2]+=1, U[0]-=1
        # M7: A01-A11 -> U[1]+=1, U[3]-=1
    ]
    
    # Correct Strassen U vectors (size 4)
    sU = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Correct Strassen V vectors (size 4)
    sV = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, -1, 0],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # Correct Strassen W vectors (size 4)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    sW = [
        [1, 0, 0, 1],  # M1: C00, C11
        [0, 0, 1, -1], # M2: C10, -C11
        [0, 1, 0, 1],  # M3: C01, C11
        [-1, 0, 1, 0], # M4: -C00, C10
        [1, 0, 0, 0],  # M5: C00
        [0, 0, 0, 1],  # M6: C11
        [-1, 1, 0, 0]  # M7: -C00, C01
    ]
    
    # Schoolbook 1x2x2
    # A is 1x2 (indices 0,1)
    # B is 2x2 (indices 0,1,2,3)
    # C is 1x2 (indices 0,1)
    # S1 = A0 B0
    # S2 = A1 B1
    # S3 = A0 B2
    # S4 = A1 B3
    # C0 = S1 + S2
    # C1 = S3 + S4
    
    sbU = [
        [1, 0], # S1
        [0, 1], # S2
        [1, 0], # S3
        [0, 1]  # S4
    ]
    
    sbV = [
        [1, 0, 0, 0], # S1
        [0, 1, 0, 0], # S2
        [0, 0, 1, 0], # S3
        [0, 0, 0, 1]  # S4
    ]
    
    sbW = [
        [1, 0], # S1
        [1, 0], # S2
        [0, 1], # S3
        [0, 1]  # S4
    ]
    
    for m_block in range(2):
        for p_block in range(4):
            # Process N0 (size 2)
            for r in range(7):
                # Update U, V, W for Strassen term r
                # U indices: i in 0..1, j_local in 0..1
                # Global i: 0, 1
                # Global j: m_block*2 + j_local
                for i_local in range(2):
                    i = i_local
                    for j_local in range(2):
                        j = m_block * 2 + j_local
                        val = sU[r][i_local * 2 + j_local]
                        if val != 0:
                            U[i * m + j][col] += val
                
                for j_local in range(2):
                    j = m_block * 2 + j_local
                    for k_local in range(2):
                        k = p_block * 2 + k_local
                        val = sV[r][j_local * 2 + k_local]
                        if val != 0:
                            V[j * p + k][col] += val
                
                for i_local in range(2):
                    i = i_local
                    for k_local in range(2):
                        k = p_block * 2 + k_local
                        val = sW[r][k_local * 2 + i_local]
                        if val != 0:
                            W[k * n + i][col] += val
                            
                col += 1
            
            # Process N1 (size 1)
            for r in range(4):
                # Update U, V, W for Schoolbook term r
                # U indices: i=2, j_local in 0..1
                for j_local in range(2):
                    i = 2
                    j = m_block * 2 + j_local
                    val = sbU[r][j_local]
                    if val != 0:
                        U[i * m + j][col] += val
                
                for j_local in range(2):
                    j = m_block * 2 + j_local
                    for k_local in range(2):
                        k = p_block * 2 + k_local
                        val = sbV[r][j_local * 2 + k_local]
                        if val != 0:
                            V[j * p + k][col] += val
                
                for k_local in range(2):
                    i = 2
                    k = p_block * 2 + k_local
                    val = sbW[r][k_local]
                    if val != 0:
                        W[k * n + i][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
