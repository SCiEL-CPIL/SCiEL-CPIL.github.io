"""Matrix multiplication 2x3x4 rank-minimal decomposition via Strassen block decomposition.

Rank 18 decomposition:
- Split A (2x3) into A1 (2x2, rows i=0,1 cols j=0,1) and A2 (2x1, rows i=0,1 col j=2)
- Split B (3x4) into B1 (2x4, rows j=0,1) and B2 (1x4, row j=2)
- C = A1*B1 + A2*B2
- A2*B2 is 2x1x4, needs rank 4 (outer product)
- A1*B1 is 2x2x4, split into two 2x2x2 blocks, each uses Strassen rank 7
- Total rank: 4 + 7 + 7 = 18
"""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # U shape: (n*m, rank) = (6, 18)
    # V shape: (m*p, rank) = (12, 18)
    # W shape: (p*n, rank) = (8, 18)
    
    # Rank 18 decomposition
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 12 x 18
    W = [[0] * rank for _ in range(p * n)]  # 8 x 18
    
    # Indices: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Part 1: A2 * B2 (2x1x4), rank 4
    # A2 is rows i=0,1, col j=2 of A
    # B2 is row j=2, cols k=0,1,2,3 of B
    # For each k in 0..3: U[i,2,k] * V[2,k,k] * W[k,i,k] = 1
    # Use r = 0..3 for this block
    for k in range(4):
        r = k
        for i in range(2):
            U[i * m + 2][r] = 1  # U[i,2,r]
        V[2 * p + k][r] = 1      # V[2,k,r]
        for i in range(2):
            W[k * n + i][r] = 1  # W[k,i,r]
    
    # Part 2: A1 * B1 (2x2x4), split into left (k=0,1) and right (k=2,3)
    # A1 is rows i=0,1, cols j=0,1 of A
    # B1 is rows j=0,1, cols k=0,1,2,3 of B
    
    # Strassen for 2x2x2:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For left block (k=0,1): r = 4..10 (7 ranks)
    # For right block (k=2,3): r = 11..17 (7 ranks)
    
    def add_strassen_block(k_offset, r_offset):
        """Add Strassen decomposition for 2x2x2 block at columns j=0,1 and k=k_offset,k_offset+1."""
        # Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
        # Map to global: i = i_loc, j = j_loc, k = k_offset + k_loc
        # Strassen factors:
        # M1: U[i, j, r] = a11+a22 -> U[0,0,r]=1, U[1,1,r]=1
        #     V[j, k, r] = b11+b22 -> V[0,0,r]=1, V[1,1,r]=1
        #     W[k, i, r] = C11,C22 contribution
        # M2: U[i, j, r] = a21+a22 -> U[1,0,r]=1, U[1,1,r]=1
        #     V[j, k, r] = b11 -> V[0,0,r]=1
        #     W[k, i, r] = C21 contribution
        # M3: U[i, j, r] = a11 -> U[0,0,r]=1
        #     V[j, k, r] = b12-b22 -> V[0,1,r]=1, V[1,1,r]=-1
        #     W[k, i, r] = C12,C22 contribution
        # M4: U[i, j, r] = a22 -> U[1,1,r]=1
        #     V[j, k, r] = b21-b11 -> V[1,0,r]=1, V[0,0,r]=-1
        #     W[k, i, r] = C11,C21 contribution
        # M5: U[i, j, r] = a11+a12 -> U[0,0,r]=1, U[0,1,r]=1
        #     V[j, k, r] = b22 -> V[1,1,r]=1
        #     W[k, i, r] = C11 contribution
        # M6: U[i, j, r] = a21-a11 -> U[1,0,r]=1, U[0,0,r]=-1
        #     V[j, k, r] = b11+b12 -> V[0,0,r]=1, V[0,1,r]=1
        #     W[k, i, r] = C22 contribution
        # M7: U[i, j, r] = a12-a22 -> U[0,1,r]=1, U[1,1,r]=-1
        #     V[j, k, r] = b21+b22 -> V[1,0,r]=1, V[1,1,r]=1
        #     W[k, i, r] = C11 contribution
        
        # M1: r_offset + 0
        r = r_offset + 0
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + (k_offset + 0)][r] = 1; V[1 * p + (k_offset + 1)][r] = 1
        # W for M1: contributes to C11 and C22
        W[(k_offset + 0) * n + 0][r] = 1  # C11
        W[(k_offset + 1) * n + 1][r] = 1  # C22
        
        # M2: r_offset + 1
        r = r_offset + 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + (k_offset + 0)][r] = 1
        # W for M2: contributes to C21
        W[(k_offset + 0) * n + 1][r] = 1  # C21
        
        # M3: r_offset + 2
        r = r_offset + 2
        U[0 * m + 0][r] = 1
        V[0 * p + (k_offset + 1)][r] = 1; V[1 * p + (k_offset + 1)][r] = -1
        # W for M3: contributes to C12 and C22
        W[(k_offset + 1) * n + 0][r] = 1  # C12
        W[(k_offset + 1) * n + 1][r] = 1  # C22
        
        # M4: r_offset + 3
        r = r_offset + 3
        U[1 * m + 1][r] = 1
        V[1 * p + (k_offset + 0)][r] = 1; V[0 * p + (k_offset + 0)][r] = -1
        # W for M4: contributes to C11 and C21
        W[(k_offset + 0) * n + 0][r] = 1  # C11
        W[(k_offset + 0) * n + 1][r] = 1  # C21
        
        # M5: r_offset + 4
        r = r_offset + 4
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + (k_offset + 1)][r] = 1
        # W for M5: contributes to C11
        W[(k_offset + 0) * n + 0][r] = -1  # C11 = M1+M4-M5+M7
        
        # M6: r_offset + 5
        r = r_offset + 5
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + (k_offset + 0)][r] = 1; V[0 * p + (k_offset + 1)][r] = 1
        # W for M6: contributes to C22
        W[(k_offset + 1) * n + 1][r] = 1  # C22 = M1-M2+M3+M6
        
        # M7: r_offset + 6
        r = r_offset + 6
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + (k_offset + 0)][r] = 1; V[1 * p + (k_offset + 1)][r] = 1
        # W for M7: contributes to C11
        W[(k_offset + 0) * n + 0][r] = 1  # C11 = M1+M4-M5+M7
    
    # Left block: k=0,1, r=4..10
    add_strassen_block(0, 4)
    # Right block: k=2,3, r=11..17
    add_strassen_block(2, 11)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
