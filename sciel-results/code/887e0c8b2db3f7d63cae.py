"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strassen 2x2x2 decomposition (rank 7)
    # M1 = (A11+A22)*(B12-B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4: Kronecker product of Strassen with itself -> rank 49
    # For the 5th column: 16 schoolbook terms
    
    # Build 4x4x4 decomposition of rank 49
    # Each Strassen term for 2x2x2 becomes 7 terms
    # Kronecker: 7 * 7 = 49 terms
    
    # Strassen factors for 2x2x2
    # U: (m*n=4) x 7, V: (m*p=4) x 7, W: (p*n=4) x 7
    # Using indices: A is 2x2, B is 2x2, C is 2x2
    # U[i*m+j, c] = coefficient for A_{i,j}
    # V[j*p+k, c] = coefficient for B_{j,k}
    # W[k*n+i, c] = coefficient for C_{i,k}
    
    # Strassen 2x2x2:
    # U (4x7): rows are (0,0),(0,1),(1,0),(1,1)
    # V (4x7): rows are (0,0),(0,1),(1,0),(1,1)
    # W (4x7): rows are (0,0),(0,1),(1,0),(1,1)
    
    U_222 = [
        # M1, M2, M3, M4, M5, M6, M7
        [1, 0, 1, 0, 1, -1, 1],   # A11 (i=0,j=0)
        [0, 0, 0, 0, 1, 0, -1],   # A12 (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],    # A21 (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1],   # A22 (i=1,j=1)
    ]
    V_222 = [
        # M1, M2, M3, M4, M5, M6, M7
        [0, 1, 1, 0, 0, 1, 0],    # B11 (j=0,k=0)
        [1, 0, 0, -1, 0, 1, 0],   # B12 (j=0,k=1)
        [-1, 0, -1, 0, 0, 0, 1],  # B21 (j=1,k=0)
        [-1, 0, 0, 1, 1, 0, 1],   # B22 (j=1,k=1)
    ]
    W_222 = [
        # M1, M2, M3, M4, M5, M6, M7
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # C22 (k=1,i=1)
    ]
    
    # Verify Strassen 2x2x2
    # C = sum_c U[:,c] * V[:,c] * W[:,c] (outer product)
    # C_{i,k} = sum_j sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    def verify_222():
        for i in range(2):
            for k in range(2):
                total = 0
                for j in range(2):
                    for c in range(7):
                        total += U_222[i*2+j][c] * V_222[j*2+k][c] * W_222[k*2+i][c]
                # Should be 1 if i==k... wait, no.
                # C_{i,k} = sum_j A_{i,j} * B_{j,k}
                # The tensor T_{i,j,k} = delta_{i,k}... no.
                # T_{i,j,k} = 1 if this term contributes to C_{i,k} from A_{i,j}*B_{j,k}
                # So T_{i,j,k} = 1 always (it's just the multiplication tensor)
                # The decomposition: T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{i,k,c}
                # For matrix mult, T_{i,j,k} = 1 for all i,j,k
                # So sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] should = 1 for all i,j,k
                if total != 1:
                    return False
        return True
    
    # Build 4x4x4 via Kronecker
    # 4x4 = 2x2 blocks. A is 4x4, split into four 2x2 blocks: A_{ab} where a,b in {0,1}
    # A_{ab} is 2x2 block at (rows 2a:2a+2, cols 2b:2b+2)
    # Similarly for B and C.
    # C_{ac} = sum_b A_{ab} * B_{bc}
    # Using Strassen on the 2x2 block level: rank 7 for the block structure
    # Each Strassen term involves 2x2 matrix multiplications, each of rank 7
    # Total rank: 7 * 7 = 49
    
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]  # 4*4
    V_444 = [[0]*rank_444 for _ in range(16)]  # 4*4
    W_444 = [[0]*rank_444 for _ in range(16)]  # 4*4
    
    for c1 in range(7):  # outer Strassen (block level)
        for c2 in range(7):  # inner Strassen (within 2x2 blocks)
            c = c1 * 7 + c2
            # For each block position (a,b) of A, (b,c_block) of B, (c_block,a) of C
            for a in range(2):
                for b in range(2):
                    for c_block in range(2):
                        # A block (a,b): rows 2a:2a+2, cols 2b:2b+2
                        # B block (b,c_block): rows 2b:2b+2, cols 2*c_block:2*c_block+2
                        # C block (c_block,a): rows 2*c_block:2*c_block+2, cols 2a:2a+2
                        
                        # Inner Strassen indices within each 2x2 block
                        for i_local in range(2):
                            for j_local in range(2):
                                # A index: row 2a+i_local, col 2b+j_local
                                a_idx = (2*a + i_local) * 4 + (2*b + j_local)
                                # B index: row 2b+j_local, col 2*c_block + k_local
                                # W index: row 2*c_block + k_local, col 2a + i_local
                                for k_local in range(2):
                                    b_idx = (2*b + j_local) * 4 + (2*c_block + k_local)
                                    w_idx = (2*c_block + k_local) * 4 + (2*a + i_local)
                                    
                                    u_val = U_222[a*2+b][c1] * U_222[i_local*2+j_local][c2]
                                    v_val = V_222[b*2+c_block][c1] * V_222[j_local*2+k_local][c2]
                                    w_val = W_222[c_block*2+a][c1] * W_222[k_local*2+i_local][c2]
                                    
                                    U_444[a_idx][c] += u_val
                                    V_444[b_idx][c] += v_val
                                    W_444[w_idx][c] += w_val
    
    # Now add 16 terms for the 5th column (col 4 of B, col 4 of C)
    # C_{i,4} = sum_j A_{i,j} * B_{j,4}
    # Schoolbook: 16 terms
    rank_total = rank_444 + 16
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(20)]  # 4*5
    W = [[0]*rank_total for _ in range(20)]  # 5*4
    
    # Copy 4x4x4 part
    for i in range(16):
        for c in range(rank_444):
            U[i][c] = U_444[i][c]
            V[i][c] = V_444[i][c]
            W[i][c] = W_444[i][c]
    
    # Add 5th column terms
    for i in range(4):
        for j in range(4):
            c = rank_444 + i*4 + j
            U[i*4+j][c] = 1
            V[j*5+4][c] = 1
            W[4*4+i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
