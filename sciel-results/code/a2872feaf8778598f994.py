"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    A = [[A00, A01], [A10, A11]] -> A[i*2+j]
    B = [[B00, B10], [B01, B11]] -> B[j*2+k]
    C = A*B
    C[i*2+k] = sum_j A[i*2+j] * B[j*2+k]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c0: A00 * B00 -> C00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # c1: A10 * B01 -> C10
    U[2][1] = 1; V[1][1] = 1; W[2][1] = 1
    # c2: A01 * B10 -> C01
    U[1][2] = 1; V[2][2] = 1; W[1][2] = 1
    # c3: A11 * B11 -> C11
    U[3][3] = 1; V[3][3] = 1; W[3][3] = 1
    # c4: (A00+A11)*(B00+B11) -> C00 + C11
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[3][4] = 1
    # c5: (A10+A11)*B11 -> C11 - C10? No, C11 uses (A10+A11)*B11 - A10*B01?
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's use the standard Strassen decomposition directly.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[0][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_decomps(U1, V1, W1, U2, V2, W2, offset, add_to_U, add_to_V, add_to_W):
    pass # Not needed

def construct():
    start_time = time.time()
    deadline = 115.0 # 150s limit, keep 35s margin for safety
    
    n, m, p = DIMENSIONS
    
    # 1. Get 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomp()
    
    # 2. Build 4x4x4 decomposition using 2x2 Strassen
    # A (4x4) viewed as 2x2 blocks of 2x2
    # B (4x4) viewed as 2x2 blocks of 2x2
    # C = A B
    # C_blocks[i][j] = sum_k A_blocks[i][k] * B_blocks[k][j]
    # We need 7 block multiplications, each of rank 7.
    # Total rank 49.
    
    rank_44 = 49
    U4 = [[0]*rank_44 for _ in range(16)] # n*m = 16
    V4 = [[0]*rank_44 for _ in range(16)] # m*p = 16
    W4 = [[0]*rank_44 for _ in range(16)] # p*n = 16
    
    # Block indices:
    # A_block[r][c] -> rows r*2..r*2+1, cols c*2..c*2+1
    # A_flat[i*2+j] = A[i][j]
    # Block (r,c) corresponds to flat indices:
    # r=0,c=0: 0,1,2,3
    # r=0,c=1: 4,5,6,7
    # r=1,c=0: 8,9,10,11
    # r=1,c=1: 12,13,14,15
    
    # Strassen for 2x2 matrix of blocks:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term t (0..6):
    # U term: linear combination of A blocks
    # V term: linear combination of B blocks
    # W term: linear combination of C blocks (which are sums of block products)
    
    # Let's define the block indices for A, B, C
    # A blocks: 0,1,2,3 (r,c)
    # B blocks: 0,1,2,3 (c,k)
    # C blocks: 0,1,2,3 (r,k)
    
    # For a given Strassen term t:
    # U4 for A blocks involved:
    # V4 for B blocks involved:
    # W4 for C blocks involved:
    
    # We need to map block indices to flat indices.
    # A block (r,c) -> flat indices r*4 + c*2 ... r*4 + c*2+1 ? No.
    # A_flat[i*m+j] with m=4.
    # A[0][0]=0, A[0][1]=1, A[1][0]=2, A[1][1]=3 -> Block 00
    # A[0][2]=4, A[0][3]=5, A[1][2]=6, A[1][3]=7 -> Block 01
    # A[2][0]=8, A[2][1]=9, A[3][0]=10, A[3][1]=11 -> Block 10
    # A[2][2]=12, A[2][3]=13, A[3][2]=14, A[3][3]=15 -> Block 11
    
    # So block (r,c) corresponds to flat indices:
    # base = r*4 + c*2
    # indices = [base, base+1, base+2, base+3]
    
    # B block (c,k) -> flat indices:
    # B_flat[j*p+k] with p=4.
    # B[0][0]=0, B[1][0]=1, B[2][0]=2, B[3][0]=3 -> Block 00 (c=0,k=0)
    # B[0][1]=4, B[1][1]=5, B[2][1]=6, B[3][1]=7 -> Block 01 (c=0,k=1)
    # B[0][2]=8, B[1][2]=9, B[2][2]=10, B[3][2]=11 -> Block 10 (c=1,k=0)
    # B[0][3]=12, B[1][3]=13, B[2][3]=14, B[3][3]=15 -> Block 11 (c=1,k=1)
    # base = c*4 + k*2
    
    # C block (r,k) -> flat indices:
    # W_flat[k*n+i] with n=4.
    # C[0][0]=0, C[1][0]=1, C[2][0]=2, C[3][0]=3 -> Block 00 (r=0,k=0)
    # C[0][1]=4, C[1][1]=5, C[2][1]=6, C[3][1]=7 -> Block 01 (r=0,k=1)
    # C[0][2]=8, C[1][2]=9, C[2][2]=10, C[3][2]=11 -> Block 10 (r=1,k=0)
    # C[0][3]=12, C[1][3]=13, C[2][3]=14, C[3][3]=15 -> Block 11 (r=1,k=1)
    # base = k*4 + r*2
    
    # Strassen terms definition:
    # t, A_blocks, B_blocks, C_blocks
    # A_blocks: list of (block, coeff)
    # B_blocks: list of (block, coeff)
    # C_blocks: list of (block, coeff)
    
    strassen_terms = [
        # M1
        {
            'A': [(0,1), (3,1)],
            'B': [(0,1), (3,1)],
            'C': [(0,1), (3,1)]
        },
        # M2
        {
            'A': [(2,1), (3,1)],
            'B': [(0,1)],
            'C': [(2,1), (3,1)]
        },
        # M3
        {
            'A': [(0,1)],
            'B': [(1,1), (3,-1)],
            'C': [(1,1), (3,1)]
        },
        # M4
        {
            'A': [(3,1)],
            'B': [(2,1), (0,-1)],
            'C': [(0,1), (2,1)]
        },
        # M5
        {
            'A': [(0,1), (1,1)],
            'B': [(3,1)],
            'C': [(1,1), (0,1)]
        },
        # M6
        {
            'A': [(2,1), (0,-1)],
            'B': [(0,1), (1,1)],
            'C': [(3,1)]
        },
        # M7
        {
            'A': [(1,1), (3,-1)],
            'B': [(2,1), (3,1)],
            'C': [(0,1)]
        }
    ]
    
    # Now, for each Strassen term t, and for each 2x2 multiplication inside the block,
    # we combine the decompositions.
    
    # For each t in 0..6:
    #   For each 2x2 decomposition component c in 0..6 (from _strassen_2x2_decomp):
    #       col = t*7 + c
    #       For each A block (b, coeff) in strassen_terms[t]['A']:
    #           For each flat index (u, u_coeff) in U2[c]:
    #               U4[base_A(b) + u][col] += coeff * u_coeff
    #       Similarly for V and W.
    
    for t, term in enumerate(strassen_terms):
        for c in range(7):
            col = t*7 + c
            # U part
            for b, coeff in term['A']:
                base_A = b*4 # Wait, block index mapping?
                # Block 00 -> base 0, 01 -> base 4, 10 -> base 8, 11 -> base 12
                # So base = b * 4 ?
                # b=0 -> 0, b=1 -> 4, b=2 -> 8, b=3 -> 12. Yes.
                for u in range(4):
                    if U2[u][c] != 0:
                        U4[base_A + u][col] += coeff * U2[u][c]
                        
            # V part
            for b, coeff in term['B']:
                base_B = b*4
                for v in range(4):
                    if V2[v][c] != 0:
                        V4[base_B + v][col] += coeff * V2[v][c]
                        
            # W part
            for b, coeff in term['C']:
                base_C = b*4
                for w in range(4):
                    if W2[w][c] != 0:
                        W4[base_C + w][col] += coeff * W2[w][c]

    # 3. Extend to 4x4x9
    # We have 4x4x4 (rank 49).
    # We need 4x4x9.
    # We can use the 4x4x4 decomposition for the first 8 columns, and schoolbook for the 9th?
    # Wait, the 4x4x4 decomposition multiplies A (4x4) by B (4x4).
    # Here B is 4x9.
    # We can split B into B1 (4x8) and B2 (4x1).
    # C1 = A * B1 (4x8), C2 = A * B2 (4x1).
    # C1 requires 4x4x8.
    # We can use the 4x4x4 decomposition for columns 0..3 and 4..7.
    # For columns 0..3: use 4x4x4 decomp.
    # For columns 4..7: use another 4x4x4 decomp?
    # But the U matrix is the same for all columns of B?
    # No, U depends on A.
    # In the tensor decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # If we fix the 4x4x4 decomp, it handles k=0..3.
    # To handle k=4..7, we can replicate the decomp, but with different V and W?
    # Actually, since A is the same, the U vectors can be shared?
    # No, the columns of the decomposition tensor are independent.
    # However, if we just append the 4x4x4 decomp for the second 4 columns, the rank increases by 49.
    # So 4x4x8 would be 98?
    # Wait, we can merge them?
    # No, because the V and W matrices are different for different k.
    # Actually, for 4x4x8, we can just use two 4x4x4 decompositions.
    # Rank 98.
    # Then 4x4x1 is rank 4.
    # Total 102.
    
    # But wait, we can do better.
    # The task is 4x4x9.
    # We can just use schoolbook for the whole thing? Rank 144.
    # Or use Strassen for 4x4x4 and schoolbook for the rest?
    
    # Let's check the time.
    if time.time() - start_time > deadline:
        # Fallback to schoolbook
        rank = n*m*p
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
        return {"U": U, "V": V, "W": W}

    # Construct 4x4x9
    rank_total = 49 + 49 + 4 # 102
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(36)]
    W = [[0]*rank_total for _ in range(36)]
    
    # Copy 4x4x4 decomp for k=0..3
    for c in range(49):
        for i in range(16):
            U[i][c] = U4[i][c]
        for j in range(16):
            V[j][c] = V4[j][c]
        for k in range(16):
            W[k][c] = W4[k][c]
            
    # Copy 4x4x4 decomp for k=4..7
    # V indices for k=4..7:
    # B[j*9+k]
    # For k=4..7, j=0..3
    # V_flat[j*9+k]
    # In V4, we have indices for k=0..3 (j*4+k)
    # We need to map V4[j*4+k] to V[j*9 + (k+4)]
    # W indices for k=4..7:
    # W_flat[k*4+i]
    # In W4, we have indices for k=0..3 (k*4+i)
    # We need to map W4[k*4+i] to W[(k+4)*4 + i]
    
    for c in range(49):
        col2 = 49 + c
        for i in range(16):
            U[i][col2] = U4[i][c]
            
        for j in range(4):
            for k in range(4):
                v_idx4 = j*4 + k
                v_idx9 = j*9 + (k+4)
                V[v_idx9][col2] = V4[v_idx4][c]
                
        for k in range(4):
            for i in range(4):
                w_idx4 = k*4 + i
                w_idx9 = (k+4)*4 + i
                W[w_idx9][col2] = W4[w_idx4][c]
                
    # Schoolbook for k=8
    for i in range(4):
        for j in range(4):
            col3 = 98 + (i*4 + j)
            U[i*4+j][col3] = 1
            V[j*9+8][col3] = 1
            W[8*4+i][col3] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

