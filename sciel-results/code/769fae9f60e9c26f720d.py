"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_decomp():
    """Return (U, V, W) for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    # U is 4x7, V is 4x7, W is 4x7
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen's 7 terms:
    # T1 = (A11+A22)(B11+B22)
    # T2 = (A21+A22)B11
    # T3 = A11(B21-B22)
    # T4 = A22(B12-B11)
    # T5 = (A11+A12)B22
    # T6 = (A21-A11)(B11+B12)
    # T7 = (A12-A22)(B21+B22)
    #
    # C11 = T1 + T4 - T5 - T7
    # C12 = T3 + T5
    # C21 = T2 + T4
    # C22 = T1 - T2 + T3 + T7
    
    # We need to express the matrix multiplication tensor as sum of rank-1 terms.
    # The standard Strassen decomposition gives 7 rank-1 tensors.
    # 
    # For each term T_t, we have:
    # U[:, t] gives the coefficients for A
    # V[:, t] gives the coefficients for B  
    # W[:, t] gives the coefficients for C
    #
    # Term 1: (A11+A22)(B11+B22)
    # A part: A11 + A22 -> U[0,0]=1, U[3,0]=1
    # B part: B11 + B22 -> V[0,0]=1, V[3,0]=1
    # C part: C11 += 1, C22 += 1 -> W[0,0]=1 (C11 is W[k*2+i] with k=0,i=0), W[3,0]=1 (C22 is W[3,0])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: (A21+A22)B11
    # A part: A21 + A22 -> U[2,1]=1, U[3,1]=1
    # B part: B11 -> V[0,1]=1
    # C part: C21 += 1, C22 -= 1 -> W[2,1]=1 (C21: k=1,i=0 -> index 1*2+0=2), W[3,1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: A11(B21-B22)
    # A part: A11 -> U[0,2]=1
    # B part: B21 - B22 -> V[2,2]=1, V[3,2]=-1
    # C part: C12 += 1, C22 += 1 -> W[1,2]=1 (C12: k=0,i=1 -> 0*2+1=1), W[3,2]=1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 4: A22(B12-B11)
    # A part: A22 -> U[3,3]=1
    # B part: B12 - B11 -> V[1,3]=1, V[0,3]=-1
    # C part: C11 += 1, C21 += 1 -> W[0,3]=1 (C11), W[2,3]=1 (C21)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5: (A11+A12)B22
    # A part: A11 + A12 -> U[0,4]=1, U[1,4]=1
    # B part: B22 -> V[3,4]=1
    # C part: C11 -= 1, C12 += 1 -> W[0,4]=-1 (C11), W[1,4]=1 (C12)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: (A21-A11)(B11+B12)
    # A part: A21 - A11 -> U[2,5]=1, U[0,5]=-1
    # B part: B11 + B12 -> V[0,5]=1, V[1,5]=1
    # C part: C21 += 1 -> W[2,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # Term 7: (A12-A22)(B21+B22)
    # A part: A12 - A22 -> U[1,6]=1, U[3,6]=-1
    # B part: B21 + B22 -> V[2,6]=1, V[3,6]=1
    # C part: C11 -= 1, C22 += 1 -> W[0,6]=-1 (C11), W[3,6]=1 (C22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W


def _build_2x2x6_block():
    """Build U, V, W for 2x2x6 matrix multiplication using 3 Strassen 2x2x2 blocks. Rank 21."""
    U, V, W = [], [], []
    base_U, base_V, base_W = _strassen_decomp()
    
    for s in range(3):
        # For the s-th 2x2x2 block (columns 2s, 2s+1 of the 6-column result)
        u_block = [row[:] for row in base_U]
        v_block = [row[:] for row in base_V]
        w_block = [row[:] for row in base_W]
        
        for c in range(7):
            U.append(u_block[c] if False else [0.0]*4)
            V.append([0.0]*4)
            W.append([0.0]*4)
            
    # Rebuild properly
    U = [[0.0]*21 for _ in range(4)]
    V = [[0.0]*21 for _ in range(4)]
    W = [[0.0]*21 for _ in range(4)]
    
    for s in range(3):
        offset = s * 7
        for c in range(7):
            for i in range(4):
                U[i][offset + c] = base_U[i][c]
                V[i][offset + c] = base_V[i][c]
                W[i][offset + c] = base_W[i][c]
    
    return U, V, W


def _build_1x4x6_block():
    """Build U, V, W for 1x4x6 matrix multiplication. Rank 24 (schoolbook)."""
    rank = 24
    U = [[0.0]*rank for _ in range(4)]   # n*m = 1*4 = 4 rows
    V = [[0.0]*rank for _ in range(24)]  # m*p = 4*6 = 24 rows
    W = [[0.0]*rank for _ in range(6)]   # p*n = 6*1 = 6 rows
    
    col = 0
    for j in range(4):
        for k in range(6):
            U[j][col] = 1
            V[j*6+k][col] = 1
            W[k][col] = 1
            col += 1
    
    return U, V, W


def construct():
    """
    Decompose 3x4x6 matrix multiplication into:
    - 1x4x6 block (top row of 3x4, times 4x6): rank 24
    - 2x4x6 block (bottom 2 rows of 3x4, times 4x6): split m=4 into 2+2, giving two 2x2x6 blocks, each rank 21
    Total rank: 24 + 21 + 21 = 66
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Build the two component blocks
    U1, V1, W1 = _build_1x4x6_block()    # rank 24
    U2, V2, W2 = _build_2x2x6_block()    # rank 21 each
    
    total_rank = 24 + 21 + 21  # 66
    
    # Full U: n*m = 12 rows, 66 columns
    U = [[0.0]*total_rank for _ in range(12)]
    # Full V: m*p = 24 rows, 66 columns
    V = [[0.0]*total_rank for _ in range(24)]
    # Full W: p*n = 18 rows, 66 columns
    W = [[0.0]*total_rank for _ in range(18)]
    
    # Place 1x4x6 block (rows 0-2 of A, i.e., i=0)
    # A rows 0-3 map to i=0, j=0..3 -> U indices 0..3
    # B rows 0-23 map to j=0..3, k=0..5 -> V indices 0..23
    # C columns: k=0..5, i=0 -> W indices k*3+0 = 0,3,6,9,12,15
    for c in range(24):
        for i in range(4):
            U[i][c] = U1[i][c]
        for j in range(24):
            V[j][c] = V1[j][c]
        for k in range(6):
            W[k*3 + 0][c] = W1[k][c]
    
    # Place first 2x2x6 block (rows 2-3 of A, i.e., i=1,2; columns 0-1 of B, j=0,1)
    # A rows: i=1,j=0 -> U[2], i=1,j=1 -> U[3], i=2,j=0 -> U[6], i=2,j=1 -> U[7]
    # B rows: j=0,k=0..5 -> V[0..5], j=1,k=0..5 -> V[6..11]
    # C cols: k=0..5, i=1 -> W[k*3+1], k=0..5, i=2 -> W[k*3+2]
    offset1 = 24
    for c in range(21):
        # U mapping: local (i_local, j_local) where i_local in {0,1} for i in {1,2}, j_local in {0,1}
        # Local U row 0 = i=1,j=0 -> global U[2]
        # Local U row 1 = i=1,j=1 -> global U[3]
        # Local U row 2 = i=2,j=0 -> global U[6]
        # Local U row 3 = i=2,j=1 -> global U[7]
        u_map = [2, 3, 6, 7]
        for lu in range(4):
            U[u_map[lu]][offset1 + c] = U2[lu][c]
        
        # V mapping: local row j_local*2+k -> global V[j_local*4 + k]
        # Local V row 0 = j=0,k=0 -> V[0]
        # Local V row 1 = j=0,k=1 -> V[1]
        # ...
        # Local V row j*2+k -> V[j*4+k]
        for lv in range(4):
            j_local = lv // 2
            k_local = lv % 2
            for k in range(6):
                # Actually, the 2x2x6 block has V with 4 rows (m*p for m=2,p=6... wait, m=2,p=6 -> 12 rows? No.)
                # 2x2x6: m=2, p=6, so V has m*p = 12 rows. But we built it as 4 rows?
                # Wait, _build_2x2x6_block returns U (4x21), V (4x21), W (4x21)
                # That's wrong. For 2x2x6, U should be 4x21, V should be 12x21, W should be 12x21
                pass
        break
    
    # I need to fix the 2x2x6 block construction. Let me rebuild it properly.
    
    # Actually, let me just build the whole thing directly.
    U = [[0.0]*total_rank for _ in range(12)]
    V = [[0.0]*total_rank for _ in range(24)]
    W = [[0.0]*total_rank for _ in range(18)]
    
    base_U, base_V, base_W = _strassen_decomp()
    
    # 1x4x6 block: rank 24
    col = 0
    for j in range(4):
        for k in range(6):
            U[j][col] = 1          # i=0, row j
            V[j*6+k][col] = 1
            W[k*3][col] = 1        # i=0, so k*3+0
            col += 1
    
    # First 2x2x6 block: rows i=1,2; cols j=0,1
    # Split p=6 into three 2s
    for s in range(3):
        for c in range(7):
            idx = 24 + s*7 + c
            # A part: i in {1,2}, j in {0,1}
            # base_U has 4 rows: (i_local, j_local) for i_local in {0,1}, j_local in {0,1}
            # i_local=0 -> i=1, i_local=1 -> i=2
            # j_local=0 -> j=0, j_local=1 -> j=1
            for lu in range(4):
                i_local = lu // 2
                j_local = lu % 2
                i_global = 1 + i_local
                j_global = 0 + j_local
                U[i_global*4 + j_global][idx] = base_U[lu][c]
            
            # B part: j in {0,1}, k in {2s, 2s+1}
            # base_V has 4 rows: (j_local, k_local) for j_local in {0,1}, k_local in {0,1}
            for lv in range(4):
                j_local = lv // 2
                k_local = lv % 2
                j_global = 0 + j_local
                k_global = 2*s + k_local
                V[j_global*6 + k_global][idx] = base_V[lv][c]
            
            # C part: k in {2s, 2s+1}, i in {1,2}
            # base_W has 4 rows: (k_local, i_local) for k_local in {0,1}, i_local in {0,1}
            for lw in range(4):
                k_local = lw // 2
                i_local = lw % 2
                k_global = 2*s + k_local
                i_global = 1 + i_local
                W[k_global*3 + i_global][idx] = base_W[lw][c]
    
    # Second 2x2x6 block: rows i=1,2; cols j=2,3
    for s in range(3):
        for c in range(7):
            idx = 45 + s*7 + c
            for lu in range(4):
                i_local = lu // 2
                j_local = lu % 2
                i_global = 1 + i_local
                j_global = 2 + j_local
                U[i_global*4 + j_global][idx] = base_U[lu][c]
            
            for lv in range(4):
                j_local = lv // 2
                k_local = lv % 2
                j_global = 2 + j_local
                k_global = 2*s + k_local
                V[j_global*6 + k_global][idx] = base_V[lv][c]
            
            for lw in range(4):
                k_local = lw // 2
                i_local = lw % 2
                k_global = 2*s + k_local
                i_global = 1 + i_local
                W[k_global*3 + i_global][idx] = base_W[lw][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
