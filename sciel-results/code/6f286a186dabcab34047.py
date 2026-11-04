import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    p1, p2 = 4, 1
    rank_strassen = 49
    rank_trivial = 4
    rank_total = rank_strassen + rank_trivial
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- M_{4,4,4} part ---
    # Use 2-level Strassen.
    # Base: 2x2x2 Strassen
    def get_strassen_2x2x2():
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1 # C[1,0] -> W[0*2+1]=W[1]
        
        # M3 = A00(B10-B11)
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1 # C[0,1] -> W[1*2+0]=W[2]
        
        # M4 = A11(B01-B10)
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[2, 3] = -1
        W2[0, 3] = 1; W2[3, 3] = 1 # C[0,0] and C[1,1]
        
        # M5 = (A00+A10)B11
        U2[0, 4] = 1; U2[2, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1 # C[0,0] and C[1,0]
        
        # M6 = (A11-A01)(B00+B10)
        U2[3, 5] = 1; U2[1, 5] = -1
        V2[0, 5] = 1; V2[2, 5] = 1
        W2[3, 5] = 1 # C[1,1]
        
        # M7 = (A01-A11)(B11+B01)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[3, 6] = 1; V2[1, 6] = 1
        W2[0, 6] = -1; W2[2, 6] = 1 # C[0,0] and C[0,1]
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2x2()
    
    # Assemble 4x4x4
    # Block indices: br, bc in {0,1}
    # A[br*2:br*2+2, bc*2:bc*2+2]
    # U4 index: (i*4+j) where i=br*2+ii, j=bc*2+jj
    # V4 index: (j*4+k) where j=br*2+ii, k=bc*2+jj (Note: V is indexed by j,k of B)
    # Wait, V is for B. B is 4x4. B[j,k].
    # In Strassen, we multiply A_block * B_block.
    # A_block is 2x2, B_block is 2x2.
    # The scalar Strassen U2, V2, W2 correspond to:
    # U2[ii*2+jj, st] for A_block[ii, jj]
    # V2[ii*2+jj, st] for B_block[ii, jj]
    # W2[jj*2+ii, st] for C_block[jj, ii] ?
    # C = A B. C[i,k] = sum_j A[i,j] B[j,k].
    # In 2x2 Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    # W2 is indexed [k*2+i] for C[i,k].
    # So W2[0,0] is C[0,0]. W2[1,0] is C[0,1]. W2[0,1] is C[1,0]. W2[1,1] is C[1,1].
    # My previous W2 assignments:
    # Term 0 (M1): W2[0,0]=1 (C00), W2[3,0]=1 (C11). Correct.
    # Term 1 (M2): W2[1,1]=1 (C10). Correct.
    # Term 2 (M3): W2[2,2]=1 (C01). Correct.
    # Term 3 (M4): W2[0,3]=1 (C00), W2[3,3]=1 (C11). Correct.
    # Term 4 (M5): W2[0,4]=-1 (C00), W2[1,4]=1 (C10). Correct.
    # Term 5 (M6): W2[3,5]=1 (C11). Correct.
    # Term 6 (M7): W2[0,6]=-1 (C00), W2[2,6]=1 (C01). Correct.
    
    # Block Terms for 4x4x4:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B10-B11) -> C01 += M3
    # M4 = A11(B01-B10) -> C00 += M4, C11 += M4
    # M5 = (A00+A10)B11 -> C00 -= M5, C10 += M5
    # M6 = (A11-A01)(B00+B10) -> C11 += M6
    # M7 = (A01-A11)(B11+B01) -> C00 -= M7, C01 += M7
    
    # Define block contributions:
    # (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (br, bc, coeff)
    # B_blocks: list of (br, bc, coeff)
    # C_blocks: list of (br, bc, coeff)
    
    block_terms = [
        # M1
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # M2
        ([((1,0,1), (1,1,1))], [((0,0,1))], [((1,0,1))]),
        # M3
        ([((0,0,1))], [((1,0,1), (1,1,-1))], [((0,1,1))]),
        # M4
        ([((1,1,1))], [((0,1,1), (1,0,-1))], [((0,0,1), (1,1,1))]),
        # M5
        ([((0,0,1), (1,0,1))], [((1,1,1))], [((0,0,-1), (1,0,1))]),
        # M6
        ([((1,1,1), (0,1,-1))], [((0,0,1), (1,0,1))], [((1,1,1))]),
        # M7
        ([((0,1,1), (1,1,-1))], [((1,1,1), (0,1,1))], [((0,0,-1), (0,1,1))]),
    ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    col = 0
    for bt in range(7):
        A_blocks, B_blocks, C_blocks = block_terms[bt]
        for st in range(7):
            # A contribution
            for (abr, abc, ca) in A_blocks:
                for ii in range(2):
                    for jj in range(2):
                        u_idx = (abr * 2 + ii) * 4 + (abc * 2 + jj)
                        u2_idx = ii * 2 + jj
                        U4[u_idx, col] += ca * U2[u2_idx, st]
                        
            # B contribution
            for (bbr, bbc, cb) in B_blocks:
                for ii in range(2):
                    for jj in range(2):
                        v_idx = (bbr * 2 + ii) * 4 + (bbc * 2 + jj)
                        v2_idx = ii * 2 + jj
                        V4[v_idx, col] += cb * V2[v2_idx, st]
                        
            # C contribution
            for (cbr, cbc, cc) in C_blocks:
                for ii in range(2):
                    for jj in range(2):
                        # C block index: cbr, cbc
                        # Global i = cbr*2 + ii, Global k = cbc*2 + jj
                        # W index: k*4 + i
                        w_idx = (cbc * 2 + jj) * 4 + (cbr * 2 + ii)
                        # W2 index: jj*2 + ii (since W2 is [k*2+i] for 2x2)
                        w2_idx = jj * 2 + ii
                        W4[w_idx, col] += cc * W2[w2_idx, st]
                        
            col += 1
            
    U[:, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4
    
    # --- M_{4,4,1} part ---
    # Global k = 4
    for j in range(4):
        c = 49 + j
        # U: i*4+j
        for i in range(4):
            U[i * 4 + j, c] = 1
        # V: j*5+4
        V[j * 5 + 4, c] = 1
        # W: 4*4+i
        for i in range(4):
            W[4 * 4 + i, c] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
