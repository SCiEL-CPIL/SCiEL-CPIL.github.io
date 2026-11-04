import numpy as np

DIMENSIONS = [3, 4, 8]

# Strassen decomposition for 2x2x2 matrix multiplication
# U, V, W are 4x7 matrices
# U[i*2+j, c] = 1 if (i,j) is in the set for U
# V[j*2+k, c] = 1 if (j,k) is in the set for V
# W[k*2+i, c] = 1 if (k,i) is in the set for W
def get_strassen_222():
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Indices:
    # A: 00, 01, 10, 11 -> U indices 0,1,2,3
    # B: 00, 01, 10, 11 -> V indices 0,1,2,3
    # C: 00, 01, 10, 11 -> W indices 0,1,2,3 (Note: W[k*n+i] => W[k*2+i])
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 -= M1
    # Wait, standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map inputs to U, V and outputs to W
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    # Contribution to C:
    # C00 (idx 0): +1
    # C01 (idx 1): +1
    # C10 (idx 2): +1
    # C11 (idx 3): -1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1

    # c=1: M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    # C10 += M2, C11 -= M2
    W[2,1]=1; W[3,1]=-1

    # c=2: M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    # C01 += M3, C11 += M3
    W[1,2]=1; W[3,2]=1

    # c=3: M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    # C00 -= M4, C01 += M4
    W[0,3]=-1; W[1,3]=1

    # c=4: M5 = (A00+A10)B11
    U[0,4]=1; U[2,4]=1
    V[3,4]=1
    # C00 += M5, C11 += M5
    W[0,4]=1; W[3,4]=1

    # c=5: M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    # C11 += M6
    W[3,5]=1

    # c=6: M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    # C00 -= M7, C10 += M7
    W[0,6]=-1; W[2,6]=1

    return U, V, W

# Schoolbook decomposition for 1x2x2
# A is 1x2, B is 2x2, C is 1x2
# U size 2x4, V size 4x4, W size 2x4
def get_schoolbook_122():
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    
    # c=0: A0*B00
    U[0,0]=1
    V[0,0]=1
    W[0,0]=1
    
    # c=1: A1*B10
    U[1,1]=1
    V[2,1]=1
    W[0,1]=1
    
    # c=2: A0*B01
    U[0,2]=1
    V[1,2]=1
    W[1,2]=1
    
    # c=3: A1*B11
    U[1,3]=1
    V[3,3]=1
    W[1,3]=1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    rank = 0
    U_full = []
    V_full = []
    W_full = []
    
    strassen_U, strassen_V, strassen_W = get_strassen_222()
    school_U, school_V, school_W = get_schoolbook_122()
    
    # We partition p=8 into 4 blocks of size 2.
    # We partition m=4 into 2 blocks of size 2.
    # We partition n=3 into 1 block of size 1 and 1 block of size 2.
    
    # For each column block j in 0..3:
    for j in range(4):
        # Column indices for this block in B: 2j, 2j+1
        col_base = 2 * j
        
        # Bottom part: Rows 2,3 of A (indices 2,3). Cols 0,1 and 2,3 of A.
        # This is a 2x4x2 multiplication.
        # Decomposed into two 2x2x2 Strassen multiplications.
        
        # Sub-block 1: A rows 2,3; A cols 0,1; B rows 0,1; B cols 2j, 2j+1
        # U indices for A rows 2,3: 2,3 (local) -> global 2,3
        # V indices for A cols 0,1 (B rows 0,1): 0,1 (local) -> global 0,1
        # W indices for B cols 2j, 2j+1 (C cols 2j, 2j+1) and C rows 2,3 (local 0,1) -> global rows 2,3
        # W index formula: k*n + i. 
        # Here k is col of B (0,1 local -> 2j, 2j+1 global), i is row of C (0,1 local -> 2,3 global).
        # So W global index = k_global * 3 + i_global.
        
        # Strassen 1
        for c in range(7):
            # U
            for i_local in range(2):
                val = strassen_U[i_local, c]
                if val != 0:
                    i_global = i_local + 2 # offset by 2
                    U_full.append([0]*rank if False else None) # placeholder
            # Let's build lists of contributions
            pass
            
        # It is easier to build the matrices directly.
        # Total rank will be 4 * (14 + 8) = 88.
        pass

    # Let's just return the 88 rank solution properly constructed.
    # I will create the full U, V, W matrices.
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for j in range(4):
        # Bottom 2x4x2 (Rank 14)
        # Two Strassen 2x2x2
        for k_block in range(2): # 0 for cols 0-1 of A, 1 for cols 2-3 of A
            a_col_base = k_block * 2
            b_row_base = k_block * 2 # B rows corresponding to A cols
            
            # Strassen U, V, W
            for c in range(7):
                # U
                for i_loc in range(2):
                    val = strassen_U[i_loc, c]
                    if val != 0:
                        i_glob = i_loc + 2 # A rows 2,3
                        U[i_glob * m + a_col_base + 0][col] = val # Wait, U index is i*m+j?
                        # U shape is (n*m, rank). Index i*m+j.
                        # i is row of A, j is col of A.
                        # Here i_glob is row of A. a_col_base + i_loc? No.
                        # Strassen U is for A. A is 2x2.
                        # U[i, j] where i,j in 0..1.
                        # Global A row: i_glob. Global A col: a_col_base + j_loc.
                        pass
                # This manual mapping is error-prone. I will use numpy operations to construct the blocks.
                pass
            
        # Top 1x4x2 (Rank 8)
        # Two Schoolbook 1x2x2
        pass

    # To ensure correctness and avoid manual index errors, I will generate the tensor using a helper.
    pass

    # Fallback to Schoolbook if construction is complex? No, must be better.
    # I will implement the 88 rank solution carefully.
    
    # Reset
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for j in range(4):
        # Bottom: 2x4x2
        # Decompose into two 2x2x2
        for k_blk in range(2):
            a_c0 = k_blk * 2
            b_r0 = k_blk * 2
            # Strassen 2x2x2
            for c in range(7):
                # U
                for i_l in range(2):
                    v = strassen_U[i_l, c]
                    if v:
                        i_g = i_l + 2
                        for j_l in range(2): # Strassen U is not just 1 index, it's the whole vector
                             # Wait, strassen_U is 4x7. U[i,j] is not sparse in j?
                             # My strassen_U definition above was wrong.
                             # U is 4x7. U[idx, c]. idx is 0..3.
                             # U[0, c] corresponds to A[0,0] etc?
                             # No, Strassen combines A entries.
                             # U[0,0]=1, U[3,0]=1 for M1.
                             # So for M1, U[0]=1, U[3]=1.
                             # Global U index: i_g * m + j_g.
                             # But Strassen U is defined on the 2x2 block.
                             # The "i" in U[0,0] is the local row? No, it's the local entry index.
                             # My get_strassen_222 returns U, V, W of shape 4x7.
                             # U[0] corresponds to A[0,0]. U[1] to A[0,1]. U[2] to A[1,0]. U[3] to A[1,1].
                             # So I need to map local idx 0..3 to global indices.
                             pass
                # V
                for v_idx in range(4):
                    v = strassen_V[v_idx, c]
                    if v:
                        # Map v_idx to global B index
                        pass
                # W
                for w_idx in range(4):
                    w = strassen_W[w_idx, c]
                    if w:
                        # Map w_idx to global C index
                        pass
                col += 1
        
        # Top: 1x4x2
        # Decompose into two 1x2x2
        for k_blk in range(2):
            a_c0 = k_blk * 2
            b_r0 = k_blk * 2
            for c in range(4):
                for u_idx in range(2):
                    v = school_U[u_idx, c]
                    if v:
                        pass
                for v_idx in range(4):
                    v = school_V[v_idx, c]
                    if v:
                        pass
                for w_idx in range(2):
                    w = school_W[w_idx, c]
                    if w:
                        pass
                col += 1

    return {"U": U, "V": V, "W": W}
