import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper for Strassen 2x2x2
    # a_idx: list of 4 U indices for A[[0,1],[0,1]] -> [a00, a01, a10, a11]
    # b_idx: list of 4 V indices for B[[0,1],[0,1]] -> [b00, b01, b10, b11]
    # c_idx: list of 4 W indices for C[[0,1],[0,1]] -> [c00, c01, c10, c11]
    def add_strassen(a_idx, b_idx, c_idx):
        # M1 = (a00+a11)(b00+b11)
        add_term(a_idx[0], b_idx[0], c_idx[0], 1, 1, 1)
        add_term(a_idx[0], b_idx[3], c_idx[0], 1, 1, 1)
        add_term(a_idx[3], b_idx[0], c_idx[0], 1, 1, 1)
        add_term(a_idx[3], b_idx[3], c_idx[0], 1, 1, 1)
        
        # M2 = (a10+a11)b00
        add_term(a_idx[2], b_idx[0], c_idx[2], 1, 1, 1)
        add_term(a_idx[3], b_idx[0], c_idx[2], 1, 1, 1)
        
        # M3 = a00(b01-b11)
        add_term(a_idx[0], b_idx[1], c_idx[1], 1, 1, 1)
        add_term(a_idx[0], b_idx[3], c_idx[1], 1, -1, 1)
        
        # M4 = a11(b00-b10)
        add_term(a_idx[3], b_idx[0], c_idx[0], 1, 1, 1) # Wait, M4 contributes to c00, c01, c10, c11?
        # Let's re-evaluate Strassen contributions carefully.
        
    # Let's just hardcode the 7 terms and their contributions to c00, c01, c10, c11.
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M1 - M4 + M6
    # c10 = M2 + M4 - M7
    # c11 = M1 - M2 + M3 + M7
    
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # M2 = (a10+a11)b00 = a10b00 + a11b00
    # M3 = a00(b01-b11) = a00b01 - a00b11
    # M4 = a11(b00-b10) = a11b00 - a11b10
    # M5 = (a00+a01)b11 = a00b11 + a01b11
    # M6 = (a10-a00)(b00+b01) = a10b00 + a10b01 - a00b00 - a00b01
    # M7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11
    
    # This is getting complex to map to "add_term" which adds rank-1 tensors.
    # It is better to define the 7 multiplications M1..M7 as rank-1 tensors in U,V,W?
    # No, M1 is a product of sums.
    # M1 = (a00+a11)(b00+b11).
    # This corresponds to 4 rank-1 terms in the tensor decomposition of M1 itself?
    # No, M1 is one "scalar" in the Strassen algorithm.
    # In the tensor decomposition, each "multiply" is a rank-1 term in the U,V,W representation?
    # Yes.
    # U, V, W are such that C = sum_r U_r V_r W_r^T.
    # So each rank-1 term is defined by a column in U, V, W.
    # For M1, we need to compute (a00+a11) and (b00+b11).
    # This is NOT a rank-1 tensor.
    # Wait. The Strassen algorithm reduces the number of multiplications.
    # Each multiplication is of two linear combinations.
    # In the tensor decomposition, a rank-1 term is $u \otimes v \otimes w$.
    # If we have a term like $(a_{00}+a_{11})(b_{00}+b_{11}) c_{00}$, this is not rank 1.
    # It is sum of 4 rank 1 terms: $a_{00}b_{00}c_{00} + a_{00}b_{11}c_{00} + a_{11}b_{00}c_{00} + a_{11}b_{11}c_{00}$.
    # So Strassen does NOT reduce the rank of the tensor decomposition in this naive sense?
    # Ah. The rank of the tensor is the minimum number of rank-1 tensors.
    # Strassen shows that the multiplication tensor can be decomposed into 7 rank-1 tensors?
    # No, Strassen shows that you can compute the product with 7 scalar multiplications.
    # Each scalar multiplication is of linear forms.
    # $M_1 = L_A \cdot L_B$.
    # The contribution to $C$ is $M_1 \cdot L_C$.
    # $C_{ij} = \sum M_k L_{C,ij}^k$.
    # So $C = \sum_k (L_{A,k} \cdot L_{B,k}) L_{C,k}$.
    # This is a sum of 7 terms. Each term is a product of 3 linear forms?
    # No, $L_A$ is a vector in $A$-space, $L_B$ in $B$-space, $L_C$ in $C$-space.
    # The term is $L_A \otimes L_B \otimes L_C$.
    # This IS a rank-1 tensor!
    # So yes, Strassen gives a rank-7 decomposition for $2 \times 2 \times 2$.
    
    # So we just need to define the 7 linear forms for A, B, C for each block.
    
    # Let's refine the add_strassen function.
    # It should add 7 columns to U, V, W.
    
    def add_strassen_block(a_rows, b_rows, c_rows):
        # a_rows: [a00, a01, a10, a11] indices in U
        # b_rows: [b00, b01, b10, b11] indices in V
        # c_rows: [c00, c01, c10, c11] indices in W
        
        # M1: A=(1,0,0,1), B=(1,0,0,1), C=(1,1,0,1) -> c00=1, c01=1, c10=0, c11=1?
        # c00 += M1, c01 += M1, c11 += M1. c10 += 0.
        # Wait, c01 = M3 + M1 - M4 + M6. So c01 gets +M1.
        # c00 = M1 + M4 - M5 + M7. So c00 gets +M1.
        # c11 = M1 - M2 + M3 + M7. So c11 gets +M1.
        # c10 = M2 + M4 - M7. So c10 gets 0.
        
        # M1
        add_term(a_rows[0], b_rows[0], c_rows[0], 1, 1, 1)
        add_term(a_rows[0], b_rows[3], c_rows[0], 1, 1, 1)
        add_term(a_rows[3], b_rows[0], c_rows[0], 1, 1, 1)
        add_term(a_rows[3], b_rows[3], c_rows[0], 1, 1, 1)
        # This is WRONG. M1 is ONE rank-1 tensor in the Strassen decomposition.
        # The rank-1 tensor is defined by the linear forms.
        # U col: [1, 0, 0, 1] at a_rows.
        # V col: [1, 0, 0, 1] at b_rows.
        # W col: [1, 1, 0, 1] at c_rows.
        
        # Let's implement add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if u_val != 0: U[u_idx][col] += u_val
        if v_val != 0: V[v_idx][col] += v_val
        if w_val != 0: W[w_idx][col] += w_val
        col += 1

    def add_strassen(a_rows, b_rows, c_rows):
        # M1: A=(1,0,0,1), B=(1,0,0,1), C=(1,1,0,1)
        add_rank1(a_rows[0], 1, b_rows[0], 1, c_rows[0], 1)
        add_rank1(a_rows[0], 1, b_rows[3], 1, c_rows[0], 1) # No, this is expanding the linear form into basis vectors.
        # The linear form is a single vector.
        # U column is a vector.
        # So we just set the values in the column.
        
        # Let's do it properly.
        # M1
        U[a_rows[0]][col] += 1
        U[a_rows[3]][col] += 1
        V[b_rows[0]][col] += 1
        V[b_rows[3]][col] += 1
        W[c_rows[0]][col] += 1
        W[c_rows[1]][col] += 1
        W[c_rows[3]][col] += 1
        col += 1
        
        # M2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,0)
        U[a_rows[2]][col] += 1
        U[a_rows[3]][col] += 1
        V[b_rows[0]][col] += 1
        W[c_rows[2]][col] += 1
        col += 1
        
        # M3: A=(1,0,0,0), B=(0,1,0,-1), C=(0,1,0,0)
        U[a_rows[0]][col] += 1
        V[b_rows[1]][col] += 1
        V[b_rows[3]][col] += -1
        W[c_rows[1]][col] += 1
        col += 1
        
        # M4: A=(0,0,0,1), B=(1,0,-1,0), C=(1,1,1,0)
        U[a_rows[3]][col] += 1
        V[b_rows[0]][col] += 1
        V[b_rows[2]][col] += -1
        W[c_rows[0]][col] += 1
        W[c_rows[1]][col] += 1
        W[c_rows[2]][col] += 1
        col += 1
        
        # M5: A=(1,1,0,0), B=(0,0,0,1), C=(-1,0,0,0)
        U[a_rows[0]][col] += 1
        U[a_rows[1]][col] += 1
        V[b_rows[3]][col] += 1
        W[c_rows[0]][col] += -1
        col += 1
        
        # M6: A=(-1,0,1,0), B=(1,1,0,0), C=(0,1,0,0)
        U[a_rows[0]][col] += -1
        U[a_rows[2]][col] += 1
        V[b_rows[0]][col] += 1
        V[b_rows[1]][col] += 1
        W[c_rows[1]][col] += 1
        col += 1
        
        # M7: A=(0,1,-1,0), B=(0,0,1,1), C=(1,0,0,1)
        U[a_rows[1]][col] += 1
        U[a_rows[2]][col] += -1
        V[b_rows[2]][col] += 1
        V[b_rows[3]][col] += 1
        W[c_rows[0]][col] += 1
        W[c_rows[3]][col] += 1
        col += 1

    # Block 1: A_L2 (cols 0,1), B_T2 (rows 0,1)
    # A rows: 0,1,5,6
    # B rows: 0,1,6,7 (b00,b01,b10,b11) -> B[0,0], B[0,1], B[1,0], B[1,1]
    # C rows: 0,2,6,8? No.
    # C is 2x6.
    # c00 is C[0,0] -> W[0*2+0]=0
    # c01 is C[0,1] -> W[1*2+0]=2
    # c10 is C[1,0] -> W[0*2+1]=1
    # c11 is C[1,1] -> W[1*2+1]=3
    
    # We have 3 blocks of 2x2 in B for the 2x2x6.
    # B_T2 is 2x6.
    # Block 1.1: B cols 0,1.
    # Block 1.2: B cols 2,3.
    # Block 1.3: B cols 4,5.
    
    a_rows_1 = [0, 1, 5, 6]
    
    # 1.1
    b_rows_1_1 = [0, 1, 6, 7]
    c_rows_1_1 = [0, 2, 1, 3]
    add_strassen(a_rows_1, b_rows_1_1, c_rows_1_1)
    
    # 1.2
    b_rows_1_2 = [2, 3, 8, 9]
    c_rows_1_2 = [4, 6, 5, 7]
    add_strassen(a_rows_1, b_rows_1_2, c_rows_1_2)
    
    # 1.3
    b_rows_1_3 = [4, 5, 10, 11]
    c_rows_1_3 = [8, 10, 9, 11]
    add_strassen(a_rows_1, b_rows_1_3, c_rows_1_3)
    
    # Block 2: A_L1 (col 2), B_T1 (row 2)
    # A rows: 2, 7
    # B rows: 12..17
    # C rows: 0..11
    # Rank 12.
    # C[i,k] = A[i,2] B[2,k]
    # i in 0,1. k in 0..5.
    for i in range(2):
        for k in range(6):
            u_idx = 2 + i*5 # A[i,2] is row i, col 2. U index i*5+2.
            v_idx = 2*6 + k # B[2,k] is row 2, col k. V index 2*6+k.
            w_idx = k*2 + i # C[i,k] is row k, col i? No, C is 2x6. W index k*n+i.
            # W[k*n+i] = W[k*2+i].
            add_rank1(u_idx, 1, v_idx, 1, w_idx, 1)
            
    # Block 3: A_R (cols 3,4), B_B (rows 3,4)
    # A rows: 3,4,8,9
    a_rows_3 = [3, 4, 8, 9]
    
    # 3.1: B cols 0,1
    b_rows_3_1 = [18, 19, 24, 25]
    c_rows_3_1 = [0, 2, 1, 3]
    add_strassen(a_rows_3, b_rows_3_1, c_rows_3_1)
    
    # 3.2: B cols 2,3
    b_rows_3_2 = [20, 21, 26, 27]
    c_rows_3_2 = [4, 6, 5, 7]
    add_strassen(a_rows_3, b_rows_3_2, c_rows_3_2)
    
    # 3.3: B cols 4,5
    b_rows_3_3 = [22, 23, 28, 29]
    c_rows_3_3 = [8, 10, 9, 11]
    add_strassen(a_rows_3, b_rows_3_3, c_rows_3_3)
    
    return {"U": U, "V": V, "W": W}
