"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Rank 88 decomposition
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, u_val, v_val, w_val
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen decomposition for 2x2 by 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = AB
    # We need to map A indices to U, B indices to V, C indices to W.
    # This function assumes we are processing a specific 2x2 block of A and B.
    # It adds 7 terms to U, V, W.
    
    def add_strassen_2x2(a_base_u, b_base_v, c_base_w, a_vals, b_vals):
        # a_vals: [a11, a12, a21, a22]
        # b_vals: [b11, b12, b21, b22]
        # a_base_u: (row_offset_in_U, col_offset_in_U) ? No, U is flat.
        # U index for A element (r, c) in the 2x2 block:
        # The 2x2 block is part of a larger structure.
        # We will pass the exact U, V, W indices for the 4 elements of A, B, and C.
        pass

    # Since the structure is fixed, we can hardcode the indices or calculate them.
    # A is 3x4. Split into A_L (3x2) and A_R (3x2).
    # A_L cols 0,1. A_R cols 2,3.
    # B is 4x8. Split into B_T (2x8) and B_B (2x8).
    # B_T rows 0,1. B_B rows 2,3.
    
    # Term 1: A_L B_T
    # A_L is 3x2. Rows 0,1,2. Cols 0,1.
    # B_T is 2x8. Rows 0,1. Cols 0..7.
    # Split B_T into 4 blocks of 2x2: B_T1 (cols 0,1), B_T2 (cols 2,3), B_T3 (cols 4,5), B_T4 (cols 6,7).
    
    # For each block, we compute A_L * B_Tk.
    # A_L * B_Tk is 3x2.
    # Split A_L into A_L_top (2x2, rows 0,1) and A_L_bot (1x2, row 2).
    # C_top = A_L_top * B_Tk (2x2). Uses Strassen (7 terms).
    # C_bot = A_L_bot * B_Tk (1x2). Uses 4 terms.
    
    # Indices for A_L_top in U:
    # U index for A[i, j] is i*m + j = i*4 + j.
    # A_L_top: i in [0,1], j in [0,1].
    # a11 -> U[0*4+0] = U[0]
    # a12 -> U[0*4+1] = U[1]
    # a21 -> U[1*4+0] = U[4]
    # a22 -> U[1*4+1] = U[5]
    
    # Indices for B_Tk in V:
    # V index for B[j, k] is j*p + k = j*8 + k.
    # B_Tk: j in [0,1], k in [start, start+1].
    
    # Indices for C_top in W:
    # W index for C[i, k] is k*n + i = k*3 + i.
    # C_top: i in [0,1], k in [start, start+1].
    
    # Indices for A_L_bot in U:
    # A_L_bot: i=2, j in [0,1].
    # a11_bot -> U[2*4+0] = U[8]
    # a12_bot -> U[2*4+1] = U[9]
    
    # Indices for C_bot in W:
    # C_bot: i=2, k in [start, start+1].
    
    # Let's define the blocks for B_T
    # Block 0: k in [0,1]
    # Block 1: k in [2,3]
    # Block 2: k in [4,5]
    # Block 3: k in [6,7]
    
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # --- Strassen part (A_L_top * B_T_block) ---
        # A_L_top indices in U:
        u_a11 = 0
        u_a12 = 1
        u_a21 = 4
        u_a22 = 5
        
        # B_T_block indices in V:
        v_b11 = 0 * 8 + k_start
        v_b12 = 0 * 8 + k_start + 1
        v_b21 = 1 * 8 + k_start
        v_b22 = 1 * 8 + k_start + 1
        
        # C_top indices in W:
        w_c11 = k_start * 3 + 0
        w_c12 = (k_start + 1) * 3 + 0
        w_c21 = k_start * 3 + 1
        w_c22 = (k_start + 1) * 3 + 1
        
        # Strassen M1: (a11+a22)(b11+b22)
        add_term(u_a11, v_b11, w_c11, 1, 1, 1)
        add_term(u_a11, v_b22, w_c11, 1, 1, 1)
        add_term(u_a22, v_b11, w_c11, 1, 1, 1)
        add_term(u_a22, v_b22, w_c11, 1, 1, 1)
        # Wait, M1 is a product of sums. It's not a single rank-1 term in the tensor sense if we don't expand.
        # The tensor decomposition requires U, V, W to be linear in A, B, C.
        # M1 = (a11+a22)(b11+b22).
        # This corresponds to 4 rank-1 terms in the tensor if we expand?
        # No, the rank of the multiplication tensor is the minimum number of rank-1 tensors.
        # Strassen's algorithm says we can do it with 7 "multiplications" which are bilinear forms.
        # Each bilinear form $x y$ corresponds to a rank-1 term in the decomposition if we view it as $u \cdot v \cdot w$.
        # Actually, the term $(a_{11}+a_{22})(b_{11}+b_{22})$ is not a single rank-1 term of the form $u_{ij} v_{jk} w_{ki}$.
        # It is a sum of 4 rank-1 terms?
        # Let's check.
        # The tensor is $T_{ijk} = \delta_{ik} \delta_{jl}$? No.
        # The multiplication tensor is $\sum_{i,j,k} A_{ij} B_{jk} C_{ki}$.
        # A rank-1 decomposition is $\sum_{r} u_r \otimes v_r \otimes w_r$.
        # $u_r$ is a vector in $\mathbb{F}^{nm}$, $v_r$ in $\mathbb{F}^{mp}$, $w_r$ in $\mathbb{F}^{pn}$.
        # The term $u_r \otimes v_r \otimes w_r$ contributes $u_{ij} v_{jk} w_{ki}$ to the coefficient of $A_{ij} B_{jk} C_{ki}$.
        # Strassen's $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This bilinear form can be represented as a sum of rank-1 tensors?
        # $(A_{11}+A_{22})(B_{11}+B_{22}) = A_{11}B_{11} + A_{11}B_{22} + A_{22}B_{11} + A_{22}B_{22}$.
        # Each $A_{ij}B_{jk}$ is a rank-1 tensor (if we fix the output indices appropriately).
        # But Strassen claims rank 7. This means the 7 bilinear forms are linearly independent and span the space.
        # However, each bilinear form like $A_{11}B_{11}$ is NOT a rank-1 term of the full multiplication tensor unless it contributes to exactly one output element $C_{kl}$.
        # In Strassen, the output $C$ is a linear combination of the $M$'s.
        # $C_{11} = M_1 + M_4 - M_5 + M_7$.
        # So the contribution to $C_{11}$ from $M_1$ is $M_1$.
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
        # This term appears in $C_{11}$, $C_{22}$ (with -1), etc.
        # So for each $M_r$, we need to create a rank-1 term?
        # No, $M_r$ is a scalar computed from A and B.
        # The decomposition is $\sum_r M_r \cdot (linear combo of C_{kl})$.
        # This is not a standard rank-1 decomposition of the tensor $\sum A_{ij} B_{jk} C_{ki}$.
        # The standard rank-1 decomposition is $\sum_r u_r \otimes v_r \otimes w_r$.
        # This implies $C_{ki} = \sum_r u_{ij} v_{jk} w_{ki}$.
        # In Strassen, $C_{ki}$ is a linear combination of the $M$'s.
        # $M_r$ are not $u_r v_r$. They are bilinear forms.
        # To convert Strassen to a rank-1 decomposition, we need to express each $M_r$ as a sum of rank-1 terms?
        # No, the rank of the tensor is the minimum $r$ such that $T = \sum_{l=1}^r u_l \otimes v_l \otimes w_l$.
        # Strassen's algorithm uses 7 bilinear multiplications.
        # Does this imply the rank is 7?
        # Yes, if the bilinear forms can be embedded into the rank-1 structure.
        # Actually, the rank of $2 \times 2$ matrix multiplication is 7.
        # The decomposition is:
        # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$
        # ...
        # $C_{11} = M_1 + M_4 - M_5 + M_7$
        # This is a valid rank-7 decomposition.
        # How?
        # Each $M_r$ is a product of two linear forms.
        # $M_r = (\sum a_{ij} \alpha_{ij}) (\sum b_{jk} \beta_{jk})$.
        # This is not a single rank-1 term $u \otimes v \otimes w$.
        # A rank-1 term is $u_{ij} v_{jk} w_{ki}$.
        # The product of two linear forms is a sum of rank-1 terms?
        # $(\sum \alpha_{ij} A_{ij}) (\sum \beta_{jk} B_{jk}) = \sum_{ij, jk} \alpha_{ij} \beta_{jk} A_{ij} B_{jk}$.
        # This requires $w_{ki}$ to be non-zero for all $k,i$ where the product contributes?
        # No, the output $C_{ki}$ must be recovered.
        # In the rank-1 decomposition, $C_{ki} = \sum_r u^{(r)}_{ij} v^{(r)}_{jk} w^{(r)}_{ki}$.
        # For Strassen, we define 7 sets of coefficients $(u^{(r)}, v^{(r)}, w^{(r)})$.
        # It turns out that Strassen's 7 multiplications correspond to 7 rank-1 terms.
        # Specifically, for each $r$, we define:
        # $u^{(r)}_{ij} = \alpha^{(r)}_{ij}$
        # $v^{(r)}_{jk} = \beta^{(r)}_{jk}$
        # $w^{(r)}_{ki} = \gamma^{(r)}_{ki}$
        # Such that $\sum_r u^{(r)}_{ij} v^{(r)}_{jk} w^{(r)}_{ki} = A_{ij} B_{jk}$? No.
        # The identity is $A B = \sum_r M_r \cdot C_r$ where $C_r$ is a fixed matrix?
        # No, $C = \sum_r M_r \cdot E_r$ where $E_r$ are fixed matrices?
        # Yes, $C_{ki} = \sum_r M_r E^{(r)}_{ki}$.
        # And $M_r = \sum_{ij, jk} \alpha^{(r)}_{ij} \beta^{(r)}_{jk} A_{ij} B_{jk}$.
        # This is NOT of the form $\sum_r u^{(r)}_{ij} v^{(r)}_{jk} w^{(r)}_{ki}$.
        # The standard rank-1 form requires the coefficient of $A_{ij} B_{jk}$ in $C_{ki}$ to be $u^{(r)}_{ij} v^{(r)}_{jk} w^{(r)}_{ki}$.
        # In Strassen, the coefficient of $A_{ij} B_{jk}$ in $C_{ki}$ is $\sum_r \alpha^{(r)}_{ij} \beta^{(r)}_{jk} E^{(r)}_{ki}$.
        # This is a sum of products of 3 factors, but the factors depend on $r$ and the indices.
        # Is this a rank-1 decomposition?
        # Yes, if we define $u^{(r)}_{ij} = \alpha^{(r)}_{ij}$, $v^{(r)}_{jk} = \beta^{(r)}_{jk}$, and $w^{(r)}_{ki} = E^{(r)}_{ki}$.
        # Then $\sum_r u^{(r)}_{ij} v^{(r)}_{jk} w^{(r)}_{ki} = \sum_r \alpha^{(r)}_{ij} \beta^{(r)}_{jk} E^{(r)}_{ki}$.
        # This matches the Strassen formula.
        # So yes, Strassen's 7 multiplications correspond to 7 rank-1 terms.
        
        # So we can just add 7 terms for each 2x2 block.
        
        # Strassen Coefficients:
        # M1: A: [1,0,0,1], B: [1,0,0,1], C: [1,0,0,1] (C11=1, C12=0, C21=0, C22=1? No)
        # C11 gets +1 from M1. C22 gets -1 from M1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's list the 7 terms (u, v, w) for the 2x2 block.
        # u is 4-dim (a11,a12,a21,a22), v is 4-dim (b11,b12,b21,b22), w is 4-dim (c11,c12,c21,c22).
        
        # Term 1 (M1):
        # u = [1, 0, 1, 0] (a11+a22)
        # v = [1, 0, 0, 1] (b11+b22)
        # w = [1, 0, 0, 1] (c11+1, c22-1? No, w is the coefficient for C)
        # C11 += 1, C22 += -1?
        # Formula: C11 = ... + M1 ...
        # C22 = ... + M1 ...
        # So w = [1, 0, 0, 1].
        
        # Term 2 (M2):
        # u = [0, 0, 1, 1] (a21+a22)
        # v = [1, 0, 0, 0] (b11)
        # w = [0, 0, 1, 0] (c21)
        
        # Term 3 (M3):
        # u = [1, 0, 0, 0] (a11)
        # v = [0, 1, 0, -1] (b12-b22)
        # w = [0, 1, 0, -1] (c12+1, c22-1)
        
        # Term 4 (M4):
        # u = [0, 0, 0, 1] (a22)
        # v = [0, 0, 1, -1] (b21-b11)
        # w = [1, 0, 1, 0] (c11+1, c21+1)
        
        # Term 5 (M5):
        # u = [1, 1, 0, 0] (a11+a12)
        # v = [0, 0, 0, 1] (b22)
        # w = [-1, 1, 0, 0] (c11-1, c12+1)
        
        # Term 6 (M6):
        # u = [0, 0, 1, -1] (a21-a11)
        # v = [1, 1, 0, 0] (b11+b12)
        # w = [0, 0, 0, 1] (c22)
        
        # Term 7 (M7):
        # u = [0, 1, 0, -1] (a12-a22)
        # v = [0, 0, 1, 1] (b21+b22)
        # w = [1, 0, 0, -1] (c11+1, c22-1)
        
        # We will add these 7 terms.
        
        # Mapping local indices to global U, V, W indices.
        # Local A indices: 0:a11, 1:a12, 2:a21, 3:a22
        # Global U indices for A_L_top:
        # a11 -> U[0], a12 -> U[1], a21 -> U[4], a22 -> U[5]
        
        # Local B indices: 0:b11, 1:b12, 2:b21, 3:b22
        # Global V indices for B_T_block:
        # b11 -> V[0*8+k_start], b12 -> V[0*8+k_start+1]
        # b21 -> V[1*8+k_start], b22 -> V[1*8+k_start+1]
        
        # Local C indices: 0:c11, 1:c12, 2:c21, 3:c22
        # Global W indices for C_top:
        # c11 -> W[k_start*3+0], c12 -> W[(k_start+1)*3+0]
        # c21 -> W[k_start*3+1], c22 -> W[(k_start+1)*3+1]
        
        # Define the 7 terms
        strassen_terms = [
            ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
        ]
        
        u_map = [0, 1, 4, 5]
        v_map = [0*8+k_start, 0*8+k_start+1, 1*8+k_start, 1*8+k_start+1]
        w_map = [k_start*3+0, (k_start+1)*3+0, k_start*3+1, (k_start+1)*3+1]
        
        for u_vals, v_vals, w_vals in strassen_terms:
            # This is not a single rank-1 term!
            # The term is $\sum_{a,b,c} u_a v_b w_c A_a B_b C_c$.
            # This is a sum of $4 \times 4 \times 4 = 64$ rank-1 terms?
            # No, the rank-1 term in the decomposition is $u \otimes v \otimes w$.
            # The entry $U_{ij, r} = u_i$, $V_{jk, r} = v_j$, $W_{ki, r} = w_k$.
            # Wait, the indices are different.
            # In the tensor $T_{ijk}$, $i$ is row of A, $j$ is col of A/row of B, $k$ is col of B.
            # $U$ index is $(i, j)$. $V$ index is $(j, k)$. $W$ index is $(k, i)$.
            # For the 2x2 block, we have local indices $i, j, k \in \{0,1\}$.
            # The term contributes $u_i v_j w_k$ to $C_{ki}$.
            # So for each local $i, j, k$, we add $u_i v_j w_k$ to $W_{global(k,i), r}$.
            # And we set $U_{global(i,j), r} = u_i$? No.
            # $U$ is a matrix. $U_{(i,j), r}$.
            # In the sum $\sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$,
            # if we define $U_{(i,j), r} = u_i$ (independent of j?), that's not right.
            # The vector $u_r$ has components for each $(i,j)$ pair.
            # In Strassen, $M_r = (\sum_{ij} \alpha_{ij} A_{ij}) (\sum_{jk} \beta_{jk} B_{jk})$.
            # So $U_{(i,j), r} = \alpha_{ij}$.
            # $V_{(j,k), r} = \beta_{jk}$.
            # $W_{(k,i), r} = \gamma_{ki}$.
            # Then $\sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = \sum_r \alpha_{ij} \beta_{jk} \gamma_{ki}$.
            # This matches the Strassen expansion.
            
            # So for each term r:
            # We iterate through local i, j, k.
            # But u_vals is indexed by local A element (0..3).
            # Local A element 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
            # So u_vals[0] corresponds to i=0, j=0.
            # u_vals[1] corresponds to i=0, j=1.
            # u_vals[2] corresponds to i=1, j=0.
            # u_vals[3] corresponds to i=1, j=1.
            
            # Similarly for v_vals (b11, b12, b21, b22).
            # v_vals[0] -> j=0, k=0.
            # v_vals[1] -> j=0, k=1.
            # v_vals[2] -> j=1, k=0.
            # v_vals[3] -> j=1, k=1.
            
            # w_vals (c11, c12, c21, c22).
            # w_vals[0] -> k=0, i=0.
            # w_vals[1] -> k=1, i=0.
            # w_vals[2] -> k=0, i=1.
            # w_vals[3] -> k=1, i=1.
            
            # We need to add 1 term to the rank decomposition for this Strassen multiplication?
            # No, each Strassen multiplication $M_r$ corresponds to ONE rank-1 term in the decomposition?
            # No, $M_r$ is a scalar. The decomposition is $\sum_r M_r \dots$
            # The rank-1 term is defined by the vectors $u_r, v_r, w_r$.
            # So yes, each Strassen multiplication adds 1 column to U, V, W.
            
            # So we do NOT iterate over i,j,k to add multiple terms.
            # We add ONE term (one column) with the values defined by the vectors.
            
            # For this column:
            # U[global(i,j)] = u_vals[local_a_idx(i,j)]
            # V[global(j,k)] = v_vals[local_b_idx(j,k)]
            # W[global(k,i)] = w_vals[local_c_idx(k,i)]
            
            # local_a_idx:
            # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # local_b_idx:
            # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # local_c_idx:
            # (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
            
            for li in range(2):
                for lj in range(2):
                    la = li * 2 + lj
                    global_u = u_map[la]
                    U[global_u][col] = u_vals[la]
                    
                    # V and W depend on lj and lk?
                    # V index is (j, k). j is lj. k is local k.
                    # W index is (k, i). k is local k. i is li.
                    
                    # We need to fill V and W for all k.
                    # But V and W values are constant for a given j/k or k/i?
                    # No, v_vals depends on (j, k).
                    # So for fixed li, lj, we iterate lk?
                    # No, the term is a single tensor product.
                    # $U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$.
                    # So for a fixed r, U, V, W are fixed vectors.
                    
                    # So we just assign the values.
                    pass
            
            # Correct approach:
            # For this column (r):
            # Assign U values for all 4 A elements.
            for la in range(4):
                U[u_map[la]][col] = u_vals[la]
                
            # Assign V values for all 4 B elements.
            for lb in range(4):
                V[v_map[lb]][col] = v_vals[lb]
                
            # Assign W values for all 4 C elements.
            for lc in range(4):
                W[w_map[lc]][col] = w_vals[lc]
                
            col += 1
            
        # --- 1x2 part (A_L_bot * B_T_block) ---
        # A_L_bot is 1x2. a11_bot (row 2, col 0), a12_bot (row 2, col 1).
        # U indices: 8, 9.
        # B_T_block is 2x2.
        # C_bot is 1x2. c11_bot (row 2, col k_start), c12_bot (row 2, col k_start+1).
        # W indices: k_start*3+2, (k_start+1)*3+2.
        
        # 4 terms:
        # 1. a11_bot * b11 -> c11_bot
        # 2. a12_bot * b21 -> c11_bot
        # 3. a11_bot * b12 -> c12_bot
        # 4. a12_bot * b22 -> c12_bot
        
        # Term 1
        U[8][col] = 1
        U[9][col] = 0
        V[v_map[0]][col] = 1 # b11
        V[v_map[1]][col] = 0
        V[v_map[2]][col] = 0
        V[v_map[3]][col] = 0
        W[w_map[2]][col] = 1 # c11_bot is W[k_start*3+2]
        W[w_map[3]][col] = 0
        col += 1
        
        # Term 2
        U[8][col] = 0
        U[9][col] = 1
        V[v_map[0]][col] = 0
        V[v_map[1]][col] = 0
        V[v_map[2]][col] = 1 # b21
        V[v_map[3]][col] = 0
        W[w_map[2]][col] = 1
        W[w_map[3]][col] = 0
        col += 1
        
        # Term 3
        U[8][col] = 1
        U[9][col] = 0
        V[v_map[0]][col] = 0
        V[v_map[1]][col] = 1 # b12
        V[v_map[2]][col] = 0
        V[v_map[3]][col] = 0
        W[w_map[2]][col] = 0
        W[w_map[3]][col] = 1 # c12_bot
        col += 1
        
        # Term 4
        U[8][col] = 0
        U[9][col] = 1
        V[v_map[0]][col] = 0
        V[v_map[1]][col] = 0
        V[v_map[2]][col] = 0
        V[v_map[3]][col] = 1 # b22
        W[w_map[2]][col] = 0
        W[w_map[3]][col] = 1
        col += 1

    # Repeat for A_R B_B
    # A_R is cols 2,3.
    # A_R_top: rows 0,1, cols 2,3.
    # U indices:
    # a11 -> U[0*4+2] = 2
    # a12 -> U[0*4+3] = 3
    # a21 -> U[1*4+2] = 6
    # a22 -> U[1*4+3] = 7
    # A_R_bot: row 2, cols 2,3.
    # U indices: 10, 11.
    
    # B_B is rows 2,3.
    # V indices:
    # b11 -> V[2*8+k_start]
    # b12 -> V[2*8+k_start+1]
    # b21 -> V[3*8+k_start]
    # b22 -> V[3*8+k_start+1]
    
    # C is same W indices.
    
    for block_idx in range(4):
        k_start = block_idx * 2
        
        u_map_r = [2, 3, 6, 7]
        v_map_r = [2*8+k_start, 2*8+k_start+1, 3*8+k_start, 3*8+k_start+1]
        w_map_r = [k_start*3+0, (k_start+1)*3+0, k_start*3+1, (k_start+1)*3+1]
        
        for u_vals, v_vals, w_vals in strassen_terms:
            for la in range(4):
                U[u_map_r[la]][col] = u_vals[la]
            for lb in range(4):
                V[v_map_r[lb]][col] = v_vals[lb]
            for lc in range(4):
                W[w_map_r[lc]][col] = w_vals[lc]
            col += 1
            
        # 1x2 part
        U_bot_r = [10, 11]
        
        # Term 1
        U[10][col] = 1
        U[11][col] = 0
        V[v_map_r[0]][col] = 1
        V[v_map_r[1]][col] = 0
        V[v_map_r[2]][col] = 0
        V[v_map_r[3]][col] = 0
        W[w_map_r[2]][col] = 1
        W[w_map_r[3]][col] = 0
        col += 1
        
        # Term 2
        U[10][col] = 0
        U[11][col] = 1
        V[v_map_r[0]][col] = 0
        V[v_map_r[1]][col] = 0
        V[v_map_r[2]][col] = 1
        V[v_map_r[3]][col] = 0
        W[w_map_r[2]][col] = 1
        W[w_map_r[3]][col] = 0
        col += 1
        
        # Term 3
        U[10][col] = 1
        U[11][col] = 0
        V[v_map_r[0]][col] = 0
        V[v_map_r[1]][col] = 1
        V[v_map_r[2]][col] = 0
        V[v_map_r[3]][col] = 0
        W[w_map_r[2]][col] = 0
        W[w_map_r[3]][col] = 1
        col += 1
        
        # Term 4
        U[10][col] = 0
        U[11][col] = 1
        V[v_map_r[0]][col] = 0
        V[v_map_r[1]][col] = 0
        V[v_map_r[2]][col] = 0
        V[v_map_r[3]][col] = 1
        W[w_map_r[2]][col] = 0
        W[w_map_r[3]][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
