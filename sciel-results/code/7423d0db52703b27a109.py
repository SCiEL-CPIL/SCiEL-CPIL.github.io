import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We will build a decomposition of rank 65.
    # Part 1: Rank 49 for the first 4 columns of the result (4x4x4 multiplication via Strassen)
    # Part 2: Rank 16 for the 5th column of the result (4x4x1 multiplication via Schoolbook)
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # --- Part 1: 4x4x4 Strassen ---
    
    # Helper to get global indices for a 2x2 block
    # block (br, bc) where br, bc in {0,1}
    # A is 4x4. A[br*2 + r, bc*2 + c]
    # B is 4x4 (first 4 cols). B[br*2 + r, bc*2 + c]
    # C is 4x4 (first 4 cols). C[br*2 + r, bc*2 + c]
    
    def get_block_indices(br, bc):
        idxs = []
        for r in range(2):
            for c in range(2):
                idxs.append((br*2 + r, bc*2 + c))
        return idxs

    # Strassen 2x2x2 decomposition
    # Inputs: A_block_idx, B_block_idx
    # These are lists of 4 tuples (row, col)
    # Returns: list of 7 tuples (u_dict, v_dict, w_dict)
    # u_dict maps A_global_idx to coeff
    # v_dict maps B_global_idx to coeff
    # w_dict maps C_local_idx (0,1,2,3) to coeff (we will map to global later)
    
    def strassen_2x2(A_idx, B_idx):
        # A_idx[0]=00, [1]=01, [2]=10, [3]=11
        # B_idx[0]=00, [1]=01, [2]=10, [3]=11
        
        # M1 = (A00+A11)(B00+B11)
        u1 = {A_idx[0]: 1, A_idx[3]: 1}
        v1 = {B_idx[0]: 1, B_idx[3]: 1}
        
        # M2 = (A10+A11)B00
        u2 = {A_idx[2]: 1, A_idx[3]: 1}
        v2 = {B_idx[0]: 1}
        
        # M3 = A00(B01-B11)
        u3 = {A_idx[0]: 1}
        v3 = {B_idx[1]: 1, B_idx[3]: -1}
        
        # M4 = A11(B10-B00)
        u4 = {A_idx[3]: 1}
        v4 = {B_idx[2]: 1, B_idx[0]: -1}
        
        # M5 = (A00+A01)B11
        u5 = {A_idx[0]: 1, A_idx[1]: 1}
        v5 = {B_idx[3]: 1}
        
        # M6 = (A10-A00)(B00+B01)
        u6 = {A_idx[2]: 1, A_idx[0]: -1}
        v6 = {B_idx[0]: 1, B_idx[1]: 1}
        
        # M7 = (A01-A11)(B10+B11)
        u7 = {A_idx[1]: 1, A_idx[3]: -1}
        v7 = {B_idx[2]: 1, B_idx[3]: 1}
        
        # C00 = M1 + M4 - M5 + M7
        w1 = {0: 1, 3: 1} # C00 gets M1, M4? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We return the 7 multiplications.
        # The caller will combine them to form the C blocks.
        # Actually, it's better to return the contributions to C directly.
        # But C is a sum of products.
        # Product r is U_r * V_r.
        # C = sum W_r * (U_r * V_r)
        
        # So we need to output 7 terms.
        # Term 1: u1, v1, w_contrib
        # The w_contrib is the coefficient of this product in the final C.
        # But Strassen computes M1..M7, and then C is a linear combo of M1..M7.
        # So M1 appears in C00 and C11.
        # This means the "term" M1 is not a single rank-1 term in the tensor decomposition?
        # No, a rank-1 term is $u \otimes v \otimes w$.
        # If $M_1 = u_1 \cdot v_1$, and $C_{00} = M_1 + ...$, then $C_{00} = u_1 v_1 + ...$
        # The term $u_1 \otimes v_1 \otimes e_{00}$ is a valid rank-1 term contributing to C00.
        # The term $u_1 \otimes v_1 \otimes e_{11}$ is a valid rank-1 term contributing to C11.
        # So M1 generates TWO rank-1 terms?
        # No, the rank of the tensor is the number of terms in the sum.
        # $C = \sum_{r=1}^7 M_r \otimes W_r$? No.
        # $C_{ij} = \sum_k M_k \cdot c_{ijk}$.
        # This is not a sum of rank-1 tensors $u \otimes v \otimes w$.
        # A rank-1 tensor is $u \otimes v \otimes w$ where $C_{ijk} = u_i v_j w_k$.
        # In Strassen, $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This is a scalar product of a vector in A-space and a vector in B-space.
        # Let $u^{(1)}$ be the vector in A-space corresponding to $A_{00}+A_{11}$.
        # Let $v^{(1)}$ be the vector in B-space corresponding to $B_{00}+B_{11}$.
        # Then $M_1 = u^{(1)} \cdot A \cdot v^{(1)} \cdot B$? No, $M_1 = (u^{(1)} \cdot A) (v^{(1)} \cdot B)$.
        # This is a product of two linear forms.
        # This corresponds to the rank-1 tensor $u^{(1)} \otimes v^{(1)} \otimes w^{(1)}$?
        # No, the output is a matrix C.
        # $C = M_1 W^{(1)} + M_2 W^{(2)} + ...$
        # Where $W^{(1)}$ is the matrix with 1s at positions where $M_1$ contributes.
        # $W^{(1)}$ has 1 at C00 and C11.
        # Is $u^{(1)} \otimes v^{(1)} \otimes W^{(1)}$ a valid term?
        # The term is $\sum_{i,j,k} U_{ij,r} V_{jk,r} W_{k,i,r}$.
        # Here, for a fixed r, $U_{ij,r}$ and $V_{jk,r}$ are fixed.
        # $M_r = \sum_{ij} U_{ij,r} A_{ij} \sum_{jk} V_{jk,r} B_{jk}$.
        # Then $C_{ik} = \sum_r M_r W_{ik,r}$.
        # Yes, this is the definition.
        # So for each Strassen multiplication $M_r$, we define $U_r, V_r$.
        # And $W_r$ is the matrix of coefficients for that $M_r$ in the final C.
        
        # So for M1:
        # U1 is the vector for $A_{00}+A_{11}$.
        # V1 is the vector for $B_{00}+B_{11}$.
        # W1 is the matrix with 1 at C00 and C11.
        
        # So we can just return the 7 (U, V, W) tuples for the 2x2x2 case.
        
        res = []
        
        # M1
        w1_mat = {0: 1, 3: 1} # C00, C11
        res.append((u1, v1, w1_mat))
        
        # M2
        w2_mat = {2: 1, 3: -1} # C10, C11
        res.append((u2, v2, w2_mat))
        
        # M3
        w3_mat = {1: 1, 3: 1} # C01, C11
        res.append((u3, v3, w3_mat))
        
        # M4
        w4_mat = {0: 1, 2: 1} # C00, C10
        res.append((u4, v4, w4_mat))
        
        # M5
        w5_mat = {0: -1, 1: 1} # C00, C01
        res.append((u5, v5, w5_mat))
        
        # M6
        w6_mat = {3: 1} # C11
        res.append((u6, v6, w6_mat))
        
        # M7
        w7_mat = {0: 1} # C00
        res.append((u7, v7, w7_mat))
        
        return res

    # Now apply to 4x4x4
    # Top level blocks
    A_blocks = [get_block_indices(0,0), get_block_indices(0,1), get_block_indices(1,0), get_block_indices(1,1)]
    B_blocks = [get_block_indices(0,0), get_block_indices(0,1), get_block_indices(1,0), get_block_indices(1,1)]
    
    # Top level Strassen
    # P1 = (A00+A11)(B00+B11)
    # We need to form the linear combinations of A and B for the inputs to the 2x2x2 Strassen.
    
    def combine_blocks(block_idxs, coeffs):
        # block_idxs: list of 4 lists of indices
        # coeffs: list of 4 coeffs
        # Returns dict of global index -> coeff
        d = {}
        for i in range(4):
            for idx in block_idxs[i]:
                d[idx] = d.get(idx, 0) + coeffs[i]
        return d

    # P1: A = A00 + A11, B = B00 + B11
    A_P1 = combine_blocks(A_blocks, [1, 0, 0, 1])
    B_P1 = combine_blocks(B_blocks, [1, 0, 0, 1])
    
    # We need to pass the indices to strassen_2x2.
    # strassen_2x2 expects A_idx and B_idx as lists of 4 indices.
    # But A_P1 is a dict of global indices.
    # We need to map the local 2x2 indices of the "virtual" matrix to global indices.
    # The virtual matrix X has entries X00, X01, X10, X11.
    # X00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # X01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # X10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # X11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    
    # So we need to provide to strassen_2x2 the global indices corresponding to X00, X01, X10, X11.
    # But X00 is a sum of two global indices.
    # So we cannot just pass a list of 4 indices. We must pass the linear combination.
    
    # Let's modify strassen_2x2 to accept dicts (linear combinations) for A and B.
    
    def strassen_2x2_lin(A_lin, B_lin):
        # A_lin: dict {global_idx: coeff} representing the 2x2 matrix entries?
        # No, A_lin is the linear combination for the WHOLE 2x2 block?
        # No, Strassen needs the 4 entries of the 2x2 matrix separately.
        # So A_lin should be a dict of 4 entries: {0: dict, 1: dict, 2: dict, 3: dict}
        # where 0=00, 1=01, 2=10, 3=11.
        
        # M1 = (A00+A11)(B00+B11)
        # A00+A11 is the sum of the dicts for entry 0 and 3.
        
        def add_dicts(d1, d2):
            res = d1.copy()
            for k, v in d2.items():
                res[k] = res.get(k, 0) + v
            return res

        def sub_dicts(d1, d2):
            res = d1.copy()
            for k, v in d2.items():
                res[k] = res.get(k, 0) - v
            return res

        A00, A01, A10, A11 = A_lin[0], A_lin[1], A_lin[2], A_lin[3]
        B00, B01, B10, B11 = B_lin[0], B_lin[1], B_lin[2], B_lin[3]
        
        u1 = add_dicts(A00, A11)
        v1 = add_dicts(B00, B11)
        w1_mat = {0: 1, 3: 1}
        
        u2 = add_dicts(A10, A11)
        v2 = B00
        w2_mat = {2: 1, 3: -1}
        
        u3 = A00
        v3 = sub_dicts(B01, B11)
        w3_mat = {1: 1, 3: 1}
        
        u4 = A11
        v4 = sub_dicts(B10, B00)
        w4_mat = {0: 1, 2: 1}
        
        u5 = add_dicts(A00, A01)
        v5 = B11
        w5_mat = {0: -1, 1: 1}
        
        u6 = sub_dicts(A10, A00)
        v6 = add_dicts(B00, B01)
        w6_mat = {3: 1}
        
        u7 = sub_dicts(A01, A11)
        v7 = add_dicts(B10, B11)
        w7_mat = {0: 1}
        
        return [
            (u1, v1, w1_mat),
            (u2, v2, w2_mat),
            (u3, v3, w3_mat),
            (u4, v4, w4_mat),
            (u5, v5, w5_mat),
            (u6, v6, w6_mat),
            (u7, v7, w7_mat)
        ]

    # Prepare A_lin and B_lin for the 7 top-level products
    # A_lin is a list of 4 dicts, one for each 2x2 entry of the virtual matrix.
    
    def get_virtual_block_lin(block_idxs, coeffs):
        # block_idxs: list of 4 lists of global indices
        # coeffs: list of 4 coeffs
        # Returns list of 4 dicts (for 00, 01, 10, 11)
        res = [{} for _ in range(4)]
        for i in range(4):
            c = coeffs[i]
            if c == 0: continue
            # block_idxs[i] is a list of 4 global indices for the i-th 2x2 block
            # We need to distribute these to the 4 entries of the virtual block?
            # No, the virtual block IS the sum of the physical blocks.
            # So virtual entry 00 = sum of physical block entries at (0,0).
            # Physical block i has entries at indices block_idxs[i][0..3].
            # So virtual entry 00 gets block_idxs[i][0] with coeff c.
            # virtual entry 01 gets block_idxs[i][1] with coeff c.
            # etc.
            for j in range(4):
                idx = block_idxs[i][j]
                res[j][idx] = res[j].get(idx, 0) + c
        return res

    # P1: (A00+A11)(B00+B11)
    A_lin_P1 = get_virtual_block_lin(A_blocks, [1, 0, 0, 1])
    B_lin_P1 = get_virtual_block_lin(B_blocks, [1, 0, 0, 1])
    
    # P2: (A10+A11)B00
    A_lin_P2 = get_virtual_block_lin(A_blocks, [0, 0, 1, 1])
    B_lin_P2 = get_virtual_block_lin(B_blocks, [1, 0, 0, 0])
    
    # P3: A00(B01-B11)
    A_lin_P3 = get_virtual_block_lin(A_blocks, [1, 0, 0, 0])
    B_lin_P3 = get_virtual_block_lin(B_blocks, [0, 1, 0, -1])
    
    # P4: A11(B10-B00)
    A_lin_P4 = get_virtual_block_lin(A_blocks, [0, 0, 0, 1])
    B_lin_P4 = get_virtual_block_lin(B_blocks, [0, 0, 1, 0]) # Wait, B10 is block 2?
    # B_blocks: 0=00, 1=01, 2=10, 3=11.
    # B10 is index 2. B00 is index 0.
    # So coeffs [0, 0, 1, 0] for B10, and [ -1, 0, 0, 0] for B00?
    # B_lin_P4 should be B10 - B00.
    B_lin_P4 = get_virtual_block_lin(B_blocks, [-1, 0, 1, 0])
    
    # P5: (A00+A01)B11
    A_lin_P5 = get_virtual_block_lin(A_blocks, [1, 1, 0, 0])
    B_lin_P5 = get_virtual_block_lin(B_blocks, [0, 0, 0, 1])
    
    # P6: (A10-A00)(B00+B01)
    A_lin_P6 = get_virtual_block_lin(A_blocks, [-1, 0, 1, 0])
    B_lin_P6 = get_virtual_block_lin(B_blocks, [1, 1, 0, 0])
    
    # P7: (A01-A11)(B10+B11)
    A_lin_P7 = get_virtual_block_lin(A_blocks, [0, 1, 0, -1])
    B_lin_P7 = get_virtual_block_lin(B_blocks, [0, 0, 1, 1])
    
    # Generate terms
    all_terms = []
    
    def process_product(A_lin, B_lin, c_block_offset):
        # c_block_offset: (br, bc) for the C block
        terms = strassen_2x2_lin(A_lin, B_lin)
        for u, v, w_mat in terms:
            # Map w_mat local indices to global C indices
            # w_mat keys are 0,1,2,3 corresponding to C00, C01, C10, C11 of the block
            w_global = {}
            for local_idx, coeff in w_mat.items():
                r = local_idx // 2
                c = local_idx % 2
                global_r = c_block_offset[0]*2 + r
                global_c = c_block_offset[1]*2 + c
                # W index is k*n + i. Here k is col of C, i is row of C.
                # In our W matrix, row index is k*n + i.
                # So global index is global_c * 4 + global_r.
                w_idx = global_c * 4 + global_r
                w_global[w_idx] = w_global.get(w_idx, 0) + coeff
            all_terms.append((u, v, w_global))

    # P1 contributes to C00 block (0,0)
    process_product(A_lin_P1, B_lin_P1, (0,0))
    # P2 contributes to C10 block (1,0)
    process_product(A_lin_P2, B_lin_P2, (1,0))
    # P3 contributes to C01 block (0,1)
    process_product(A_lin_P3, B_lin_P3, (0,1))
    # P4 contributes to C11 block (1,1)
    process_product(A_lin_P4, B_lin_P4, (1,1))
    # P5 contributes to C11 block (1,1)
    process_product(A_lin_P5, B_lin_P5, (1,1))
    # P6 contributes to C11 block (1,1)
    process_product(A_lin_P6, B_lin_P6, (1,1))
    # P7 contributes to C00 block (0,0)
    process_product(A_lin_P7, B_lin_P7, (0,0))

    # --- Part 2: 4x4x1 Schoolbook ---
    # C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            u = {(i,j): 1}
            v = {(j,4): 1}
            # W index: k=4, i=i. idx = 4*4 + i = 16+i
            w = {16+i: 1}
            all_terms.append((u, v, w))

    # --- Build Matrices ---
    rank = len(all_terms)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for idx, coeff in u.items():
            i, j = idx
            U[i*4+j][r] += coeff
        for idx, coeff in v.items():
            j, k = idx
            V[j*5+k][r] += coeff
        for idx, coeff in w.items():
            W[idx][r] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
