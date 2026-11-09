import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 50s margin
    
    n, m, p = DIMENSIONS
    # We will build rank 77 decomposition
    # Part 1: 4x4 x 4x4 (Rank 49)
    # Part 2: 4x4 x 4x2 (Rank 28)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # --- Part 1: 4x4 x 4x4 Strassen ---
    # We need to generate the 49 rank-1 terms.
    # Strassen 2x2 terms
    # Let's define the 7 Strassen products for 2x2.
    # Each product is (LinearComb(A)) * (LinearComb(B)).
    # This expands to sum of rank-1 terms.
    
    # Instead of symbolic expansion, we can use the known structure.
    # For 2x2, rank is 7.
    # The 7 terms correspond to the 7 Strassen multiplications.
    # But each multiplication is of 2x2 matrices, so it itself has rank 7?
    # No. The 7 Strassen multiplications are of SCALARS if we go down to 1x1.
    # For 2x2, we treat entries as scalars.
    # So the 7 terms are rank-1 tensors for the 2x2x2 problem.
    
    # Let's generate the 2x2x2 Strassen decomposition.
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices: A[i,j], B[j,k], C[i,k].
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # We can represent each Ms as a rank-1 tensor?
    # No, Ms is a scalar product of two linear forms.
    # A linear form in A is a vector u in R^4 (flattened).
    # A linear form in B is a vector v in R^4.
    # The product u^T A v (flattened) is a rank-1 tensor?
    # Yes, the tensor T[i,j,k] = u[i,j] v[j,k] w[i,k] is rank 1 if u,v,w are vectors.
    # Here, the term is (sum a_ij u_ij) * (sum b_jk v_jk).
    # This is a bilinear form.
    # To decompose into rank-1 tensors T = sum_r U_r \otimes V_r \otimes W_r,
    # we need to expand the product of sums.
    
    # For M1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each term A_ij B_jk is a rank-1 tensor for the 2x2x2 case?
    # Yes, T[i,j,k] = 1 if (i,j)=(0,0) and (j,k)=(0,0) -> C[0,0] gets 1.
    # So A00B00 contributes to C00.
    # U has 1 at (0,0). V has 1 at (0,0). W has 1 at (0,0).
    
    # So for each Ms, we expand it into 4 terms (or fewer if coefficients are 0).
    # M1 has 4 terms.
    # M2 has 2 terms: (A10+A11)B00 = A10B00 + A11B00.
    # M3 has 2 terms: A00(B01-B11) = A00B01 - A00B11.
    # M4 has 2 terms: A11(B10-B00) = A11B10 - A11B00.
    # M5 has 2 terms: (A00+A01)B11 = A00B11 + A01B11.
    # M6 has 4 terms: (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01.
    # M7 has 4 terms: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
    
    # Total terms: 4+2+2+2+2+4+4 = 20?
    # But rank is 7. Why 20?
    # Because we are expanding into "schoolbook-like" terms but with coefficients.
    # The rank of the tensor is the minimum number of rank-1 tensors.
    # Strassen says 7 is sufficient.
    # The 7 rank-1 tensors are NOT the expanded scalar products.
    # They are the tensors defined by the linear forms.
    # Wait. A rank-1 tensor is $u \otimes v \otimes w$.
    # $T[i,j,k] = u[i,j] v[j,k] w[i,k]$.
    # If we take $u$ to be the vector for $(A00+A11)$, i.e., $u[0,0]=1, u[1,1]=1, 0$ else.
    # And $v$ to be the vector for $(B00+B11)$, i.e., $v[0,0]=1, v[1,1]=1, 0$ else.
    # Then $u \otimes v \otimes w$ where $w$ is determined by the contribution to C?
    # No, $w$ is fixed for the term.
    # The term $M_1$ contributes to C as:
    # C00 += 1 * M1
    # C11 += 1 * M1
    # So $w[0,0]=1, w[1,1]=1, 0$ else?
    # Let's check.
    # Term 1: $u_1 = [1,0,0,1]^T$ (A00, A11). $v_1 = [1,0,0,1]^T$ (B00, B11).
    # $w_1$ must satisfy $w_1[i,k] = \delta_{i,0}\delta_{k,0} + \delta_{i,1}\delta_{k,1}$?
    # Then $T[i,j,k] = u_1[i,j] v_1[j,k] w_1[i,k]$.
    # For i=0, k=0: $u_1[0,j] v_1[j,0] w_1[0,0]$.
    # $u_1[0,0]=1, u_1[0,1]=0$. $v_1[0,0]=1, v_1[1,0]=0$.
    # Sum over j: $j=0 \implies 1 \cdot 1 \cdot 1 = 1$.
    # This gives 1 to C00. Correct.
    # For i=1, k=1: $u_1[1,j] v_1[j,1] w_1[1,1]$.
    # $u_1[1,1]=1$. $v_1[1,1]=1$. $w_1[1,1]=1$.
    # Sum: 1. Correct.
    # For i=0, k=1: $w_1[0,1]=0$. Correct.
    
    # So each Strassen product $M_s$ corresponds to ONE rank-1 tensor in the 2x2x2 decomposition?
    # Yes! The 7 rank-1 tensors are defined by the 7 linear forms.
    # So for 2x2x2, we have 7 terms.
    
    # Now for 4x4x4.
    # We use 7 "blocks" of 2x2x2 Strassen?
    # No. The 4x4x4 Strassen algorithm uses 7 multiplications of 2x2 matrices.
    # Each multiplication is a 2x2x2 problem.
    # So we have 7 instances of the 2x2x2 problem.
    # Each instance has rank 7.
    # Total rank $7 \times 7 = 49$.
    
    # We need to map the indices.
    # The 7 instances correspond to the 7 Strassen products $M_1...M_7$ for the 4x4 case.
    # Each $M_s$ is a 2x2 matrix product.
    # The inputs to $M_s$ are linear combinations of the 2x2 blocks of A and B.
    # Let $A$ be divided into 4 blocks $A_{pq}$ ($p,q \in \{0,1\}$).
    # Let $B$ be divided into 4 blocks $B_{rs}$ ($r,s \in \{0,1\}$).
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # ...
    
    # For each $M_s$, we have a 2x2x2 Strassen decomposition.
    # The U factor for this instance is the linear combination of A blocks.
    # The V factor is the linear combination of B blocks.
    # The W factor is the contribution to the 4x4 result.
    
    # We will generate these 49 terms.
    
    def get_2x2_strassen_terms():
        """Returns list of (u, v, w) for 2x2x2 Strassen.
        u, v are vectors of length 4 (flattened 2x2).
        w is vector of length 4 (flattened 2x2 result).
        """
        terms = []
        # M1
        u = [1,0,0,1] # A00, A11
        v = [1,0,0,1] # B00, B11
        w = [1,0,0,1] # C00, C11
        terms.append((u,v,w))
        # M2
        u = [0,0,1,1] # A10, A11
        v = [1,0,0,0] # B00
        w = [0,0,1,0] # C10
        terms.append((u,v,w))
        # M3
        u = [1,0,0,0] # A00
        v = [0,1,0,-1] # B01, -B11
        w = [0,1,0,0] # C01
        terms.append((u,v,w))
        # M4
        u = [0,0,0,1] # A11
        v = [0,0,1,-1] # B10, -B00
        w = [0,0,1,0] # C10
        terms.append((u,v,w))
        # M5
        u = [1,1,0,0] # A00, A01
        v = [0,0,0,1] # B11
        w = [0,1,0,0] # C01
        terms.append((u,v,w))
        # M6
        u = [0,0,1,-1] # A10, -A00
        v = [1,1,0,0] # B00, B01
        w = [1,0,0,0] # C00
        terms.append((u,v,w))
        # M7
        u = [0,1,0,-1] # A01, -A11
        v = [0,0,1,1] # B10, B11
        w = [0,0,0,1] # C11
        terms.append((u,v,w))
        return terms

    base_terms = get_2x2_strassen_terms()
    
    # Now expand to 4x4x4.
    # We have 7 groups, each with 7 terms.
    # For each group s (0..6), we determine the U and V linear forms on the 4x4 blocks.
    # And the W linear form on the 4x4 result.
    
    # The 7 Strassen products for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Note: A00 is top-left 2x2 block. A11 is bottom-right.
    # In flattened 16x1 vector:
    # A00: indices 0-3
    # A01: indices 4-7
    # A10: indices 8-11
    # A11: indices 12-15
    
    # For each group s, we have a "macro" U and V (linear combos of blocks).
    # And for each of the 7 sub-terms in that group, we have a "micro" U and V (linear combos of entries in the 2x2 block).
    # The total U is the tensor product of macro and micro?
    # Yes.
    
    # Let's define the macro coefficients for A and B for each of the 7 products.
    # prod 0 (M1): A: [1,0,0,1], B: [1,0,0,1]
    # prod 1 (M2): A: [0,0,1,1], B: [1,0,0,0]
    # prod 2 (M3): A: [1,0,0,0], B: [0,1,0,-1]
    # prod 3 (M4): A: [0,0,0,1], B: [0,0,1,-1]
    # prod 4 (M5): A: [1,1,0,0], B: [0,0,0,1]
    # prod 5 (M6): A: [0,0,1,-1], B: [1,1,0,0]
    # prod 6 (M7): A: [0,1,0,-1], B: [0,0,1,1]
    
    macro_A = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,-1],
        [0,1,0,-1]
    ]
    macro_B = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [0,0,1,-1],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    
    # The result of each product M_s is a 2x2 matrix.
    # We need to add these to the 4x4 result C.
    # C is 4x4.
    # M1 contributes to all 4 blocks of C?
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11).
    # C00 += M1 - M4 - M6 + M7
    # C01 += M3 + M5
    # C10 += M2 + M4
    # C11 += M1 - M3 + M5 - M7
    
    # So for each product s, it contributes to specific C blocks with specific signs.
    # And within each C block, the 2x2 result of the multiplication is placed.
    
    # We will iterate s from 0 to 6.
    # For each s, we iterate t from 0 to 6 (the 7 sub-terms of the 2x2 Strassen).
    # For each (s,t), we construct U, V, W.
    
    for s in range(7):
        for t in range(7):
            u_micro, v_micro, w_micro = base_terms[t]
            
            # Construct U (length 16)
            U = [0]*16
            for block in range(4):
                coeff = macro_A[s][block]
                if coeff != 0:
                    for idx in range(4):
                        U[block*4 + idx] = coeff * u_micro[idx]
                        
            # Construct V (length 16) - but V is for B (4x4 part)
            # Wait, V in the final output is for the full B (4x6).
            # Here we are only doing the 4x4 part of B.
            # So V_local (length 16) will be placed in the first 16 cols of V_final.
            V_local = [0]*16
            for block in range(4):
                coeff = macro_B[s][block]
                if coeff != 0:
                    for idx in range(4):
                        V_local[block*4 + idx] = coeff * v_micro[idx]
                        
            # Construct W (length 16 for the 4x4 result part)
            # We need to map the 2x2 result w_micro to the 4x4 result.
            # Which C blocks does product s contribute to?
            # And with what sign?
            
            w_block_coeffs = {}
            if s == 0: # M1
                w_block_coeffs = {0: 1, 3: 1} # C00, C11
            elif s == 1: # M2
                w_block_coeffs = {2: 1} # C10
            elif s == 2: # M3
                w_block_coeffs = {1: 1} # C01
            elif s == 3: # M4
                w_block_coeffs = {0: -1, 2: 1} # C00, C10
            elif s == 4: # M5
                w_block_coeffs = {1: 1} # C01
            elif s == 5: # M6
                w_block_coeffs = {0: -1} # C00
            elif s == 6: # M7
                w_block_coeffs = {0: 1, 3: -1} # C00, C11
            
            W_local = [0]*16
            for c_block, sign in w_block_coeffs.items():
                for idx in range(4):
                    W_local[c_block*4 + idx] = sign * w_micro[idx]
                    
            # Now we have U (16), V_local (16), W_local (16).
            # We need to pad V_local to length 24 (for 4x6 B).
            # The 4x4 part is cols 0-3 of B.
            # So V_final[0:16] = V_local, V_final[16:24] = 0.
            V_final = V_local + [0]*8
            
            # We need to pad W_local to length 24 (for 4x6 Result).
            # The result is 4x6.
            # The 4x4 part corresponds to cols 0-3 of the result.
            # W index is k*n + i.
            # k is col of result (0..5). i is row of result (0..3).
            # For the 4x4 part, k is 0..3.
            # W_local index is c_block*4 + idx.
            # c_block is 0..3 (which is k_block? No).
            # In 4x4 result, index is k*4 + i.
            # In 4x6 result, index is k*4 + i.
            # So for k in 0..3, the index is the same.
            # For k in 4..5, index is 16..23.
            # So W_final[0:16] = W_local, W_final[16:24] = 0.
            W_final = W_local + [0]*8
            
            add_term(U, V_final, W_final)

    # --- Part 2: 4x4 x 4x2 ---
    # Rank 28.
    # We identified 4 independent 2x2x2 Strassen multiplications.
    # A11 * C_top, A12 * C_bot, A21 * C_top, A22 * C_bot.
    # C_top is rows 0-1 of C (4x2).
    # C_bot is rows 2-3 of C.
    
    # For each of these 4 multiplications, we use the 2x2 Strassen decomposition (7 terms).
    # Total 4 * 7 = 28 terms.
    
    for block_a in range(4):
        for block_c in [0, 1]: # 0 for top, 1 for bot
            # Determine which A block and C block
            # A blocks: 0:00, 1:01, 2:10, 3:11
            # C blocks: 0:top (rows 0-1), 1:bot (rows 2-3)
            
            # We need to map the 2x2 A block to the 16-dim U vector.
            # And the 2x2 C block to the 16-dim V vector (part of 24).
            # And the 2x2 result to the 16-dim W vector (part of 24).
            
            for t in range(7):
                u_micro, v_micro, w_micro = base_terms[t]
                
                U = [0]*16
                # A block index
                # block_a 0 -> A00 (0-3)
                # block_a 1 -> A01 (4-7)
                # block_a 2 -> A10 (8-11)
                # block_a 3 -> A11 (12-15)
                for idx in range(4):
                    U[block_a*4 + idx] = u_micro[idx]
                    
                V_final = [0]*24
                # C is 4x2.
                # V index is j*p + k. j in 0..3, k in 0..1.
                # C_top is rows 0-1. j=0,1.
                # C_bot is rows 2-3. j=2,3.
                # k is 0,1.
                # The 2x2 C block has indices (j_local, k).
                # j_local 0,1.
                # If block_c == 0 (top): j = j_local.
                # If block_c == 1 (bot): j = j_local + 2.
                
                for idx in range(4):
                    # v_micro is flattened 2x2. idx = j_local*2 + k.
                    j_local = idx // 2
                    k = idx % 2
                    j = j_local + (2 if block_c == 1 else 0)
                    V_final[j*2 + k] = v_micro[idx]
                    
                W_final = [0]*24
                # Result is 4x2.
                # W index is k*n + i. k in 0..1, i in 0..3.
                # The 2x2 result corresponds to i_block (0 or 1) and k.
                # Wait. The multiplication is A_block (2x2) * C_block (2x2).
                # Result is 2x2.
                # The rows of the result correspond to the rows of A_block.
                # A_block 0 (A00) -> rows 0,1 of result.
                # A_block 1 (A01) -> rows 0,1 of result.
                # A_block 2 (A10) -> rows 2,3 of result.
                # A_block 3 (A11) -> rows 2,3 of result.
                
                # So i_offset = 0 if block_a < 2 else 2.
                i_offset = 0 if block_a < 2 else 2
                
                for idx in range(4):
                    # w_micro is flattened 2x2. idx = i_local*2 + k.
                    i_local = idx // 2
                    k = idx % 2
                    i = i_local + i_offset
                    W_final[k*4 + i] = w_micro[idx]
                    
                add_term(U, V_final, W_final)

    # Convert lists to matrices
    rank = len(U_list)
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    # Return as lists of lists to match interface?
    # Interface: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # The seed program returns lists of lists.
    
    U_out = U_mat.tolist()
    V_out = V_mat.tolist()
    W_out = W_mat.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    construct()
