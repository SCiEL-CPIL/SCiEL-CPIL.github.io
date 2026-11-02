import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 85 decomposition
    # 1. A1 B11 (4x4 x 4x4) via Strassen (Rank 49)
    # 2. A1 B12 (4x4 x 4x1) Rank 16
    # 3. A2 B21 (4x1 x 1x4) Rank 16
    # 4. A2 B22 (4x1 x 1x1) Rank 4
    
    rank = 49 + 16 + 16 + 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # --- Part 1: 4x4 x 4x4 using Strassen (Rank 49) ---
    # Strassen for 2x2x2 has rank 7.
    # We use 7 such blocks.
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen steps M1..M7
    # Each Mi is a 2x2 matrix product? No, Mi is a scalar product of 2x2 matrices?
    # No, Mi = X * Y where X, Y are 2x2 matrices.
    # So we need to decompose the 2x2 matrix multiplication.
    # The 2x2 multiplication tensor M_2 has rank 7.
    # We need to generate the U, V, W for M_2.
    
    # M_2 factors (Rank 7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index 2i+j, V index 2j+k, W index 2k+i
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the linear forms for A and B for each of the 7 multiplications.
    # A forms: u1..u7 (vectors in R^4)
    # B forms: v1..v7 (vectors in R^4)
    # C forms: w1..w7 (vectors in R^4)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1 # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1 # B00, B11
    W2[0, 0] = 1; W2[2, 0] = 1 # C00, C11 (C00 += M1, C11 += M1)
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1 # A10, A11
    V2[0, 1] = 1 # B00
    W2[2, 1] = 1; W2[3, 1] = 1 # C10 += M2, C11 -= M2 (Wait, C11 = M1 - M2 + M6)
    # Correction: C11 = M1 - M2 + M6. So W for M2 in C11 is -1.
    W2[2, 1] = 1 # C10
    W2[3, 1] = -1 # C11
    
    # M3
    U2[0, 2] = 1 # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, -B11
    W2[1, 2] = 1; W2[2, 2] = 1 # C01 += M3, C10 += M3
    
    # M4
    U2[3, 3] = 1 # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, -B00
    W2[0, 3] = -1; W2[1, 3] = 1 # C00 -= M4, C01 += M4
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1 # A00, A01
    V2[3, 4] = 1 # B11
    W2[0, 4] = 1; W2[3, 4] = 1 # C00 += M5, C11 += M5
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, -A00
    V2[0, 5] = 1; V2[1, 5] = 1 # B00, B01
    W2[3, 5] = 1 # C11 += M6
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, -A11
    V2[2, 6] = 1; V2[3, 6] = 1 # B10, B11
    W2[0, 6] = 1; W2[1, 6] = 1 # C00 += M7, C01 += M7
    
    # Now, for 4x4x4, we have 4 blocks of A and B.
    # The 7 Strassen steps operate on the 2x2 blocks.
    # For each step t (0..6), we have a 2x2 multiplication.
    # This 2x2 multiplication is decomposed into 7 terms (s=0..6).
    # So total 49 terms.
    
    # Mapping of 4x4 indices to 2x2 block indices
    # A_block[r, c] corresponds to A[2*r + i, 2*c + j]
    # For a specific term (t, s), the U factor for 4x4 is:
    # U_4[2*(2*r+u_row) + (2*c+u_col)] = U2[u_row*2+u_col, s]
    # where (r, c) is determined by the Strassen step t's A-form.
    
    # We need to map the U2 vector (length 4) to the 4x4 matrix indices.
    # The U2 vector corresponds to the 2x2 matrix A_sub.
    # The A_sub indices are (0,0), (0,1), (1,0), (1,1).
    # In the 4x4 matrix, these correspond to specific rows and cols.
    
    # Let's define the A-forms for the 7 steps as 4x4 matrices (sparse).
    # Step 0: A00 + A11.
    # A00 is top-left 2x2. A11 is bottom-right 2x2.
    # So U4 for step 0 is non-zero at indices corresponding to A00 and A11.
    
    # We will generate U4, V4, W4 for the 49 terms.
    
    # Helper to map 2x2 local index (r,c) to 4x4 global index (R,C)
    # Given block offset (br, bc)
    def local_to_global(r, c, br, bc):
        return (2*br + r) * 4 + (2*bc + c) # U index: i*4+j? No, n=4, m=4. i*4+j.
        # Wait, U shape is n*m = 16. Index i*m+j.
    
    # A-forms for steps 0..6 (as list of (local_r, local_c, coeff, block_r, block_c))
    # This is complex. Easier to just build the 4x4 U matrix for each step.
    
    A_forms = []
    B_forms = []
    C_forms = []
    
    # Step 0: (A00+A11)(B00+B11)
    # A: A00 (block 0,0) + A11 (block 1,1)
    # B: B00 (block 0,0) + B11 (block 1,1)
    # C: C00 (block 0,0) + C11 (block 1,1)
    # Note: C00 and C11 are 2x2 blocks of C.
    # The contribution to C is C00 += M1, C11 += M1.
    
    # We define the forms as 16-dim vectors for U, V, W.
    
    def make_A_form(blocks):
        # blocks: list of (br, bc, coeff)
        vec = np.zeros(16)
        for br, bc, coeff in blocks:
            for r in range(2):
                for c in range(2):
                    idx = (2*br + r) * 4 + (2*bc + c)
                    vec[idx] = coeff
        return vec

    def make_B_form(blocks):
        vec = np.zeros(25) # m*p = 5*5? No, for 4x4 part, m=4, p=4.
        # But we are embedding in 5x5.
        # The 4x4 part uses j in 0..3, k in 0..3.
        # V index is j*5 + k? No, V shape is m*p = 5*5 = 25.
        # For the 4x4 sub-problem, we only use j in 0..3, k in 0..3.
        # So V index is j*5 + k.
        vec = np.zeros(25)
        for br, bc, coeff in blocks:
            for r in range(2): # r is row of B (j)
                for c in range(2): # c is col of B (k)
                    j = 2*br + r
                    k = 2*bc + c
                    idx = j * 5 + k
                    vec[idx] = coeff
        return vec

    def make_C_form(blocks):
        # C is 4x4. W shape is p*n = 5*4 = 20.
        # W index is k*4 + i.
        # For 4x4 part, k in 0..3, i in 0..3.
        vec = np.zeros(20)
        for br, bc, coeff in blocks:
            for r in range(2): # r is row of C (i)
                for c in range(2): # c is col of C (k)
                    i = 2*br + r
                    k = 2*bc + c
                    idx = k * 4 + i
                    vec[idx] = coeff
        return vec

    # Define the 7 steps
    # Step 0: (A00+A11)(B00+B11) -> C00 += M, C11 += M
    A_forms.append(make_A_form([(0,0,1), (1,1,1)]))
    B_forms.append(make_B_form([(0,0,1), (1,1,1)]))
    C_forms.append(make_C_form([(0,0,1), (1,1,1)]))
    
    # Step 1: (A10+A11)B00 -> C10 += M, C11 -= M
    A_forms.append(make_A_form([(1,0,1), (1,1,1)]))
    B_forms.append(make_B_form([(0,0,1)]))
    C_forms.append(make_C_form([(1,0,1), (1,1,-1)]))
    
    # Step 2: A00(B01-B11) -> C01 += M, C10 += M
    A_forms.append(make_A_form([(0,0,1)]))
    B_forms.append(make_B_form([(0,1,1), (1,1,-1)]))
    C_forms.append(make_C_form([(0,1,1), (1,0,1)]))
    
    # Step 3: A11(B10-B00) -> C00 -= M, C01 += M
    A_forms.append(make_A_form([(1,1,1)]))
    B_forms.append(make_B_form([(1,0,1), (0,0,-1)]))
    C_forms.append(make_C_form([(0,0,-1), (0,1,1)]))
    
    # Step 4: (A00+A01)B11 -> C00 += M, C11 += M
    A_forms.append(make_A_form([(0,0,1), (0,1,1)]))
    B_forms.append(make_B_form([(1,1,1)]))
    C_forms.append(make_C_form([(0,0,1), (1,1,1)]))
    
    # Step 5: (A10-A00)(B00+B01) -> C11 += M
    A_forms.append(make_A_form([(1,0,1), (0,0,-1)]))
    B_forms.append(make_B_form([(0,0,1), (0,1,1)]))
    C_forms.append(make_C_form([(1,1,1)]))
    
    # Step 6: (A01-A11)(B10+B11) -> C00 += M, C01 += M
    A_forms.append(make_A_form([(0,1,1), (1,1,-1)]))
    B_forms.append(make_B_form([(1,0,1), (1,1,1)]))
    C_forms.append(make_C_form([(0,0,1), (0,1,1)]))
    
    # For each of the 7 steps, we use the 7-term Strassen decomposition of 2x2x2.
    # The U2, V2, W2 defined above are for the 2x2 matrix.
    # We need to "expand" them to the 4x4 indices.
    # The U2 vector (length 4) corresponds to the 2x2 matrix A_sub.
    # The actual 4x4 A matrix is formed by placing A_sub in specific locations?
    # No, the A_form already defines the linear combination of the 4x4 A elements.
    # The Strassen step computes M = A_form * B_form.
    # This M is a 2x2 matrix? No, A_form and B_form are 4x4 matrices (sparse).
    # Their product is a 4x4 matrix.
    # But Strassen's algorithm treats the 2x2 blocks as scalars.
    # So M is a 2x2 matrix of "scalar" products.
    # Each "scalar" product is actually a 2x2 matrix product?
    # No, the inputs to the 7 multiplications are 2x2 matrices.
    # So we have 7 sub-problems: Compute P_t = X_t * Y_t where X_t, Y_t are 2x2.
    # The result P_t is a 2x2 matrix.
    # This 2x2 matrix product P_t is then added to the blocks of C.
    # So for each step t, we need to decompose the 2x2 matrix multiplication X_t * Y_t.
    # We use the rank 7 decomposition U2, V2, W2.
    # The X_t is a 2x2 matrix. Its elements are linear combinations of A's elements.
    # The Y_t is a 2x2 matrix. Its elements are linear combinations of B's elements.
    
    # So, for step t, we have a 2x2 matrix X_t.
    # X_t[r, c] = sum of A elements.
    # We need to map the 2x2 indices (r,c) of X_t to the 4x4 indices.
    # This is already done in A_forms?
    # No, A_forms defines the *entire* 4x4 matrix for the step.
    # But Strassen requires the 7 multiplications to be on 2x2 blocks.
    # So we should define A_t as a 2x2 matrix, where each element is a vector of 4x4 A indices.
    
    # Let's refine:
    # For step t, define A_t as 2x2 array of 16-dim vectors.
    # A_t[r, c] is the linear form for the element (r,c) of the 2x2 matrix X_t.
    
    # Step 0: X0 = [[A00, A01], [A10, A11]]? No.
    # X0 = A00 + A11. This is a 2x2 matrix where every element is A00+A11?
    # No, A00 is a 2x2 matrix.
    # So X0 is a 2x2 matrix of 2x2 matrices?
    # No, Strassen works on 2x2 blocks.
    # The "elements" of the 2x2 block matrix are the 2x2 sub-matrices.
    # So X0 = [[A00+A11, 0], [0, 0]]? No.
    # X0 is the 2x2 matrix:
    # [[ A00+A11, 0 ],
    #  [ 0, A00+A11 ]]? No.
    # The formula is M1 = (A00+A11)(B00+B11).
    # Here A00+A11 is a 2x2 matrix.
    # So M1 is the product of two 2x2 matrices.
    # So we just need to decompose the 2x2 matrix multiplication.
    # The inputs are 2x2 matrices.
    # So we can reuse U2, V2, W2.
    # We just need to map the 2x2 indices of the input matrices to the 4x4 indices.
    
    # For step t, the input matrix X_t is a 2x2 matrix.
    # X_t[r, c] corresponds to a specific 2x2 block of A?
    # No, X_t is formed by summing blocks.
    # Example Step 0: X0 = A00 + A11.
    # This means X0[0,0] = A00[0,0] + A11[0,0].
    # So X0 is a 2x2 matrix where each element is a sum of 2 elements from A.
    # We can represent X0 by 4 vectors (one for each element of the 2x2 X0).
    # Vec_X0_00 = e_{0,0} + e_{2,2} (indices in 4x4)
    # Vec_X0_01 = e_{0,1} + e_{2,3}
    # Vec_X0_10 = e_{1,0} + e_{3,2}
    # Vec_X0_11 = e_{1,1} + e_{3,3}
    
    # Then we use U2, V2, W2 to decompose the product.
    # For each of the 7 terms s in U2:
    # U_4 = sum_{r,c} U2[r*2+c, s] * Vec_X_t_rc
    # V_4 = sum_{r,c} V2[r*2+c, s] * Vec_Y_t_rc
    # W_4 = sum_{r,c} W2[r*2+c, s] * Vec_Z_t_rc
    
    # This is the correct way.
    
    # Define the 2x2 block structures for each step.
    # Step 0: X = A00+A11, Y = B00+B11, Z = C00+C11
    # X blocks: (0,0) and (1,1)
    # Y blocks: (0,0) and (1,1)
    # Z blocks: (0,0) and (1,1)
    
    steps = [
        {
            'X': [(0,0,1), (1,1,1)],
            'Y': [(0,0,1), (1,1,1)],
            'Z': [(0,0,1), (1,1,1)]
        },
        {
            'X': [(1,0,1), (1,1,1)],
            'Y': [(0,0,1)],
            'Z': [(1,0,1), (1,1,-1)]
        },
        {
            'X': [(0,0,1)],
            'Y': [(0,1,1), (1,1,-1)],
            'Z': [(0,1,1), (1,0,1)]
        },
        {
            'X': [(1,1,1)],
            'Y': [(1,0,1), (0,0,-1)],
            'Z': [(0,0,-1), (0,1,1)]
        },
        {
            'X': [(0,0,1), (0,1,1)],
            'Y': [(1,1,1)],
            'Z': [(0,0,1), (1,1,1)]
        },
        {
            'X': [(1,0,1), (0,0,-1)],
            'Y': [(0,0,1), (0,1,1)],
            'Z': [(1,1,1)]
        },
        {
            'X': [(0,1,1), (1,1,-1)],
            'Y': [(1,0,1), (1,1,1)],
            'Z': [(0,0,1), (0,1,1)]
        }
    ]
    
    for step in steps:
        # Build 4 vectors for X, Y, Z (2x2 matrices)
        # Each vector is length 16 (for X), 25 (for Y), 20 (for Z)
        # But we only need the non-zero parts.
        
        # For each of the 7 Strassen terms s:
        for s in range(7):
            U4 = np.zeros(16)
            V4 = np.zeros(25)
            W4 = np.zeros(20)
            
            # X is 2x2. Elements (r,c) of X are sums of blocks.
            # U2 index is 2*r + c.
            for r in range(2):
                for c in range(2):
                    coeff_u = U2[r*2+c, s]
                    if coeff_u != 0:
                        # Add coeff_u * (sum of blocks in X[r,c])
                        for br, bc, coeff_x in step['X']:
                            # Block (br, bc) contributes to X[r,c]?
                            # No, X is a 2x2 matrix.
                            # The blocks listed in step['X'] are the *entire* 2x2 matrix X.
                            # So if step['X'] has (0,0,1), it means X[0,0] += A00.
                            # So X[0,0] is a 2x2 matrix.
                            # We need to map the 2x2 indices of A00 to 4x4.
                            # A00 is at block (0,0).
                            # So A00[i,j] is at 4x4 index (2*0+i, 2*0+j).
                            # So we add coeff_u * coeff_x to U4[2*br+i + (2*bc+j)*4? No.
                            # U index is i*4+j.
                            for i in range(2):
                                for j in range(2):
                                    if r == i and c == j: # No, X[r,c] is the element.
                                        # The block (br,bc) contributes to X[br,bc]?
                                        # No, the step definition lists the blocks that form the 2x2 matrix X.
                                        # So X[br,bc] += coeff_x * A_block(br,bc).
                                        # So if r==br and c==bc, then X[r,c] has this contribution.
                                        pass
                            # Correct logic:
                            # X is a 2x2 matrix.
                            # X[r,c] = sum_{(br,bc,coeff) in step['X'] where br==r and bc==c} coeff * A_block(br,bc)
                            # So for a given r,c, we sum the blocks that match.
                            pass
            
            # Let's re-implement the accumulation.
            U4 = np.zeros(16)
            V4 = np.zeros(25)
            W4 = np.zeros(20)
            
            for r in range(2):
                for c in range(2):
                    coeff_u = U2[r*2+c, s]
                    coeff_v = V2[r*2+c, s]
                    coeff_w = W2[r*2+c, s]
                    
                    if coeff_u != 0:
                        for br, bc, cx in step['X']:
                            if br == r and bc == c:
                                for i in range(2):
                                    for j in range(2):
                                        idx = (2*br + i) * 4 + (2*bc + j)
                                        U4[idx] += coeff_u * cx
                                        
                    if coeff_v != 0:
                        for br, bc, cy in step['Y']:
                            if br == r and bc == c:
                                for i in range(2):
                                    for j in range(2):
                                        j_idx = 2*br + i
                                        k_idx = 2*bc + j
                                        idx = j_idx * 5 + k_idx
                                        V4[idx] += coeff_v * cy
                                        
                    if coeff_w != 0:
                        for br, bc, cz in step['Z']:
                            if br == r and bc == c:
                                for i in range(2):
                                    for j in range(2):
                                        i_idx = 2*br + i
                                        k_idx = 2*bc + j
                                        idx = k_idx * 4 + i_idx
                                        W4[idx] += coeff_w * cz
                                        
            U[:, col] = U4
            V[:, col] = V4
            W[:, col] = W4
            col += 1
            
    # --- Part 2: A1 B12 (4x4 x 4x1) Rank 16 ---
    # A1 is 4x4 (j=0..3). B12 is 4x1 (j=0..3, k=4).
    # C is 4x1 (i=0..3, k=4).
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1 # n=4, m=5. Index i*5+j.
            V[j*5 + 4, col] = 1 # m=5, p=5. Index j*5+k. k=4.
            W[4*4 + i, col] = 1 # p=5, n=4. Index k*4+i. k=4.
            col += 1
            
    # --- Part 3: A2 B21 (4x1 x 1x4) Rank 16 ---
    # A2 is 4x1 (j=4). B21 is 1x4 (j=4, k=0..3).
    # C is 4x4 (i=0..3, k=0..3).
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1 # j=4, k=k.
            W[k*4 + i, col] = 1
            col += 1
            
    # --- Part 4: A2 B22 (4x1 x 1x1) Rank 4 ---
    # A2 is 4x1 (j=4). B22 is 1x1 (j=4, k=4).
    # C is 4x1 (i=0..3, k=4).
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
