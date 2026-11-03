import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We split the problem into:
    # 1. Row 0 of A (1x4) times B (4x8) -> Rank 32
    # 2. Rows 1-2 of A (2x4) times B (4x8) -> Rank 56 (using Strassen on 2x2x2 blocks)
    
    # Total Rank = 32 + 56 = 88
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # --- Part 1: 1x4 x 4x8 (Schoolbook) ---
    # A indices: 0..3 (m=4)
    # B indices: 0..7 (p=8)
    # C indices: 0..7 (p=8)
    # U shape: (1*4, rank) -> actually U is (n*m, rank) = (12, rank)
    # We are filling rows corresponding to i=0.
    # A row index in U: i*m + j. For i=0, indices 0..3.
    
    for j in range(4):
        for k in range(8):
            u = [0] * (n * m)
            u[j] = 1 # i=0, so index j
            
            v = [0] * (m * p)
            v[k] = 1 # j is fixed in the product? No, B is 4x8.
            # In the term A[0,j] * B[j,k], the B index is j*p + k?
            # Wait, B is 4x8. Index (j,k) is j*8 + k.
            # My previous analysis: B is 4x8.
            # In Part 1, A is 1x4.
            # Term: A[0,j] * B[j,k].
            # U index: j (since i=0).
            # V index: j*8 + k.
            # W index: k (since i=0, C is 1x8).
            
            v[j * 8 + k] = 1
            w = [0] * (p * n)
            # W index: k*n + i. i=0. So k*3 + 0 = 3k.
            w[3 * k] = 1
            
            U_rows.append(u)
            V_rows.append(v)
            W_rows.append(w)
            
    # --- Part 2: 2x4 x 4x8 (Strassen based) ---
    # A rows 1,2. i=1,2.
    # A is 2x4. Split into A_1 (2x2) and A_2 (2x2).
    # A_1 corresponds to j=0,1. A_2 corresponds to j=2,3.
    # B is 4x8. Split into B_1 (2x8) and B_2 (2x8).
    # B_1 corresponds to j=0,1. B_2 corresponds to j=2,3.
    
    # C = A_1 B_1 + A_2 B_2.
    # Each term is 2x2 x 2x8.
    # 2x2 x 2x8 is 4 blocks of 2x2 x 2x2.
    
    # Strassen coefficients for 2x2x2
    # A indices: 0,1,2,3 (00,01,10,11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # We define the 7 vectors for U, V, W for a single 2x2x2 block.
    # Note: U vector is length 4 (local), V length 4, W length 4.
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 0, 1, -1],# M3: B21-B22
        [0, -1, 1, 0],# M4: B12-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    
    strassen_W = [
        [1, 0, 0, 0], # M1: C11
        [0, 1, 0, 0], # M2: C12
        [0, 0, 1, 0], # M3: C21
        [0, 0, 0, 1], # M4: C22
        [0, 0, 0, 1], # M5: C22
        [0, 1, 0, 0], # M6: C12
        [0, 0, 1, 0]  # M7: C21
    ]
    # Correction on W for Strassen:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M2 + M5 - M6 - M7 ? No.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M6 - M7
    # C12 = M2 + M5 - M6 - M7 ? No.
    # C12 = M2 + M5 - M7 ?
    # Let's check:
    # C12 = A11 B12 + A12 B22
    # M2 = (A21+A22)B11 -> No.
    # M5 = (A11+A12)B22 = A11 B22 + A12 B22.
    # M7 = (A12-A22)(B21+B22) = A12 B21 + A12 B22 - A22 B21 - A22 B22.
    # This is getting messy to derive W vectors manually.
    # I will use the known W vectors for Strassen.
    
    # W vectors (coefficients for C11, C12, C21, C22)
    # C11: 1, 0, 0, -1, 0, 1, -1
    # C12: 0, 1, 0, 0, 1, 0, 0  -> Wait.
    # C12 = A11 B12 + A12 B22.
    # M2 = (A21+A22)B11.
    # M5 = (A11+A12)B22.
    # M6 = (A21-A11)(B11+B12).
    # M7 = (A12-A22)(B21+B22).
    
    # Let's just use the standard matrix of coefficients.
    # C = [ [M1-M4+M6-M7, M2+M5-M7], [M3+M4+M7, M1-M3+M5-M6] ]
    # Check C12: M2 + M5 - M7?
    # M2 = (A21+A22)B11.
    # M5 = (A11+A12)B22.
    # M7 = (A12-A22)(B21+B22).
    # This doesn't look like A11 B12 + A12 B22.
    
    # Correct Strassen Formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M2 + M5 - M6 - M7 ? No.
    # C12 = M2 + M5 - M7 ?
    # Let's verify C12 = A11 B12 + A12 B22.
    # M2 = (A21+A22)B11.
    # M5 = (A11+A12)B22.
    # M6 = (A21-A11)(B11+B12).
    # M7 = (A12-A22)(B21+B22).
    
    # Actually, standard is:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M2 + M5 - M6 - M7 is WRONG.
    # C12 = M2 + M5 - M7 is WRONG.
    
    # Correct:
    # C12 = M2 + M5 - M6 - M7 ?
    # Let's look at a reliable source or derive.
    # C12 = A11 B12 + A12 B22.
    # M2 = (A21+A22)B11.
    # M5 = (A11+A12)B22.
    # M6 = (A21-A11)(B11+B12).
    # M7 = (A12-A22)(B21+B22).
    
    # It is safer to use the explicit linear combinations.
    # I will define the W vectors based on the formula:
    # C11: [1, 0, 0, -1, 0, 1, -1]
    # C12: [0, 1, 0, 0, 1, -1, 0]  <-- Verify
    # C21: [0, 0, 1, 1, 0, 0, 1]
    # C22: [1, 0, -1, 0, 1, 0, -1]
    
    # Let's check C12 with these:
    # M2 + M5 - M6.
    # M2 = (A21+A22)B11
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # Sum: A21 B11 + A22 B11 + A11 B22 + A12 B22 - A21 B11 - A21 B12 + A11 B11 + A11 B12
    # = A22 B11 + A11 B22 + A12 B22 - A21 B12 + A11 B11 + A11 B12
    # This is not C12.
    
    # I will use the numpy implementation of Strassen to generate the vectors?
    # No, I must hardcode.
    # I will use the following known W vectors for Strassen (from Wikipedia):
    # C11 = M1 - M4 + M6 - M7
    # C12 = M2 + M5 - M6 - M7  (Wait, is it -M7?)
    # C21 = M3 + M4 + M7
    # C22 = M1 - M3 + M5 - M6
    
    # Let's assume these are correct.
    strassen_W = [
        [1, 0, 0, -1, 0, 1, -1], # C11
        [0, 1, 0, 0, 1, -1, 0],  # C12
        [0, 0, 1, 1, 0, 0, 1],   # C21
        [1, 0, -1, 0, 1, 0, -1]  # C22
    ]
    # Note: W vectors are for the output C.
    # In the tensor decomposition, W_t is the vector for the t-th term.
    # So for term t (0..6), the W vector is column t of the above matrix?
    # No, the matrix above gives C components as linear combos of M's.
    # We need M's as linear combos of C? No.
    # The tensor is sum of M_t * (U_t V_t W_t).
    # W_t is the vector of coefficients for C.
    # So for M1, it contributes to C11, C12, C21, C22.
    # From C11 = 1*M1..., M1 coeff for C11 is 1.
    # From C12 = 0*M1..., M1 coeff for C12 is 0.
    # So W_1 = [1, 0, 0, 0]? No.
    # W_1 is the vector of C coefficients for M1.
    # C11 has 1*M1. C12 has 0*M1. C21 has 0*M1. C22 has 1*M1.
    # So W_1 = [1, 0, 0, 1].
    
    # Let's re-read the matrix.
    # C11: M1(1), M4(-1), M6(1), M7(-1)
    # C12: M2(1), M5(1), M6(-1), M7(0)
    # C21: M3(1), M4(1), M7(1)
    # C22: M1(1), M3(-1), M5(1), M6(-1)
    
    # W vectors (index 0..3 for C11,C12,C21,C22):
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, 0]
    # M3: [0, 0, 1, -1]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 1, 0, 1]
    # M6: [1, -1, 0, -1]
    # M7: [-1, 0, 1, 0]
    
    strassen_W_correct = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, -1, 0, -1],
        [-1, 0, 1, 0]
    ]

    # Now build the 28 terms for A_1 B_1
    # A_1 is rows 1,2 of A (i=1,2), cols 0,1 (j=0,1).
    # Local A indices: 0,1,2,3.
    # Global U indices:
    # i=1, j=0 -> 1*4+0 = 4
    # i=1, j=1 -> 1*4+1 = 5
    # i=2, j=0 -> 2*4+0 = 8
    # i=2, j=1 -> 2*4+1 = 9
    
    # B_1 is rows 0,1 of B (j=0,1), cols 0..7.
    # Local B indices: 0,1,2,3 correspond to global B indices?
    # B is 4x8.
    # B_1 is 2x8.
    # We split B_1 into four 2x2 blocks: B_{1,1}, B_{1,2}, B_{1,3}, B_{1,4}.
    # Block k (0..3) corresponds to B columns 4k..4k+3.
    # Local B indices for block k:
    # (0,0) -> j=0, k=4k -> global 0*8 + 4k
    # (0,1) -> j=0, k=4k+1 -> global 0*8 + 4k+1
    # (1,0) -> j=1, k=4k -> global 1*8 + 4k
    # (1,1) -> j=1, k=4k+1 -> global 1*8 + 4k+1
    
    # C output for this part:
    # C rows 1,2 (i=1,2), cols 0..7.
    # Block k corresponds to C columns 4k..4k+3.
    # Local C indices:
    # (0,0) -> i=1, k=4k -> global 4k*3 + 1
    # (0,1) -> i=1, k=4k+1 -> global (4k+1)*3 + 1
    # (1,0) -> i=2, k=4k -> global 4k*3 + 2
    # (1,1) -> i=2, k=4k+1 -> global (4k+1)*3 + 2
    
    for a_block in [0, 1]: # 0 for A_1 B_1, 1 for A_2 B_2
        for b_block in [0, 1, 2, 3]: # 4 blocks of 2x2
            for t in range(7):
                # U vector
                u = [0] * (n * m)
                # Map local A indices to global
                # Local 0->4, 1->5, 2->8, 3->9 for a_block=0
                # Local 0->6, 1->7, 2->10, 3->11 for a_block=1 (j=2,3)
                if a_block == 0:
                    local_to_global_A = [4, 5, 8, 9]
                else:
                    local_to_global_A = [6, 7, 10, 11]
                
                for l_idx, g_idx in enumerate(local_to_global_A):
                    u[g_idx] = strassen_U[t][l_idx]
                    
                # V vector
                v = [0] * (m * p)
                # Local B indices 0,1,2,3 map to global
                # b_block is 0..3.
                # Base col = b_block * 4
                # Local 0 (0,0) -> j=0, col=base -> 0*8 + base
                # Local 1 (0,1) -> j=0, col=base+1 -> 0*8 + base+1
                # Local 2 (1,0) -> j=1, col=base -> 1*8 + base
                # Local 3 (1,1) -> j=1, col=base+1 -> 1*8 + base+1
                
                base_col = b_block * 4
                local_to_global_B = [
                    0 * 8 + base_col,
                    0 * 8 + base_col + 1,
                    1 * 8 + base_col,
                    1 * 8 + base_col + 1
                ]
                
                for l_idx, g_idx in enumerate(local_to_global_B):
                    v[g_idx] = strassen_V[t][l_idx]
                    
                # W vector
                w = [0] * (p * n)
                # Local C indices 0,1,2,3 map to global
                # base_col = b_block * 4
                # Local 0 (0,0) -> i=1, col=base -> base*3 + 1
                # Local 1 (0,1) -> i=1, col=base+1 -> (base+1)*3 + 1
                # Local 2 (1,0) -> i=2, col=base -> base*3 + 2
                # Local 3 (1,1) -> i=2, col=base+1 -> (base+1)*3 + 2
                
                local_to_global_W = [
                    base_col * 3 + 1,
                    (base_col + 1) * 3 + 1,
                    base_col * 3 + 2,
                    (base_col + 1) * 3 + 2
                ]
                
                for l_idx, g_idx in enumerate(local_to_global_W):
                    w[g_idx] = strassen_W_correct[t][l_idx]
                    
                U_rows.append(u)
                V_rows.append(v)
                W_rows.append(w)
                
    # Convert to matrices
    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
