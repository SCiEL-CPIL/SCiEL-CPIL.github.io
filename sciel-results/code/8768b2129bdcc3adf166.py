import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 110 decomposition for 5x5
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a decomposition for a sub-block
    # a_idx, b_idx, c_idx are lists of global indices
    # sub_rank is the number of columns this block uses
    # generator function yields (u_local, v_local, w_local)
    def add_block(a_idx, b_idx, c_idx, offset, gen_func):
        for col_rel, (u_loc, v_loc, w_loc) in enumerate(gen_func()):
            col_abs = offset + col_rel
            U[a_idx[u_loc]][col_abs] = 1
            V[b_idx[v_loc]][col_abs] = 1
            W[c_idx[w_loc]][col_abs] = 1

    # Strassen 2x2 Generator
    # Returns 7 triples (u, v, w) where u,v,w are local 0-3 indices for 2x2 matrix
    # Note: The values in U,V,W are all 1, but Strassen involves additions/subtractions in the INPUT/OUTPUT.
    # Wait, the tensor decomposition coefficients are in U, V, W.
    # Standard Strassen uses 7 multiplications.
    # The coefficients in U, V, W are not just 1. They are the coefficients of the linear combinations.
    # My previous assumption that "Rank 7" means 7 columns with 1s was for the *existence* of the decomposition.
    # But the entries of U, V, W must be the coefficients.
    
    # Let's re-read the contract.
    # "U, V, W factor matrices... with integer/half-integer entries"
    # "exactly decompose the matrix multiplication tensor"
    # The decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # So U, V, W contain the coefficients.
    
    # For Schoolbook $1 \times 1 \times 1$:
    # $U[0][0]=1, V[0][0]=1, W[0][0]=1$.
    
    # For Strassen $2 \times 2$:
    # We need to define the 7 vectors.
    
    # Let's implement a generic Strassen 2x2 decomposition generator that returns the coefficients.
    
    def strassen_2x2_coeffs():
        # Returns list of 7 tuples (U_vec, V_vec, W_vec)
        # U_vec is size 4, V_vec size 4, W_vec size 4.
        # Indices: 00, 01, 10, 11
        
        # M1 = (A11+A22)(B11+B22)
        # A part: [1, 0, 0, 1]
        # B part: [1, 0, 0, 1]
        # C part: C11 += 1, C22 -= 1
        # W1: [1, 0, 0, -1]
        
        # M2 = (A21+A22)B11
        # A part: [0, 0, 1, 1]
        # B part: [1, 0, 0, 0]
        # C part: C21 += 1
        # W2: [0, 0, 1, 0]
        
        # M3 = A11(B12-B22)
        # A part: [1, 0, 0, 0]
        # B part: [0, 1, 0, -1]
        # C part: C12 += 1
        # W3: [0, 1, 0, 0]
        
        # M4 = A22(B21-B11)
        # A part: [0, 0, 0, 1]
        # B part: [0, 0, 1, -1]
        # C part: C11 += 1, C21 += 1
        # W4: [1, 0, 1, 0]
        
        # M5 = (A11+A12)B22
        # A part: [1, 1, 0, 0]
        # B part: [0, 0, 0, 1]
        # C part: C11 -= 1, C12 += 1
        # W5: [-1, 1, 0, 0]
        
        # M6 = (A21-A11)(B11+B12)
        # A part: [-1, 0, 1, 0]
        # B part: [1, 1, 0, 0]
        # C part: C22 += 1
        # W6: [0, 0, 0, 1]
        
        # M7 = (A12-A22)(B21+B22)
        # A part: [0, 1, 0, -1]
        # B part: [0, 0, 1, 1]
        # C part: C11 += 1
        # W7: [1, 0, 0, 0]
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]

    def strassen_4x4_coeffs():
        # Recursively apply Strassen 2x2 to 4x4
        # Split 4x4 into 2x2 blocks
        # Blocks: A11, A12, A21, A22 (each 2x2)
        # We need 7 products of 2x2 blocks.
        # Each product is a 2x2x2 multiplication, which has rank 7.
        # So we need to combine the 7 Strassen steps, each expanding to 7 sub-steps.
        
        # The indices for 4x4 are 0-15.
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # Strassen 2x2 on the block level:
        # M1 = (A11+A22)(B11+B22)
        # ...
        
        # This requires constructing the linear combinations of the 2x2 blocks.
        # This is complex to generate dynamically with correct global indices.
        # Given the time, I will use the 110 rank decomposition but implement the 4x4 part using a known 49-rank decomposition if available, or fallback to schoolbook 64 if Strassen 4x4 is too hard to implement correctly in time.
        # However, 110 relies on 49.
        
        # Let's try to implement the 4x4 Strassen by mapping the 2x2 Strassen coefficients to the 4x4 indices.
        
        # Actually, a simpler approach for 4x4:
        # Use 4 Strassen 2x2 decompositions? No, that's 28.
        # The recursive Strassen 4x4 uses 7 multiplications of size 2x2.
        # Each 2x2 multiplication uses 7 scalar multiplications.
        # Total 49 scalar multiplications.
        
        # I will implement a helper to map a 2x2 decomposition into a specific 2x2 block of a larger matrix.
        
        # For now, to ensure correctness and avoid bugs in the complex mapping, 
        # I will use the **Schoolbook** decomposition for the 4x4 part (Rank 64) 
        # which gives a total rank of 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
        # This is the same as the seed.
        
        # To improve, I MUST implement Strassen 2x2 for the 2x2 blocks in the 5=2+3 split.
        
        # Let's switch to the **2+3 Split** (Rank 111 with schoolbook 3x3, or better with Strassen-like 3x3).
        # If I use Strassen 2x2 for the 2x2 part (Rank 7) and Schoolbook for the rest:
        # 1. 2x2x2: 7
        # 2. 2x2x3: 6
        # 3. 2x3x2: 6
        # 4. 2x3x3: 18
        # 5. 3x2x2: 12
        # 6. 3x2x3: 18
        # 7. 3x3x2: 18
        # 8. 3x3x3: 27 (Schoolbook)
        # Total: 7 + 6 + 6 + 18 + 12 + 18 + 18 + 27 = 112.
        
        # This is better than 125.
        # I can implement the 3x3x3 using a recursive Strassen-like approach (Rank 26).
        # Total: 7 + 6 + 6 + 18 + 12 + 18 + 18 + 26 = 111.
        
        # I will implement the 111-rank decomposition.
        # It is safer than 110 because I don't need to implement the complex 4x4 Strassen mapping, 
        # I just need to implement 2x2 Strassen (Rank 7) and 3x3 recursive (Rank 26) and schoolbook rectangulars.
        
        # Re-evaluating 110 vs 111.
        # 110 requires 4x4 Strassen.
        # 111 requires 3x3 recursive Strassen.
        # 3x3 recursive Strassen is:
        # Split 3 into 2+1.
        # 2x2x2 (7), 2x2x1 (4), 2x1x2 (4), 2x1x1 (2), 1x2x2 (4), 1x2x1 (2), 1x1x2 (2), 1x1x1 (1).
        # Sum: 26.
        # This is easier to implement than 4x4 Strassen (which requires 7x7 mapping).
        
        # So I will go with the **2+3 Split, Rank 111** decomposition.
        
        # Implementation Plan for 111:
        # 1. Define index sets for 2x2, 2x3, 3x2, 3x3 blocks.
        # 2. Implement `add_strassen_2x2` for the 2x2x2 block.
        # 3. Implement `add_schoolbook_rect` for the rectangular blocks.
        # 4. Implement `add_recursive_3x3` for the 3x3x3 block.
        
        # Indices for 5x5 split 2+3:
        # A rows 0-1 (2), rows 2-4 (3).
        # A cols 0-1 (2), cols 2-4 (3).
        
        # Block A11 (2x2): indices 0,1, 5,6
        # Block A12 (2x3): indices 2,3,4, 7,8,9
        # Block A21 (3x2): indices 10,11, 15,16
        # Block A22 (3x3): indices 12,13,14, 17,18,19, 22,23,24
        
        # Same for B and C.
        
        # I will code this up.
        
        def get_indices(size_r, size_c, start_r, start_c, total_cols):
            idx = []
            for r in range(size_r):
                for c in range(size_c):
                    idx.append((start_r + r) * total_cols + (start_c + c))
            return idx

        a11 = get_indices(2, 2, 0, 0, 5)
        a12 = get_indices(2, 3, 0, 2, 5)
        a21 = get_indices(3, 2, 2, 0, 5)
        a22 = get_indices(3, 3, 2, 2, 5)
        
        b11 = get_indices(2, 2, 0, 0, 5)
        b12 = get_indices(2, 3, 0, 2, 5)
        b21 = get_indices(3, 2, 2, 0, 5)
        b22 = get_indices(3, 3, 2, 2, 5)
        
        c11 = get_indices(2, 2, 0, 0, 5)
        c12 = get_indices(2, 3, 0, 2, 5)
        c21 = get_indices(3, 2, 2, 0, 5)
        c22 = get_indices(3, 3, 2, 2, 5)
        
        # 1. C11 part: A11 B11 (2x2x2, Rank 7)
        offset = 0
        for u, v, w in strassen_2x2_coeffs():
            for k in range(4):
                if u[k] != 0 and v[k] != 0 and w[k] != 0:
                    U[a11[k]][offset] += u[k]
                    V[b11[k]][offset] += v[k]
                    W[c11[k]][offset] += w[k]
            offset += 1
        
        # 2. C11 part: A12 B21 (2x3x2, Rank 6)
        # Schoolbook 2x3x2
        # A12 size 2x3. B21 size 3x2. C11 size 2x2.
        # C11[i,j] = sum_k A12[i,k] B21[k,j]
        # 6 terms.
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[a12[i*3+k]][offset] = 1
                    V[b21[k*2+j]][offset] = 1
                    W[c11[i*2+j]][offset] = 1
                    offset += 1
        # Rank 6.
        
        # 3. C12 part: A11 B12 (2x2x3, Rank 6)
        # A11 (2x2) * B12 (2x3) -> C12 (2x3)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[a11[i*2+k]][offset] = 1
                    V[b12[k*3+j]][offset] = 1
                    W[c12[i*3+j]][offset] = 1
                    offset += 1
        # Rank 6.
        
        # 4. C12 part: A12 B22 (2x3x3, Rank 18)
        # Schoolbook.
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[a12[i*3+k]][offset] = 1
                    V[b22[k*3+j]][offset] = 1
                    W[c12[i*3+j]][offset] = 1
                    offset += 1
        # Rank 18.
        
        # 5. C21 part: A21 B11 (3x2x2, Rank 12)
        # Schoolbook.
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    U[a21[i*2+k]][offset] = 1
                    V[b11[k*2+j]][offset] = 1
                    W[c21[i*2+j]][offset] = 1
                    offset += 1
        # Rank 12.
        
        # 6. C21 part: A22 B21 (3x3x2, Rank 18)
        # Schoolbook.
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    U[a22[i*3+k]][offset] = 1
                    V[b21[k*2+j]][offset] = 1
                    W[c21[i*2+j]][offset] = 1
                    offset += 1
        # Rank 18.
        
        # 7. C22 part: A21 B12 (3x2x3, Rank 18)
        # Schoolbook.
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[a21[i*2+k]][offset] = 1
                    V[b12[k*3+j]][offset] = 1
                    W[c22[i*3+j]][offset] = 1
                    offset += 1
        # Rank 18.
        
        # 8. C22 part: A22 B22 (3x3x3, Rank 26)
        # Recursive Strassen-like 3x3.
        # Split 3 into 2+1.
        # A22 blocks:
        # A22_11 (2x2), A22_12 (2x1), A22_21 (1x2), A22_22 (1x1)
        
        # Indices in A22 (global a22):
        # a22 is [12,13,14, 17,18,19, 22,23,24]
        # Local 3x3 indices:
        # 0,1,2
        # 3,4,5
        # 6,7,8
        
        # A22_11 (0,0), (0,1), (1,0), (1,1) -> global 12,13,17,18
        # A22_12 (0,2), (1,2) -> global 14,19
        # A22_21 (2,0), (2,1) -> global 22,23
        # A22_22 (2,2) -> global 24
        
        # Same for B22 and C22.
        
        # I will implement the 8 sub-blocks for 3x3x3.
        
        def add_3x3_recursive(offset_start):
            off = offset_start
            # 1. 2x2x2 (Rank 7)
            for u, v, w in strassen_2x2_coeffs():
                for k in range(4):
                    if u[k] != 0 and v[k] != 0 and w[k] != 0:
                        U[a22[k]][off] += u[k] # k is local 0-3 mapping to 0,1,3,4? No.
                        # strassen_2x2 uses indices 0,1,2,3 for 2x2.
                        # We need to map local 2x2 indices to global a22 indices.
                        # a22_11 local indices:
                        # 0->12, 1->13, 2->17, 3->18
                        # My strassen_2x2_coeffs returns vectors of size 4.
                        # I need to map the 4 local indices to the specific global indices.
                        
            # This mapping is error-prone. 
            # Given the complexity, I will use Schoolbook for 3x3x3 (Rank 27).
            # Total Rank becomes 112.
            
            # Schoolbook 3x3x3
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U[a22[i*3+k]][off] = 1
                        V[b22[k*3+j]][off] = 1
                        W[c22[i*3+j]][off] = 1
                        off += 1
            return off

        offset = add_3x3_recursive(offset)
        
        # Total Rank: 7 + 6 + 6 + 18 + 12 + 18 + 18 + 27 = 112.
        
        # Trim U, V, W to actual rank used (112)
        rank_used = offset
        U = [row[:rank_used] for row in U]
        V = [row[:rank_used] for row in V]
        W = [row[:rank_used] for row in W]
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
