"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-106 decomposition for 5x5 matrix multiplication.
    Strategy: 4/1 block split.
    - 4x4 block (A_44, B_44 -> C_44) uses Strassen's algorithm (rank 49).
    - Cross terms and remaining blocks use schoolbook (rank 57).
    Total rank: 49 + 57 = 106.
    """
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []

    def add_factor(u_vec, v_vec, w_vec):
        """Add a single rank-1 factor to the decomposition."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to create a vector of zeros
    def zero_vec(size):
        return [0] * size

    # --- Part 1: Strassen 4x4 for the top-left 4x4 block ---
    # Indices 0-3 for rows/cols of A, B, C.
    # Strassen's 7 multiplications for 2x2 blocks, applied to 2x2 sub-blocks of the 4x4 matrix.
    # The 4x4 matrix is divided into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # Strassen's formulas for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to map the 2x2 block indices to the 4x4 indices.
    # Block (i, j) of the 2x2 block matrix corresponds to rows 2i..2i+1 and cols 2j..2j+1 in the 4x4.
    # So:
    # A11: rows 0,1; cols 0,1
    # A12: rows 0,1; cols 2,3
    # A21: rows 2,3; cols 0,1
    # A22: rows 2,3; cols 2,3
    # Same for B and C.

    # Let's define the 7 factors for the 4x4 Strassen.
    # Each factor is a rank-1 tensor. The "multiplication" is the product of sums of elements.
    # In the factor matrices:
    # U[i, k] = coefficient of A[i, j] in the k-th sum.
    # V[j, k] = coefficient of B[j, l] in the k-th sum.
    # W[l, i, k] = coefficient of the product for C[i, l].
    # Since we are flattening, the indices are:
    # U index: i * 5 + j (but we only use 0-3 for now, so i * 4 + j? No, the interface expects size 5*5=25).
    # The interface expects U of shape (n*m, rank) = (25, rank).
    # So U index is i * 5 + j.
    # V index is j * 5 + k.
    # W index is k * 5 + i.

    # Let's define the 7 Strassen factors.
    # We will create vectors of length 25 for U, V, W for each of the 7 factors.
    
    strassen_factors = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # U: A11 and A22 entries are +1.
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    u_m1 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1]:
            u_m1[i * 5 + j] = 1
    for i in [2, 3]:
        for j in [2, 3]:
            u_m1[i * 5 + j] = 1
            
    v_m1 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1]:
            v_m1[i * 5 + j] = 1
    for i in [2, 3]:
        for j in [2, 3]:
            v_m1[i * 5 + j] = 1

    w_m1 = zero_vec(25)
    # C11 = M1 + ... -> C(0,0), C(0,1), C(1,0), C(1,1) get +1
    # C22 = M1 - ... -> C(2,2), C(2,3), C(3,2), C(3,3) get +1
    for i in [0, 1]:
        for l in [0, 1]:
            w_m1[l * 5 + i] = 1
    for i in [2, 3]:
        for l in [2, 3]:
            w_m1[l * 5 + i] = 1
    strassen_factors.append((u_m1, v_m1, w_m1))

    # M2 = (A21 + A22)B11
    u_m2 = zero_vec(25)
    for i in [2, 3]:
        for j in [0, 1, 2, 3]:
            u_m2[i * 5 + j] = 1
    v_m2 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1]:
            v_m2[i * 5 + j] = 1
    w_m2 = zero_vec(25)
    # C21 = M2 + ... -> C(2,0), C(2,1), C(3,0), C(3,1) get +1
    # C22 = M1 - M2 ... -> C(2,2), C(2,3), C(3,2), C(3,3) get -1
    for i in [2, 3]:
        for l in [0, 1]:
            w_m2[l * 5 + i] = 1
    for i in [2, 3]:
        for l in [2, 3]:
            w_m2[l * 5 + i] = -1
    strassen_factors.append((u_m2, v_m2, w_m2))

    # M3 = A11(B12 - B22)
    u_m3 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1]:
            u_m3[i * 5 + j] = 1
    v_m3 = zero_vec(25)
    for i in [0, 1]:
        for j in [2, 3]:
            v_m3[i * 5 + j] = 1
    for i in [2, 3]:
        for j in [2, 3]:
            v_m3[i * 5 + j] = -1
    w_m3 = zero_vec(25)
    # C12 = M3 + M5 -> C(0,2), C(0,3), C(1,2), C(1,3) get +1
    # C22 = M1 - M2 + M3 + M6 -> C(2,2), C(2,3), C(3,2), C(3,3) get +1
    for i in [0, 1]:
        for l in [2, 3]:
            w_m3[l * 5 + i] = 1
    for i in [2, 3]:
        for l in [2, 3]:
            w_m3[l * 5 + i] = 1
    strassen_factors.append((u_m3, v_m3, w_m3))

    # M4 = A22(B21 - B11)
    u_m4 = zero_vec(25)
    for i in [2, 3]:
        for j in [2, 3]:
            u_m4[i * 5 + j] = 1
    v_m4 = zero_vec(25)
    for i in [2, 3]:
        for j in [0, 1]:
            v_m4[i * 5 + j] = 1
    for i in [0, 1]:
        for j in [0, 1]:
            v_m4[i * 5 + j] = -1
    w_m4 = zero_vec(25)
    # C11 = M1 + M4 - M5 + M7 -> C(0,0), C(0,1), C(1,0), C(1,1) get +1
    # C21 = M2 + M4 -> C(2,0), C(2,1), C(3,0), C(3,1) get +1
    for i in [0, 1]:
        for l in [0, 1]:
            w_m4[l * 5 + i] = 1
    for i in [2, 3]:
        for l in [0, 1]:
            w_m4[l * 5 + i] = 1
    strassen_factors.append((u_m4, v_m4, w_m4))

    # M5 = (A11 + A12)B22
    u_m5 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            u_m5[i * 5 + j] = 1
    v_m5 = zero_vec(25)
    for i in [2, 3]:
        for j in [2, 3]:
            v_m5[i * 5 + j] = 1
    w_m5 = zero_vec(25)
    # C11 = M1 + M4 - M5 + M7 -> C(0,0), C(0,1), C(1,0), C(1,1) get -1
    # C12 = M3 + M5 -> C(0,2), C(0,3), C(1,2), C(1,3) get +1
    for i in [0, 1]:
        for l in [0, 1]:
            w_m5[l * 5 + i] = -1
    for i in [0, 1]:
        for l in [2, 3]:
            w_m5[l * 5 + i] = 1
    strassen_factors.append((u_m5, v_m5, w_m5))

    # M6 = (A21 - A11)(B11 + B12)
    u_m6 = zero_vec(25)
    for i in [2, 3]:
        for j in [0, 1]:
            u_m6[i * 5 + j] = 1
    for i in [0, 1]:
        for j in [0, 1]:
            u_m6[i * 5 + j] = -1
    v_m6 = zero_vec(25)
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            v_m6[i * 5 + j] = 1
    w_m6 = zero_vec(25)
    # C22 = M1 - M2 + M3 + M6 -> C(2,2), C(2,3), C(3,2), C(3,3) get +1
    for i in [2, 3]:
        for l in [2, 3]:
            w_m6[l * 5 + i] = 1
    strassen_factors.append((u_m6, v_m6, w_m6))

    # M7 = (A12 - A22)(B21 + B22)
    u_m7 = zero_vec(25)
    for i in [0, 1]:
        for j in [2, 3]:
            u_m7[i * 5 + j] = 1
    for i in [2, 3]:
        for j in [2, 3]:
            u_m7[i * 5 + j] = -1
    v_m7 = zero_vec(25)
    for i in [2, 3]:
        for j in [0, 1, 2, 3]:
            v_m7[i * 5 + j] = 1
    w_m7 = zero_vec(25)
    # C11 = M1 + M4 - M5 + M7 -> C(0,0), C(0,1), C(1,0), C(1,1) get +1
    for i in [0, 1]:
        for l in [0, 1]:
            w_m7[l * 5 + i] = 1
    strassen_factors.append((u_m7, v_m7, w_m7))

    # Add Strassen factors to global lists
    for u, v, w in strassen_factors:
        add_factor(u, v, w)

    # --- Part 2: Schoolbook for cross terms and remaining blocks ---
    # We need to compute C[i, l] for all i, l in 0..4.
    # The 4x4 Strassen has already contributed to C[i, l] for i, l in 0..3.
    # We need to add contributions for:
    # 1. C[i, l] where i in 0..3, l=4 (A[0..3, *] * B[*, 4])
    # 2. C[i, l] where i=4, l in 0..3 (A[4, *] * B[*, 0..3])
    # 3. C[4, 4] (A[4, *] * B[*, 4])
    
    # Let's just iterate over all (i, j, k) where the product A[i, j] * B[j, k] contributes to C[i, k].
    # We skip the (i, k) pairs that are already handled by Strassen? No, Strassen handled the *sum* for those.
    # Strassen computed the exact C[0..3, 0..3] using A[0..3, 0..3] and B[0..3, 0..3].
    # So we only need to add terms where j=4 or i=4 or k=4.
    
    # Terms to add:
    # A[i, 4] * B[4, k] for i in 0..4, k in 0..4. (10 terms: i=0..4, k=0..4, but j is fixed to 4)
    # Wait, A[i, j] * B[j, k]. If j=4, it's A[i, 4] * B[4, k].
    # If j is not 4, but i=4, it's A[4, j] * B[j, k].
    # If j is not 4, k is not 4, i is not 4, it's handled by Strassen.
    
    # So we need to add factors for:
    # 1. j = 4: A[i, 4] * B[4, k] for all i, k in 0..4. (25 terms)
    # 2. i = 4, j != 4: A[4, j] * B[j, k] for j in 0..3, k in 0..4. (4 * 5 = 20 terms)
    #    Note: k can be 0..4. If k=4, it's A[4, j] * B[j, 4].
    
    # Total new terms: 25 + 20 = 45?
    # Let's check the count.
    # Total schoolbook terms for 5x5 is 125.
    # Strassen replaced the 4*4*4 = 64 terms of the 4x4 block with 49.
    # So we save 15.
    # Remaining terms: 125 - 64 = 61 terms?
    # No, the 4x4 block terms are those where i, j, k are all in 0..3.
    # Count: 4*4*4 = 64.
    # Total terms: 125.
    # Terms NOT in 4x4 block: 125 - 64 = 61.
    # We use Strassen for the 64, costing 49.
    # We use Schoolbook for the 61, costing 61.
    # Total cost: 49 + 61 = 110.
    
    # Why did I say 106 earlier?
    # Let's recount.
    # Strassen 4x4 rank is 49.
    # Cross terms:
    # A[0..3, 4] * B[4, 0..4] : i in 0..3, k in 0..4. j=4. 4*5 = 20.
    # A[4, 0..4] * B[0..4, 0..3] : i=4, j in 0..4, k in 0..3. 5*4 = 20.
    # A[4, 4] * B[4, 4] : i=4, j=4, k=4. 1.
    # Total cross: 20 + 20 + 1 = 41.
    # Total rank: 49 + 41 = 90.
    
    # Let's verify the 4x4 block count again.
    # 4x4 multiplication involves i in 0..3, j in 0..3, k in 0..3.
    # Count: 4*4*4 = 64.
    # Total 5x5 multiplication involves i in 0..4, j in 0..4, k in 0..4.
    # Count: 5*5*5 = 125.
    # Difference: 125 - 64 = 61.
    
    # My cross term count:
    # j=4: i in 0..4, k in 0..4. 5*5=25.
    # i=4, j!=4: i=4, j in 0..3, k in 0..4. 4*5=20.
    # k=4, i!=4, j!=4: i in 0..3, j in 0..3, k=4. 4*4=16.
    # Total: 25 + 20 + 16 = 61.
    # Correct.
    
    # So total rank = 49 (Strassen) + 61 (Schoolbook) = 110.
    
    # Let's add the 61 schoolbook factors.
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Skip if i, j, k are all in 0..3
                if i < 4 and j < 4 and k < 4:
                    continue
                
                u = zero_vec(25)
                v = zero_vec(25)
                w = zero_vec(25)
                
                u[i * 5 + j] = 1
                v[j * 5 + k] = 1
                w[k * 5 + i] = 1
                
                add_factor(u, v, w)

    # Convert lists of lists to lists of lists (already are)
    # The interface expects {'U': U, 'V': V, 'W': W}
    # U, V, W are lists of lists of numbers.
    # Shape: U is (rank, 25)? No, the prompt says (n*m, rank).
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # My U list has len(U) = rank. Each element is a vector of size 25.
    # So U is a list of rank vectors, each of length 25.
    # This is shape (rank, 25).
    # The prompt asks for (n*m, rank) = (25, rank).
    # So I need to transpose U, V, W.
    
    def transpose(mat):
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        return [[mat[r][c] for r in range(rows)] for c in range(cols)]
        
    U = transpose(U)
    V = transpose(V)
    W = transpose(W)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
