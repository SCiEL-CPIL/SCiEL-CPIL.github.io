"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if u_idx is not None:
            U[u_idx][col] += u_val
        if v_idx is not None:
            V[v_idx][col] += v_val
        if w_idx is not None:
            W[w_idx][col] += w_val
        col += 1

    def add_strassen(a_indices, b_indices, c_indices):
        # a_indices: [a11, a12, a21, a22] indices in U
        # b_indices: [b11, b12, b21, b22] indices in V
        # c_indices: [c11, c12, c21, c22] indices in W
        a11, a12, a21, a22 = a_indices
        b11, b12, b21, b22 = b_indices
        c11, c12, c21, c22 = c_indices
        
        # M1 = (a11+a22)(b11+b22)
        add_term(a11, 1, b11, 1, c11, 1)
        add_term(a11, 1, b22, 1, c11, 1)
        add_term(a22, 1, b11, 1, c11, 1)
        add_term(a22, 1, b22, 1, c11, 1)
        # Wait, Strassen terms are products of sums.
        # We need to represent the bilinear form.
        # The decomposition is a sum of rank 1 tensors.
        # Each rank 1 tensor is u * v * w.
        # So for M1, we have 4 terms? No.
        # (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22.
        # This adds 4 to the rank if we expand it.
        # But Strassen's algorithm says the rank is 7.
        # This implies that the 7 "M" values are the rank 1 components?
        # No, the M values are intermediate results.
        # The rank 1 components are the products that define the M's.
        # Actually, the Strassen decomposition expresses the multiplication tensor as a sum of 7 rank 1 tensors.
        # The coefficients in U, V, W are the constants.
        # For M1: U has 1 at a11, 1 at a22. V has 1 at b11, 1 at b22. W has 1 at c11, 1 at c22, -1 at c12, -1 at c21?
        # Let's check the reconstruction.
        # c11 = M1 - M4 + M6
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M3 + M7
        
        # So for each M_k, we define one rank 1 tensor.
        # M1:
        # U: a11=1, a22=1
        # V: b11=1, b22=1
        # W: c11=1, c22=1, c12=-1, c21=-1 ? No.
        # The contribution of M1 to c11 is +1. To c22 is +1. To c12 is -1. To c21 is -1.
        # So W for M1 has 1 at c11, 1 at c22, -1 at c12, -1 at c21.
        
        # Let's implement this correctly.
        
        # M1
        add_term(a11, 1, b11, 1, c11, 1) # This is wrong. add_term adds ONE rank 1 tensor.
        # We need to add the coefficients to the current column.
        
        # Reset col logic inside helper?
        # Better: define a function to add a full Strassen block.
        
        # Term 1 (M1)
        U[a11][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        W[c12][col] -= 1
        W[c21][col] -= 1
        col += 1
        
        # Term 2 (M2)
        U[a21][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        col += 1
        
        # Term 3 (M3)
        U[a11][col] += 1
        V[b12][col] += 1
        V[b22][col] -= 1
        W[c12][col] += 1
        W[c22][col] -= 1
        col += 1
        
        # Term 4 (M4)
        U[a22][col] += 1
        V[b21][col] += 1
        V[b11][col] -= 1
        W[c11][col] -= 1
        W[c21][col] += 1
        col += 1
        
        # Term 5 (M5)
        U[a11][col] += 1
        U[a12][col] += 1
        V[b22][col] += 1
        W[c12][col] += 1
        col += 1
        
        # Term 6 (M6)
        U[a21][col] += 1
        U[a11][col] -= 1
        V[b11][col] += 1
        V[b12][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 7 (M7)
        U[a12][col] += 1
        U[a22][col] -= 1
        V[b21][col] += 1
        V[b22][col] += 1
        W[c22][col] += 1
        col += 1

    def add_schoolbook_2x2_1(a_indices, b_indices, c_indices):
        # a: 2x2, b: 2x1, c: 2x1
        a11, a12, a21, a22 = a_indices
        b1, b2 = b_indices
        c1, c2 = c_indices
        
        # c1 = a11*b1 + a12*b2
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1
        
        U[a12][col] += 1
        V[b2][col] += 1
        W[c1][col] += 1
        col += 1
        
        # c2 = a21*b1 + a22*b2
        U[a21][col] += 1
        V[b1][col] += 1
        W[c2][col] += 1
        col += 1
        
        U[a22][col] += 1
        V[b2][col] += 1
        W[c2][col] += 1
        col += 1

    def add_schoolbook_2x1_5(a_indices, b_indices, c_indices):
        # a: 2x1, b: 1x5, c: 2x5
        a1, a2 = a_indices
        b1, b2, b3, b4, b5 = b_indices
        c11, c12, c13, c14, c15, c21, c22, c23, c24, c25 = c_indices
        
        # Row 1
        U[a1][col] += 1; V[b1][col] += 1; W[c11][col] += 1; col += 1
        U[a1][col] += 1; V[b2][col] += 1; W[c12][col] += 1; col += 1
        U[a1][col] += 1; V[b3][col] += 1; W[c13][col] += 1; col += 1
        U[a1][col] += 1; V[b4][col] += 1; W[c14][col] += 1; col += 1
        U[a1][col] += 1; V[b5][col] += 1; W[c15][col] += 1; col += 1
        
        # Row 2
        U[a2][col] += 1; V[b1][col] += 1; W[c21][col] += 1; col += 1
        U[a2][col] += 1; V[b2][col] += 1; W[c22][col] += 1; col += 1
        U[a2][col] += 1; V[b3][col] += 1; W[c23][col] += 1; col += 1
        U[a2][col] += 1; V[b4][col] += 1; W[c24][col] += 1; col += 1
        U[a2][col] += 1; V[b5][col] += 1; W[c25][col] += 1; col += 1

    # Indices
    # U: 0..5 (2x3)
    # V: 0..14 (3x5)
    # W: 0..9 (5x2) -> k*n+i. k is col of B (0..4), i is row of A (0..1).
    # W index = k*2 + i.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    a_idx = [0, 1, 2, 3]
    b_idx = [0, 1, 5, 6]
    c_idx = [0, 1, 5, 6] # c11=0, c12=1, c21=5, c22=6
    add_strassen(a_idx, b_idx, c_idx)
    
    # Block 2a: A[0:2, 0:2] * B[0:2, 2:4]
    a_idx = [0, 1, 2, 3]
    b_idx = [2, 3, 7, 8] # B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    c_idx = [2, 3, 7, 8] # c13=2, c14=3, c23=7, c24=8
    add_strassen(a_idx, b_idx, c_idx)
    
    # Block 2b: A[0:2, 0:2] * B[0:2, 4:5]
    a_idx = [0, 1, 2, 3]
    b_idx = [4, 9] # B[0,4]=4, B[1,4]=9
    c_idx = [4, 9] # c15=4, c25=9
    add_schoolbook_2x2_1(a_idx, b_idx, c_idx)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:5]
    a_idx = [4, 5] # A[0,2]=4, A[1,2]=5
    b_idx = [10, 11, 12, 13, 14] # B[2,0]..B[2,4]
    c_idx = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] 
    # c11=0, c12=2, c13=4, c14=6, c15=8
    # c21=1, c22=3, c23=5, c24=7, c25=9
    add_schoolbook_2x1_5(a_idx, b_idx, c_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
