import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition 5 = 4 + 1
    # Rank estimate: 49 (Strassen 4x4) + 1 (1x1) + 8 (cross 4x1) + 8 (cross 1x4) = 66
    
    rank = 66
    
    U = [[0.0] * rank for _ in range(n * m)] # 25 x 66
    V = [[0.0] * rank for _ in range(m * p)] # 25 x 66
    W = [[0.0] * rank for _ in range(p * n)] # 25 x 66
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            raise RuntimeError("Rank exceeded")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # 1. C11 = A11 * B11 (4x4) using Strassen (Rank 49)
    # A11 is indices 0..15 (4x4)
    # B11 is indices 0..15 (4x4)
    # C11 is indices 0..15 (4x4)
    
    # Strassen 4x4 decomposition into 7 2x2 mults.
    # We need to map the 2x2 blocks to the 4x4 indices.
    # Block (br, bc) of 2x2 in 4x4:
    # A block (br, bc) starts at row br*2, col bc*2.
    # A index: (br*2 + r) * 4 + (bc*2 + c)
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mk is a 2x2 multiplication.
    # We use Strassen for each 2x2 mult (Rank 7).
    # Total 7 * 7 = 49 terms.
    
    def add_strassen_2x2(a_base, b_base, c_base, offset_u, offset_v, offset_w):
        # a_base, b_base, c_base are the starting indices in the 4x4 blocks for the 2x2 sub-blocks?
        # No, this function adds the terms for ONE 2x2 multiplication.
        # The 2x2 A block is at some location in the 4x4 A.
        # Let's define the 2x2 block indices relative to the 4x4.
        pass

    # Let's explicitly define the 7 Strassen products for the 4x4 case.
    # The 4x4 A is composed of 4 2x2 blocks: A00, A01, A10, A11.
    # Indices in 4x4:
    # A00: (0,0)-(1,1) -> U indices 0,1,4,5
    # A01: (0,2)-(1,3) -> U indices 2,3,6,7
    # A10: (2,0)-(3,1) -> U indices 8,9,12,13
    # A11: (2,2)-(3,3) -> U indices 10,11,14,15
    
    # We will generate the 49 terms by iterating through the 7 Strassen steps
    # and for each, decomposing the 2x2 mult into 7 rank-1 terms.
    
    # To simplify, I will hardcode the 49 terms for the 4x4 Strassen block.
    # Or better, write a helper that takes the U, V, W indices for the 2x2 inputs and outputs
    # and adds the 7 terms.
    
    def add_2x2_mult(u_offsets, v_offsets, w_offsets):
        # u_offsets: list of 4 U indices for the 2x2 A block (row-major)
        # v_offsets: list of 4 V indices for the 2x2 B block
        # w_offsets: list of 4 W indices for the 2x2 C block
        # Adds 7 terms for Strassen 2x2.
        nonlocal col
        
        # Strassen 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        add_term(u_offsets[0], v_offsets[0], w_offsets[0], 1, 1, 1) # a00 b00 -> c00
        add_term(u_offsets[0], v_offsets[1], w_offsets[0], 1, -1, 0) # a00 b01? No.
        
        # This is getting messy to hardcode offsets.
        # Let's just use the schoolbook for the 4x4 block?
        # Schoolbook 4x4 is 16 terms.
        # Strassen is 49 terms.
        # Schoolbook is MUCH better for 4x4?
        # Wait.
        # Schoolbook rank for 4x4 is 16.
        # Strassen rank for 4x4 is 49.
        # Why did I think Strassen was better?
        # Strassen reduces 8 to 7 for 2x2.
        # For 4x4, Strassen uses 7 mults of 2x2.
        # Each 2x2 mult is 7 rank-1 terms.
        # Total 49.
        # Schoolbook 4x4 is 16 rank-1 terms.
        # So Schoolbook is BETTER for 4x4!
        
        # My mistake.
        # R(4) = 16?
        # No, R(4) is the rank of the 4x4 tensor.
        # Schoolbook is 16.
        # Is there an algorithm with rank < 16?
        # R(4) is known to be 26? No, 26 is upper bound?
        # Actually, R(4) is at least 16?
        # If Schoolbook is 16, and R(4) >= 16, then R(4)=16?
        # No, R(4) is not 16.
        # Wait, Schoolbook rank is m*n*p?
        # For 4x4x4, schoolbook is 16?
        # No, schoolbook is m*n*p?
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # There are 16 pairs (i,k). Each requires 4 mults.
        # Total 64 mults.
        # Rank 64.
        
        # Ah! Schoolbook rank is $m \times n \times p$.
        # For 4x4, it is 64.
        # Strassen is 49.
        # So Strassen is better.
        
        # Okay, back to Strassen 49.
        
        # Let's implement the 49 terms properly.
        
        # I will use a generic Strassen 2x2 decomposer.
        
        def strassen_2x2(u_idx, v_idx, w_idx):
            # u_idx: [u00, u01, u10, u11]
            # v_idx: [v00, v01, v10, v11]
            # w_idx: [w00, w01, w10, w11]
            
            # M1 = (a00+a11)(b00+b11)
            # We need to form linear combos.
            # The rank-1 term is u * v * w.
            # If we use (a00+a11), the U vector has 1 at u00 and 1 at u11.
            
            # Term 1: (a00+a11) * (b00+b11) * 1 -> contributes to c00, c11 etc?
            # No, a rank-1 term $u \otimes v \otimes w$ contributes $u_i v_j w_k$ to $T_{ijk}$.
            # For matrix mult, $T_{ijk}$ is 1 if $C_{ik}$ depends on $A_{ij} B_{jk}$.
            # Actually, the tensor is $T_{i,j,k}$ such that $C_{ik} = \sum_j T_{i,j,k} A_{ij} B_{jk}$.
            # In our decomposition, we are summing rank-1 tensors.
            
            # Standard Strassen 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Each M is a scalar product of linear combos.
            # M1: u = [1,0,0,1], v = [1,1,0,0] (if v is row-major b00,b01,b10,b11? No, v is B indices).
            # Let's say u indices are [u00, u01, u10, u11].
            # u_vec = [1, 0, 0, 1] (a00 + a11)
            # v_vec = [1, 1, 0, 0] (b00 + b11) -> Wait, b00 is v_idx[0], b11 is v_idx[3].
            # So v_vec = [1, 0, 0, 1].
            
            # Term M1:
            # U: 1 at u00, 1 at u11.
            # V: 1 at v00, 1 at v11.
            # W: 1 at w00, 1 at w11?
            # No, the W part of the rank-1 term determines which C elements get the product.
            # In Strassen, the product M1 is added to C00 and C11.
            # So W vector should have 1 at w00 and 1 at w11.
            
            add_term(u_idx[0], v_idx[0], w_idx[0], 1, 1, 1) # a00 b00 -> c00
            add_term(u_idx[0], v_idx[3], w_idx[0], 1, 1, 0) # a00 b11 -> c00? No.
            
            # This manual expansion is error-prone.
            # Let's just use the schoolbook for the 4x4 block?
            # Rank 64.
            # Total: 64 + 1 + 8 + 8 = 81.
            # 81 is better than 125.
            
            # Can we do better?
            # Use Strassen for 4x4? Rank 49.
            # Total 66.
            
            # I will implement Strassen 4x4 using a recursive call to Strassen 2x2.
            
            # For now, let's just return the schoolbook 125 to ensure correctness,
            # then I'll try to patch in the 4x4 Strassen.
            
            # Actually, the prompt asks for a rewrite.
            # I will implement the 5=3+2 split with schoolbook cross terms (Rank 70).
            # It is simpler than 4x4 Strassen.
            
            pass

    # Let's restart with the 5=3+2 split.
    # Rank 70.
    
    rank = 70
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_term(u, v, w, u_val, v_val, w_val):
        nonlocal col
        U[u][col] = u_val
        V[v][col] = v_val
        W[w][col] = w_val
        col += 1

    # 1. C11 (3x3) Laderman
    # Laderman 3x3 terms:
    # m1 = a11 b11
    # m2 = a22 b22
    # m3 = a33 b33
    # m4 = (a12+a22)(b21+b11)
    # m5 = (a23+a33)(b32+b22)
    # m6 = (a31+a11)(b13+b33)
    # m7 = (a12+a11)(b22+b12)
    # m8 = (a23+a22)(b33+b23)
    # m9 = (a31+a33)(b11+b31)
    # m10 = (a12+a22+a11+a33)(b21+b11+b33+b32) -- No, standard Laderman has 15 terms.
    
    # I will use a known list of 15 terms for 3x3.
    # Or just use schoolbook 27?
    # 27 + 4 + 24 + 24 = 79.
    # Laderman 15 + 4 + 24 + 24 = 67.
    # Strassen 2x2 is 7.
    # 15 + 7 + 24 + 24 = 70.
    
    # Let's use Laderman 15 for C11.
    
    # Laderman 3x3 indices:
    # A: 0..8, B: 0..8, C: 0..8
    
    # I will hardcode the 15 terms.
    
    # This is getting long. I will use the schoolbook for everything to be safe,
    # but that's 125.
    
    # Let's try the 5=4+1 split with Schoolbook 4x4 (Rank 64).
    # Total 81.
    
    rank = 81
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # 1. C11 (4x4) Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 4 + i
                add_term(u_idx, v_idx, w_idx, 1, 1, 1)
                
    # 2. C22 (1x1)
    u_idx = 4 * 4 + 4 # A[4,4] index 24
    v_idx = 4 * 4 + 4 # B[4,4] index 24
    w_idx = 4 * 4 + 4 # C[4,4] index 24
    add_term(u_idx, v_idx, w_idx, 1, 1, 1)
    
    # 3. C12 (4x1)
    # C[i,4] = sum A[i,j] B[j,4] + A[i,4] B[4,4]
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 4 + 4
            w_idx = 4 * 4 + i
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
        # A[i,4] B[4,4]
        u_idx = i * 4 + 4
        v_idx = 4 * 4 + 4
        w_idx = 4 * 4 + i
        add_term(u_idx, v_idx, w_idx, 1, 1, 1)
        
    # 4. C21 (1x4)
    # C[4,k] = sum A[4,j] B[j,k] + A[4,4] B[4,k]
    for k in range(4):
        for j in range(4):
            u_idx = 4 * 4 + j
            v_idx = j * 4 + k
            w_idx = k * 4 + 4
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
        # A[4,4] B[4,k]
        u_idx = 4 * 4 + 4
        v_idx = 4 * 4 + k
        w_idx = k * 4 + 4
        add_term(u_idx, v_idx, w_idx, 1, 1, 1)
        
    return {"U": U, "V": V, "W": W}

