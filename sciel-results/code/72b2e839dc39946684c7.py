import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total rank
    rank = 85
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        # u_idx is index in U (0..19)
        # v_idx is index in V (0..24)
        # w_idx is index in W (0..19)
        pass # We will fill arrays directly

    # 1. 4x4x4 Strassen Part
    # Indices:
    # A: i in 0..3, j in 0..3. U index = i*5 + j
    # B: j in 0..3, k in 0..3. V index = j*5 + k
    # C: i in 0..3, k in 0..3. W index = k*4 + i (Trace convention W[k*n+i])
    
    # Strassen 2x2x2 Coefficients
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen Steps
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coeffs for A (U2)
    # Col 0 (M1): A00(0), A11(1)
    U2[0, 0] = 1; U2[3, 0] = 1
    # Col 1 (M2): A10(2), A11(3)
    U2[2, 1] = 1; U2[3, 1] = 1
    # Col 2 (M3): A00(0)
    U2[0, 2] = 1
    # Col 3 (M4): A11(3)
    U2[3, 3] = 1
    # Col 4 (M5): A00(0), A01(1)
    U2[0, 4] = 1; U2[1, 4] = 1
    # Col 5 (M6): A10(2), -A00(0)
    U2[2, 5] = 1; U2[0, 5] = -1
    # Col 6 (M7): A01(1), -A11(3)
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # Coeffs for B (V2)
    # Col 0 (M1): B00(0), B11(3)
    V2[0, 0] = 1; V2[3, 0] = 1
    # Col 1 (M2): B00(0)
    V2[0, 1] = 1
    # Col 2 (M3): B01(1), -B11(3)
    V2[1, 2] = 1; V2[3, 2] = -1
    # Col 3 (M4): B10(2), -B00(0)
    V2[2, 3] = 1; V2[0, 3] = -1
    # Col 4 (M5): B11(3)
    V2[3, 4] = 1
    # Col 5 (M6): B00(0), B01(1)
    V2[0, 5] = 1; V2[1, 5] = 1
    # Col 6 (M7): B10(2), B11(3)
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Coeffs for C (W2)
    # C00 = M1 + M4 - M5 + M7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5
    W2[1, 2] = 1; W2[1, 4] = 1
    # C10 = M2 + M4
    W2[2, 1] = 1; W2[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Embed 4x4x4
    # Blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in 2x2 block space
    # Global A indices:
    # A11: i=0,1 j=0,1
    # A12: i=0,1 j=2,3
    # A21: i=2,3 j=0,1
    # A22: i=2,3 j=2,3
    
    # Strassen Block Coeffs for 4x4x4
    # X_s and Y_s are 2x2 matrices of blocks.
    # We need to map the 2x2 Strassen structure to the 4x4 block structure.
    
    # Strassen Steps (X_s, Y_s) in terms of blocks (11, 12, 21, 22)
    # M1: (A11+A22)(B11+B22) -> X=[1,0;0,1], Y=[1,0;0,1]
    # M2: (A21+A22)B11 -> X=[0,0;1,1], Y=[1,0;0,0]
    # M3: A11(B12-B22) -> X=[1,0;0,0], Y=[0,1;0,-1]
    # M4: A22(B21-B11) -> X=[0,0;0,1], Y=[0,-1;1,0]
    # M5: (A11+A12)B22 -> X=[1,1;0,0], Y=[0,0;0,1]
    # M6: (A21-A11)(B11+B12) -> X=[-1,0;1,0], Y=[1,1;0,0]
    # M7: (A12-A22)(B21+B22) -> X=[0,1;0,-1], Y=[0,0;1,1]
    
    # For each step s, we have X_s (2x2 blocks) and Y_s (2x2 blocks).
    # Each block is a 2x2 matrix.
    # The product X_s Y_s is a 2x2 matrix of 2x2 blocks.
    # We compute this using the 2x2x2 Strassen tensor U2, V2, W2.
    
    # For each of the 7 steps:
    for s in range(7):
        # Get X_s and Y_s block coeffs
        # X_s is 2x2, entries are 0 or 1 (or -1)
        # We define X_s and Y_s explicitly
        if s == 0:
            Xs = np.array([[1, 0], [0, 1]])
            Ys = np.array([[1, 0], [0, 1]])
        elif s == 1:
            Xs = np.array([[0, 0], [1, 1]])
            Ys = np.array([[1, 0], [0, 0]])
        elif s == 2:
            Xs = np.array([[1, 0], [0, 0]])
            Ys = np.array([[0, 1], [0, -1]])
        elif s == 3:
            Xs = np.array([[0, 0], [0, 1]])
            Ys = np.array([[0, -1], [1, 0]])
        elif s == 4:
            Xs = np.array([[1, 1], [0, 0]])
            Ys = np.array([[0, 0], [0, 1]])
        elif s == 5:
            Xs = np.array([[-1, 0], [1, 0]])
            Ys = np.array([[1, 1], [0, 0]])
        elif s == 6:
            Xs = np.array([[0, 1], [0, -1]])
            Ys = np.array([[0, 0], [1, 1]])
            
        # Now, for each internal Strassen term alpha (0..6)
        for alpha in range(7):
            # u_coeff = Xs @ U2[:, alpha] ? No.
            # U2[:, alpha] gives coeffs for A blocks (11, 12, 21, 22)
            # Wait, U2 is for 2x2 matrix mult.
            # The "A" in the 2x2 mult is X_s.
            # The "B" in the 2x2 mult is Y_s.
            # The "C" in the 2x2 mult is the result block.
            
            # The term is:
            # sum_{i,j} (Xs_{ij} * U2_{block_idx, alpha}) * (Ys_{kl} * V2_{block_idx, alpha}) * W2_{res_idx, alpha}
            
            # Actually, simpler:
            # The rank-1 term for the 4x4x4 tensor corresponding to step s and internal alpha is:
            # U_4x4 = Xs (flattened and scaled by U2)
            # V_4x4 = Ys (flattened and scaled by V2)
            # W_4x4 = W2 (scaled)
            
            # Let's construct the vectors.
            # U4 is length 16 (4x4 matrix)
            # V4 is length 16
            # W4 is length 16
            
            # Xs is 2x2 blocks. Each block is 2x2.
            # U2 is 4x7. U2[i, alpha] is coeff for A_block_i.
            # A_block_0 is A11, A_block_1 is A12, A_block_2 is A21, A_block_3 is A22.
            
            # So, for a given alpha:
            # u_vec = sum_{b=0}^3 U2[b, alpha] * Xs[b] (where Xs[b] is the 2x2 block matrix)
            # Wait, Xs is the matrix of blocks.
            # Xs[0,0] is coeff for A11.
            # So the actual 4x4 matrix X is:
            # X = [ Xs[0,0]*A11  Xs[0,1]*A12
            #       Xs[1,0]*A21  Xs[1,1]*A22 ]
            
            # We need to express X as a linear combination of basis matrices E_ij.
            # The rank-1 term is X * Y * Z.
            # This is not a single rank-1 tensor in the global space unless X, Y, Z are rank-1 matrices?
            # No, the tensor decomposition allows X, Y, Z to be any vectors.
            # The term is $u \otimes v \otimes w$.
            # Here, the "vector" u is the flattened 4x4 matrix X.
            
            # So we just need to compute the flattened X, Y, Z.
            
            X_flat = np.zeros(16)
            Y_flat = np.zeros(16)
            Z_flat = np.zeros(16)
            
            # Map block index to 4x4 positions
            # Block 0 (11): rows 0-1, cols 0-1
            # Block 1 (12): rows 0-1, cols 2-3
            # Block 2 (21): rows 2-3, cols 0-1
            # Block 3 (22): rows 2-3, cols 2-3
            
            # For X:
            for b in range(4):
                if Xs[b//2, b%2] != 0:
                    coeff = Xs[b//2, b%2] * U2[b, alpha]
                    # Add to X_flat
                    # Block b in 4x4
                    r_start = (b//2)*2
                    c_start = (b%2)*2
                    for r in range(2):
                        for c in range(2):
                            idx = (r_start+r)*4 + (c_start+c)
                            X_flat[idx] = coeff # Since A_block is identity in basis?
                            # Wait, A is the input.
                            # The tensor term is A_{ij} ...
                            # So U_flat[idx] should be 1 if A_idx is part of X.
                            # But X is a linear combo of A blocks.
                            # So U_flat[idx] = coeff.
                            
            # For Y:
            for b in range(4):
                if Ys[b//2, b%2] != 0:
                    coeff = Ys[b//2, b%2] * V2[b, alpha]
                    r_start = (b//2)*2
                    c_start = (b%2)*2
                    for r in range(2):
                        for c in range(2):
                            idx = (r_start+r)*4 + (c_start+c)
                            Y_flat[idx] = coeff

            # For Z (C block):
            # Z is the 2x2 matrix of C blocks.
            # Z = W2[:, alpha] (2x2 matrix)
            # W2[0, alpha] is coeff for C11
            # W2[1, alpha] is coeff for C12
            # W2[2, alpha] is coeff for C21
            # W2[3, alpha] is coeff for C22
            
            for b in range(4):
                if W2[b, alpha] != 0:
                    coeff = W2[b, alpha]
                    r_start = (b//2)*2
                    c_start = (b%2)*2
                    for r in range(2):
                        for c in range(2):
                            # C index is k*n + i.
                            # C block b corresponds to global C indices.
                            # C11: i=0,1 k=0,1. W idx = k*4 + i.
                            # C12: i=0,1 k=2,3. W idx = k*4 + i.
                            # C21: i=2,3 k=0,1. W idx = k*4 + i.
                            # C22: i=2,3 k=2,3. W idx = k*4 + i.
                            
                            # r is local row (i_local), c is local col (k_local)
                            # Global i = r_start + r
                            # Global k = c_start + c
                            w_idx = (c_start + c)*4 + (r_start + r)
                            Z_flat[w_idx] = coeff

            # Add to global U, V, W
            # U index for A (4x4 part) is i*5 + j.
            # Our X_flat is 4x4, idx = i*4 + j.
            # We need to map X_flat[i*4+j] to U[i*5+j].
            for i in range(4):
                for j in range(4):
                    u_idx = i*5 + j
                    U[u_idx, s*7+alpha] = X_flat[i*4+j]
                    
            # V index for B (4x4 part) is j*5 + k.
            # Our Y_flat is 4x4, idx = j*4 + k.
            for j in range(4):
                for k in range(4):
                    v_idx = j*5 + k
                    V[v_idx, s*7+alpha] = Y_flat[j*4+k]
                    
            # W index for C (4x4 part) is k*4 + i.
            # Our Z_flat is 4x4, idx = k*4 + i.
            for k in range(4):
                for i in range(4):
                    w_idx = k*4 + i
                    W[w_idx, s*7+alpha] = Z_flat[k*4+i]

    # 2. 4x4x1 Part
    # A[0:4, 0:4] B[0:4, 4] -> C[0:4, 4]
    # C_i = sum_j A_ij B_j4
    # Terms: A_ij B_j4 C_i4
    # Rank 16
    col_offset = 49
    for i in range(4):
        for j in range(4):
            u_idx = i*5 + j
            v_idx = j*5 + 4 # B col 4
            w_idx = 4*4 + i # C col 4, row i -> k=4, i=i -> 4*4+i
            U[u_idx, col_offset + i*4 + j] = 1
            V[v_idx, col_offset + i*4 + j] = 1
            W[w_idx, col_offset + i*4 + j] = 1

    # 3. 4x1x5 Part
    # A[0:4, 4] B[4, 0:5] -> C[0:4, 0:5]
    # C_ij = A_i4 B_4j
    # Terms: A_i4 B_4j C_ij
    # Rank 20
    col_offset = 65
    for i in range(4):
        for j in range(5):
            u_idx = i*5 + 4 # A col 4
            v_idx = 4*5 + j # B row 4
            w_idx = j*4 + i # C col j, row i
            U[u_idx, col_offset + i*5 + j] = 1
            V[v_idx, col_offset + i*5 + j] = 1
            W[w_idx, col_offset + i*5 + j] = 1

    return {"U": U, "V": V, "W": W}
