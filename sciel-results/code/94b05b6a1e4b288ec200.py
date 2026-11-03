import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # U, V, W matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 definitions
    # X is 2x2, Y is 2x2
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y01 - y11)
    # S4 = x11(y10 - y00)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    
    # Coefficients for X entries (x00, x01, x10, x11)
    # Coefficients for Y entries (y00, y01, y10, y11)
    # Indexing: 0:00, 1:01, 2:10, 3:11
    
    # Strassen step s (1-7)
    # u_x[s][x_idx], u_y[s][y_idx]
    strassen_x = [
        [1, 0, 0, 1], # S1
        [0, 0, 1, 1], # S2
        [1, 0, 0, 0], # S3
        [0, 0, 0, 1], # S4
        [1, 1, 0, 0], # S5
        [1, 0, 1, 0], # S6 (x10 - x00 -> -1, 1) wait. x10 is idx 2, x00 is idx 0.
        [0, 1, 0, -1] # S7 (x01 - x11 -> 1, -1)
    ]
    # Correct S6: (x10 - x00). x10 is idx 2, x00 is idx 0. So [-1, 0, 1, 0].
    strassen_x[5] = [-1, 0, 1, 0]
    
    strassen_y = [
        [1, 0, 0, 1], # S1
        [1, 0, 0, 0], # S2
        [0, 1, 0, -1], # S3
        [0, 0, 1, -1], # S4
        [0, 0, 0, 1], # S5
        [1, 1, 0, 0], # S6
        [0, 0, 1, 1]  # S7
    ]
    
    # Result Z entries (z00, z01, z10, z11) in terms of S1..S7
    # z00 = S1 + S4 - S5 + S7
    # z01 = S3 + S5
    # z10 = S2 + S4
    # z11 = S1 - S2 + S3 + S6
    strassen_z = [
        [1, 0, 0, 1, -1, 0, 1], # z00
        [0, 0, 1, 0, 1, 0, 0], # z01
        [0, 1, 0, 1, 0, 0, 0], # z10
        [1, -1, 1, 0, 0, 1, 0] # z11
    ]
    
    # Outer Strassen for 4x4
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Block indices: 0:11, 1:12, 2:21, 3:22
    # X_l entries (2x2 matrix of blocks)
    # Y_l entries (2x2 matrix of blocks)
    
    outer_x = [
        [[1,0,0,1], [0,0,0,0], [0,0,0,0], [0,0,0,0]], # M1: A11+A22 (only top-left block is sum? No, X1 is a 2x2 matrix of scalars? No, X1 is a 2x2 matrix of 2x2 blocks?
        # Wait. X1 = A11 + A22. This is a 2x2 matrix.
        # In the block multiplication, we treat A as 2x2 blocks.
        # X1 is the matrix A11+A22.
        # So X1 is a 2x2 matrix of scalars (relative to the block level).
        # But we need to decompose X1 * Y1 into scalar multiplications.
        # X1 and Y1 are 2x2 matrices.
        # So we apply the inner Strassen to X1 and Y1.
        # The entries of X1 are linear combinations of A entries.
        
        # Let's define the linear forms for the entries of X_l and Y_l directly in terms of A and B entries.
        # A entries: a[i][j] for i,j in 0..3.
        # Block A11: a[0][0], a[0][1], a[1][0], a[1][1].
        # Block A12: a[0][2], a[0][3], a[1][2], a[1][3].
        # Block A21: a[2][0], a[2][1], a[3][0], a[3][1].
        # Block A22: a[2][2], a[2][3], a[3][2], a[3][3].
        
        # For each l, X_l is a 2x2 matrix.
        # X_l[0][0], X_l[0][1], X_l[1][0], X_l[1][1].
        # Each is a linear form in A.
        
        # M1: X1 = A11 + A22.
        # X1[0][0] = a00 + a22
        # X1[0][1] = a01 + a23
        # X1[1][0] = a10 + a32
        # X1[1][1] = a11 + a33
        
        # M2: X2 = A21 + A22.
        # X2[0][0] = a20 + a22
        # X2[0][1] = a21 + a23
        # X2[1][0] = a30 + a32
        # X2[1][1] = a31 + a33
        
        # M3: X3 = A11.
        # X3[0][0] = a00
        # X3[0][1] = a01
        # X3[1][0] = a10
        # X3[1][1] = a11
        
        # M4: X4 = A22.
        # X4[0][0] = a22
        # X4[0][1] = a23
        # X4[1][0] = a32
        # X4[1][1] = a33
        
        # M5: X5 = A11 + A12.
        # X5[0][0] = a00
        # X5[0][1] = a01
        # X5[1][0] = a10
        # X5[1][1] = a11
        # Wait, A11+A12 is not a block sum in that sense.
        # A = [[A11, A12], [A21, A22]].
        # X5 = A11 + A12? No, Strassen formula:
        # M5 = (A11 + A12) B22.
        # A11 + A12 is a 2x2 matrix?
        # A11 is 2x2, A12 is 2x2.
        # A11 + A12 = [[a00+a02, a01+a03], [a10+a12, a11+a13]].
        
        # M6: X6 = A21 - A11.
        # X6 = [[a20-a00, a21-a01], [a30-a10, a31-a11]].
        
        # M7: X7 = A12 - A22.
        # X7 = [[a02-a22, a03-a23], [a12-a32, a13-a33]].
        
        # Similarly for Y_l.
        # Y1 = B11 + B22
        # Y2 = B11
        # Y3 = B12 - B22
        # Y4 = B21 - B11
        # Y5 = B22
        # Y6 = B11 + B12
        # Y7 = B21 + B22
    ]
    
    # This manual expansion is error-prone. Let's use arrays.
    
    # Indices for A: 0..15. i*4+j
    # Indices for B: 0..15. j*4+k
    
    # Define linear forms for X_l entries (4 entries) and Y_l entries (4 entries)
    # Each entry is a vector of length 16 (for A or B)
    
    def get_block(A, r, c):
        # r, c in 0..1
        # Returns 4x4 slice? No, 2x2 block of 4x4.
        # A is flattened 16.
        # Block (r,c) covers rows 2r, 2r+1 and cols 2c, 2c+1.
        pass

    # Let's just hardcode the 49 terms using the structure.
    # It's easier to generate the 49 (U,V,W) triples by iterating l, s.
    
    # For each l in 1..7:
    #   Determine X_l (2x2 matrix of linear forms in A)
    #   Determine Y_l (2x2 matrix of linear forms in B)
    #   For each s in 1..7:
    #     Determine u_x (linear form in X_l entries)
    #     Determine u_y (linear form in Y_l entries)
    #     u_A = u_x * X_l (composition of linear forms)
    #     u_B = u_y * Y_l (composition of linear forms)
    #     Determine w_C (linear form in Z_l entries) where Z_l = X_l Y_l
    #     Actually, w_C is determined by how M_l contributes to C blocks.
    
    # Let's define the linear forms for X_l and Y_l entries as vectors of length 16.
    
    # A indices: i*4+j
    # B indices: j*4+k
    
    # X_l entries: 4 vectors of length 16.
    # Y_l entries: 4 vectors of length 16.
    
    # Helper to create a vector for a single A entry
    def a_vec(i, j):
        v = [0]*16
        v[i*4+j] = 1
        return v
    
    def b_vec(j, k):
        v = [0]*16
        v[j*4+k] = 1
        return v
    
    def add(v1, v2):
        return [x+y for x,y in zip(v1,v2)]
    
    def sub(v1, v2):
        return [x-y for x,y in zip(v1,v2)]
    
    # Define X_l and Y_l
    # X1 = A11 + A22
    X1 = [
        add(a_vec(0,0), a_vec(2,2)), add(a_vec(0,1), a_vec(2,3)),
        add(a_vec(1,0), a_vec(3,2)), add(a_vec(1,1), a_vec(3,3))
    ]
    # X2 = A21 + A22
    X2 = [
        add(a_vec(2,0), a_vec(2,2)), add(a_vec(2,1), a_vec(2,3)),
        add(a_vec(3,0), a_vec(3,2)), add(a_vec(3,1), a_vec(3,3))
    ]
    # X3 = A11
    X3 = [a_vec(0,0), a_vec(0,1), a_vec(1,0), a_vec(1,1)]
    # X4 = A22
    X4 = [a_vec(2,2), a_vec(2,3), a_vec(3,2), a_vec(3,3)]
    # X5 = A11 + A12
    X5 = [
        a_vec(0,0), add(a_vec(0,1), a_vec(0,2)),
        a_vec(1,0), add(a_vec(1,1), a_vec(1,2))
        # Wait, A11+A12 means adding the blocks.
        # A11 is top-left, A12 is top-right.
        # So X5[0][0] = a00, X5[0][1] = a01 + a02? No.
        # A = [[A11, A12], [A21, A22]].
        # A11+A12 is a 2x2 matrix?
        # A11 = [[a00, a01], [a10, a11]]
        # A12 = [[a02, a03], [a12, a13]]
        # A11+A12 = [[a00+a02, a01+a03], [a10+a12, a11+a13]]
        # So X5[0][0] = a00+a02, X5[0][1] = a01+a03, etc.
    ]
    X5 = [
        add(a_vec(0,0), a_vec(0,2)), add(a_vec(0,1), a_vec(0,3)),
        add(a_vec(1,0), a_vec(1,2)), add(a_vec(1,1), a_vec(1,3))
    ]
    # X6 = A21 - A11
    X6 = [
        sub(a_vec(2,0), a_vec(0,0)), sub(a_vec(2,1), a_vec(0,1)),
        sub(a_vec(3,0), a_vec(1,0)), sub(a_vec(3,1), a_vec(1,1))
    ]
    # X7 = A12 - A22
    X7 = [
        sub(a_vec(0,2), a_vec(2,2)), sub(a_vec(0,3), a_vec(2,3)),
        sub(a_vec(1,2), a_vec(3,2)), sub(a_vec(1,3), a_vec(3,3))
    ]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    
    # Y1 = B11 + B22
    Y1 = [
        add(b_vec(0,0), b_vec(2,2)), add(b_vec(0,1), b_vec(2,3)),
        add(b_vec(1,0), b_vec(3,2)), add(b_vec(1,1), b_vec(3,3))
    ]
    # Y2 = B11
    Y2 = [b_vec(0,0), b_vec(0,1), b_vec(1,0), b_vec(1,1)]
    # Y3 = B12 - B22
    Y3 = [
        sub(b_vec(0,2), b_vec(2,2)), sub(b_vec(0,3), b_vec(2,3)),
        sub(b_vec(1,2), b_vec(3,2)), sub(b_vec(1,3), b_vec(3,3))
    ]
    # Y4 = B21 - B11
    Y4 = [
        sub(b_vec(2,0), b_vec(0,0)), sub(b_vec(2,1), b_vec(0,1)),
        sub(b_vec(3,0), b_vec(1,0)), sub(b_vec(3,1), b_vec(1,1))
    ]
    # Y5 = B22
    Y5 = [b_vec(2,2), b_vec(2,3), b_vec(3,2), b_vec(3,3)]
    # Y6 = B11 + B12
    Y6 = [
        add(b_vec(0,0), b_vec(0,2)), add(b_vec(0,1), b_vec(0,3)),
        add(b_vec(1,0), b_vec(1,2)), add(b_vec(1,1), b_vec(1,3))
    ]
    # Y7 = B21 + B22
    Y7 = [
        add(b_vec(2,0), b_vec(2,2)), add(b_vec(2,1), b_vec(2,3)),
        add(b_vec(3,0), b_vec(3,2)), add(b_vec(3,1), b_vec(3,3))
    ]
    
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Now, for each l, s, compute U, V, W contributions.
    # U, V are 16x49. W is 16x49.
    # r = (l-1)*7 + (s-1)
    
    for l_idx in range(7):
        X_l = Xs[l_idx]
        Y_l = Ys[l_idx]
        
        for s_idx in range(7):
            r = l_idx * 7 + s_idx
            
            # Get u_x, u_y for this s
            u_x = strassen_x[s_idx] # 4 coeffs
            u_y = strassen_y[s_idx] # 4 coeffs
            
            # u_A = sum_{x_idx} u_x[x_idx] * X_l[x_idx]
            u_A = [0]*16
            for x_idx in range(4):
                if u_x[x_idx] != 0:
                    for k in range(16):
                        u_A[k] += u_x[x_idx] * X_l[x_idx][k]
                        
            u_B = [0]*16
            for y_idx in range(4):
                if u_y[y_idx] != 0:
                    for k in range(16):
                        u_B[k] += u_y[y_idx] * Y_l[y_idx][k]
                        
            # Fill U and V
            for idx in range(16):
                U[idx, r] = u_A[idx]
                V[idx, r] = u_B[idx]
                
            # Now W.
            # W depends on how M_l contributes to C.
            # M_l = Z_l.
            # Z_l entries are linear combos of S_l,1..S_l,7.
            # S_l,s is the scalar product.
            # So Z_l[ab] = sum_{s'} d_{ab,s'} S_l,s'
            # Here we are at specific s. So the coefficient of S_l,s in Z_l[ab] is d_{ab,s}.
            
            # But M_l contributes to C blocks with coefficients c_{block, l}.
            # C_{block} = sum_l c_{block, l} M_l.
            # So C_{i,k} = sum_{block, l} c_{block, l} M_l[local_i, local_k].
            # M_l[local_i, local_k] = sum_s d_{local_i, local_k, s} S_l,s.
            # So coefficient of S_l,s in C_{i,k} is sum_{block} c_{block, l} * d_{local_i, local_k, s}.
            
            # We need to map (i,k) to block and local indices.
            # i = 2*I + i', k = 2*K + k'.
            # Block is (I,K).
            # Local indices are (i', k').
            
            # Coefficients c_{block, l}
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # c is 4x7 matrix (blocks x l)
            c = [
                [1, 0, 0, 1, -1, 0, 1], # C11
                [0, 0, 1, 0, 1, 0, 0], # C12
                [0, 1, 0, 1, 0, 0, 0], # C21
                [1, -1, 1, 0, 0, 1, 0] # C22
            ]
            
            # d is 4x7 matrix (local entries x s)
            # d[0] is z00, d[1] is z01, d[2] is z10, d[3] is z11
            d = strassen_z
            
            # For each i, k in 0..3
            for i in range(4):
                for k in range(4):
                    I = i // 2
                    K = k // 2
                    i_prime = i % 2
                    k_prime = k % 2
                    
                    local_idx = i_prime * 2 + k_prime # 0:00, 1:01, 2:10, 3:11
                    
                    coeff = 0
                    for block_idx in range(4):
                        # Block (I,K) corresponds to block_idx?
                        # 0:11 (I=0,K=0), 1:12 (I=0,K=1), 2:21 (I=1,K=0), 3:22 (I=1,K=1)
                        # My c matrix rows: 0:C11, 1:C12, 2:C21, 3:C22.
                        # So block_idx = I*2 + K.
                        block_idx = I * 2 + K
                        if c[block_idx][l_idx] != 0:
                            coeff += c[block_idx][l_idx] * d[local_idx][s_idx]
                            
                    W[k * 4 + i, r] = coeff # Note: W index is k*n+i
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
