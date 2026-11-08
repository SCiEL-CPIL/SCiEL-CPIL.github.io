import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _generate_strassen_4x4():
    """
    Generates the 49 rank-1 tensors for 4x4 matrix multiplication using Strassen's algorithm.
    Returns a list of tuples (u, v, w) where u, v, w are lists of 16 integers.
    """
    # Strassen 2x2 coefficients
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y01 - y11)
    # S4 = x11(y10 - y00)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    
    # Reconstruction for C (2x2)
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Coefficients for inner Strassen (d[s, a, b])
    # s is 0..6, a,b is 0..1
    d_coeffs = [
        [1, 0, 0, 0], # S1 -> C00=1, C01=0, C10=0, C11=0? No, C11=1
        # Let's define d[s] as list of 4 coeffs for C00, C01, C10, C11
        [1, 0, 0, 1], # S1
        [0, 0, 1, -1], # S2
        [0, 1, 0, 1], # S3
        [1, 0, 1, 0], # S4
        [-1, 1, 0, 0], # S5
        [0, 0, 0, 1], # S6
        [1, 0, 0, 0]  # S7
    ]
    
    # Coefficients for outer Strassen (c[k, i, j])
    # k is 0..6, i,j is 0..1 (block indices)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Helper to add/subtract linear forms (lists of 16 ints)
    def add_lf(a, b):
        return [x + y for x, y in zip(a, b)]
    
    def sub_lf(a, b):
        return [x - y for x, y in zip(a, b)]
        
    def scale_lf(a, s):
        return [x * s for x in a]

    # Helper to get 2x2 block from 16-vector
    # Block (r, c) where r,c in 0..1
    # Returns 4 lists of 16 ints: [00, 01, 10, 11]
    def get_block(vec, r, c):
        # r=0,c=0: 0,1,4,5
        # r=0,c=1: 2,3,6,7
        # r=1,c=0: 8,9,12,13
        # r=1,c=1: 10,11,14,15
        # Indices:
        # 0: r*8 + c*2 + 0
        # 1: r*8 + c*2 + 1
        # 2: r*8 + c*2 + 4
        # 3: r*8 + c*2 + 5
        # Wait, row major 4x4.
        # Row r*2, col c*2: idx = (r*2)*4 + c*2
        # Row r*2, col c*2+1: idx = (r*2)*4 + c*2 + 1
        # Row r*2+1, col c*2: idx = (r*2+1)*4 + c*2
        # Row r*2+1, col c*2+1: idx = (r*2+1)*4 + c*2 + 1
        
        i00 = (r*2)*4 + c*2
        i01 = (r*2)*4 + c*2 + 1
        i10 = (r*2+1)*4 + c*2
        i11 = (r*2+1)*4 + c*2 + 1
        
        return [vec[i00], vec[i01], vec[i10], vec[i11]]

    # Initialize A and B as symbolic variables
    # A[i*4+j] = 1 if i*4+j == k else 0
    A_vars = []
    for i in range(16):
        v = [0]*16
        v[i] = 1
        A_vars.append(v)
        
    B_vars = []
    for i in range(16):
        v = [0]*16
        v[i] = 1
        B_vars.append(v)
        
    # We need to represent the blocks as lists of linear forms
    # A_blocks[r][c] is a list of 4 linear forms
    A_blocks = [[None for _ in range(2)] for _ in range(2)]
    B_blocks = [[None for _ in range(2)] for _ in range(2)]
    
    # This is inefficient, but 16x16 is small.
    # Actually, we can just work with the full 16-vectors and define the block operations on the fly.
    # But Strassen operates on 2x2 blocks.
    
    # Let's precompute the linear forms for the blocks.
    # A00 is sum of A_vars at indices 0,1,4,5? No, A00 IS the block.
    # The linear form for A00[0,0] is A_vars[0].
    
    # Let's define a function to get the 2x2 matrix of linear forms for a block
    def get_block_lf(vars, r, c):
        # Returns list of 4 linear forms (lists of 16)
        i00 = (r*2)*4 + c*2
        i01 = (r*2)*4 + c*2 + 1
        i10 = (r*2+1)*4 + c*2
        i11 = (r*2+1)*4 + c*2 + 1
        return [vars[i00], vars[i01], vars[i10], vars[i11]]
        
    A_blocks[0][0] = get_block_lf(A_vars, 0, 0)
    A_blocks[0][1] = get_block_lf(A_vars, 0, 1)
    A_blocks[1][0] = get_block_lf(A_vars, 1, 0)
    A_blocks[1][1] = get_block_lf(A_vars, 1, 1)
    
    B_blocks[0][0] = get_block_lf(B_vars, 0, 0)
    B_blocks[0][1] = get_block_lf(B_vars, 0, 1)
    B_blocks[1][0] = get_block_lf(B_vars, 1, 0)
    B_blocks[1][1] = get_block_lf(B_vars, 1, 1)
    
    def add_blocks(A, B):
        return [add_lf(a, b) for a, b in zip(A, B)]
        
    def sub_blocks(A, B):
        return [sub_lf(a, b) for a, b in zip(A, B)]
        
    # Outer Strassen
    # M1 = (A00 + A11)(B00 + B11)
    X1 = add_blocks(A_blocks[0][0], A_blocks[1][1])
    Y1 = add_blocks(B_blocks[0][0], B_blocks[1][1])
    
    # M2 = (A10 + A11)B00
    X2 = add_blocks(A_blocks[1][0], A_blocks[1][1])
    Y2 = B_blocks[0][0]
    
    # M3 = A00(B01 - B11)
    X3 = A_blocks[0][0]
    Y3 = sub_blocks(B_blocks[0][1], B_blocks[1][1])
    
    # M4 = A11(B10 - B00)
    X4 = A_blocks[1][1]
    Y4 = sub_blocks(B_blocks[1][0], B_blocks[0][0])
    
    # M5 = (A00 + A01)B11
    X5 = add_blocks(A_blocks[0][0], A_blocks[0][1])
    Y5 = B_blocks[1][1]
    
    # M6 = (A10 - A00)(B00 + B01)
    X6 = sub_blocks(A_blocks[1][0], A_blocks[0][0])
    Y6 = add_blocks(B_blocks[0][0], B_blocks[0][1])
    
    # M7 = (A01 - A11)(B10 + B11)
    X7 = sub_blocks(A_blocks[0][1], A_blocks[1][1])
    Y7 = add_blocks(B_blocks[1][0], B_blocks[1][1])
    
    Ms = [
        (X1, Y1),
        (X2, Y2),
        (X3, Y3),
        (X4, Y4),
        (X5, Y5),
        (X6, Y6),
        (X7, Y7)
    ]
    
    tensors = []
    
    for k in range(7):
        Xk, Yk = Ms[k]
        # Inner Strassen on Xk, Yk
        # S1 = (x00 + x11)(y00 + y11)
        # ...
        
        # We need to extract the 4 linear forms for Xk and Yk
        # Xk is a list of 4 linear forms: [x00, x01, x10, x11]
        
        # S1
        L_A_1 = add_lf(Xk[0], Xk[2]) # x00 + x11
        L_B_1 = add_lf(Yk[0], Yk[2]) # y00 + y11
        
        # S2
        L_A_2 = add_lf(Xk[2], Xk[3]) # x10 + x11
        L_B_2 = Yk[0] # y00
        
        # S3
        L_A_3 = Xk[0] # x00
        L_B_3 = sub_lf(Yk[1], Yk[3]) # y01 - y11
        
        # S4
        L_A_4 = Xk[3] # x11
        L_B_4 = sub_lf(Yk[2], Yk[0]) # y10 - y00
        
        # S5
        L_A_5 = add_lf(Xk[0], Xk[1]) # x00 + x01
        L_B_5 = Yk[3] # y11
        
        # S6
        L_A_6 = sub_lf(Xk[2], Xk[0]) # x10 - x00
        L_B_6 = add_lf(Yk[0], Yk[1]) # y00 + y01
        
        # S7
        L_A_7 = sub_lf(Xk[1], Xk[3]) # x01 - x11
        L_B_7 = add_lf(Yk[2], Yk[3]) # y10 + y11
        
        S_As = [L_A_1, L_A_2, L_A_3, L_A_4, L_A_5, L_A_6, L_A_7]
        S_Bs = [L_B_1, L_B_2, L_B_3, L_B_4, L_B_5, L_B_6, L_B_7]
        
        for s in range(7):
            u = S_As[s]
            v = S_Bs[s]
            w = [0]*16
            
            # Determine contribution to C
            # c_coeffs[k] is [C00, C01, C10, C11] coeffs for Mk
            # d_coeffs[s] is [C00, C01, C10, C11] coeffs for Ss within Mk
            
            # c_coeffs[k][0] -> block (0,0)
            # c_coeffs[k][1] -> block (0,1)
            # c_coeffs[k][2] -> block (1,0)
            # c_coeffs[k][3] -> block (1,1)
            
            for bi in range(4):
                c_val = c_coeffs[k][bi]
                if c_val == 0: continue
                
                # bi=0 -> i=0, j=0
                # bi=1 -> i=0, j=1
                # bi=2 -> i=1, j=0
                # bi=3 -> i=1, j=1
                i_block = bi // 2
                j_block = bi % 2
                
                # d_coeffs[s] is [C00, C01, C10, C11] within the 2x2 block
                # d[0] -> local (0,0)
                # d[1] -> local (0,1)
                # d[2] -> local (1,0)
                # d[3] -> local (1,1)
                
                for local_bi in range(4):
                    d_val = d_coeffs[s][local_bi]
                    if d_val == 0: continue
                    
                    a_local = local_bi // 2
                    b_local = local_bi % 2
                    
                    # Global row: i_block*2 + a_local
                    # Global col: j_block*2 + b_local
                    g_row = i_block * 2 + a_local
                    g_col = j_block * 2 + b_local
                    
                    idx = g_row * 4 + g_col
                    w[idx] += c_val * d_val
                    
            tensors.append((u, v, w))
            
    return tensors

def construct():
    start_time = time.time()
    deadline = 120 - 5 # 115s limit
    
    n, m, p = DIMENSIONS
    
    tensors_4x4 = _generate_strassen_4x4()
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First 4x4 block
    for u, v, w in tensors_4x4:
        for i in range(16):
            U[i][col] = u[i]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k][col] = v[j * 4 + k]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][col] = w[k * 4 + i]
        col += 1
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Second 4x4 block
    for u, v, w in tensors_4x4:
        for i in range(16):
            U[i][col] = u[i]
        for j in range(4):
            for k in range(4):
                V[j * 9 + (k + 4)][col] = v[j * 4 + k]
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i][col] = w[k * 4 + i]
        col += 1
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # 4x1 block
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
