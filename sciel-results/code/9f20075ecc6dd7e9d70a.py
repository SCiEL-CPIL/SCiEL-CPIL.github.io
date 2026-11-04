import time
import numpy as np

def construct():
    """
    Returns a low-rank tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the first two block columns (rank 84) and schoolbook for the third (rank 48).
    Total rank: 132.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    
    # Block definitions
    # A: 4x4
    # A11: [0,1,4,5]
    # A12: [2,3,6,7]
    # A21: [8,9,12,13]
    # A22: [10,11,14,15]
    
    # B: 4x9
    # B11: [0,1,2, 9,10,11]
    # B12: [3,4,5, 12,13,14]
    # B13: [6,7,8, 15,16,17]
    # B21: [18,19,20, 27,28,29]
    # B22: [21,22,23, 30,31,32]
    # B23: [24,25,26, 33,34,35]
    
    # C: 4x9 (W convention: k*n+i)
    # C11: [0,1,2, 4,5,6]
    # C12: [12,16,20, 13,17,21]
    # C13: [24,28,32, 25,29,33]
    # C21: [2,6,10, 3,7,11]
    # C22: [14,18,22, 15,19,23]
    # C23: [26,30,34, 27,31,35]

    rank = 132
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    col = 0
    
    # Helper to add a block multiplication
    # a_blocks: list of (block_id, sign) for A
    # b_blocks: list of (block_id, sign) for B
    # c_blocks: list of (block_id, sign) for C
    # The block multiplication is (sum A) * (sum B) = sum C
    # We add 12 rank-1 terms for each (sum A) * (sum B) contribution?
    # No, (sum A) * (sum B) is a single block multiplication of 2x2 * 2x3.
    # This has rank 12.
    # We need to expand the linear combinations.
    
    def add_block_mult(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        # a_coeffs: dict {global_idx: coeff}
        # b_coeffs: dict {global_idx: coeff}
        # c_coeffs: dict {global_idx: coeff}
        
        # The tensor term is sum_{a,b,c} U[a,t] V[b,t] W[c,t]
        # For a block mult X * Y = Z, where X is 2x2, Y is 2x3, Z is 2x3.
        # X[i,j] = sum_a a_coeffs[a] A[a]
        # Y[j,k] = sum_b b_coeffs[b] B[b]
        # Z[i,k] = sum_j X[i,j] Y[j,k]
        
        # We add 12 terms for the schoolbook of 2x2 * 2x3.
        # For each i in 0..1, j in 0..1, k in 0..2:
        # Term: X[i,j] * Y[j,k] * Z[i,k]
        # X[i,j] is a linear combo of A.
        # Y[j,k] is a linear combo of B.
        # Z[i,k] is a linear combo of C.
        
        # Wait, Z[i,k] is the result. In the decomposition, W[c,t] corresponds to C[c].
        # The term is (sum_a u_a A[a]) (sum_b v_b B[b]) w_c.
        # This is a sum of rank-1 terms.
        # For each (i,j,k):
        #   For each a in a_coeffs:
        #     For each b in b_coeffs:
        #       For each c in c_coeffs:
        #         This is not right.
        
        # The block multiplication X*Y=Z is a tensor of rank 12.
        # The entries of X, Y, Z are linear combinations of global A, B, C.
        # We can treat the local 2x2, 2x3, 2x3 tensors.
        # Local U: 4x12, Local V: 6x12, Local W: 6x12.
        # Global U: 16xR.
        
        # For each local term t (0..11):
        #   local_u[i,j,t], local_v[j,k,t], local_w[i,k,t]
        #   Global U[a, col] += local_u[i,j,t] * a_coeffs[a]
        #   Global V[b, col] += local_v[j,k,t] * b_coeffs[b]
        #   Global W[c, col] += local_w[i,k,t] * c_coeffs[c]
        
        # Schoolbook for 2x2 * 2x3:
        # t=0: i=0,j=0,k=0
        # t=1: i=0,j=0,k=1
        # t=2: i=0,j=0,k=2
        # t=3: i=0,j=1,k=0
        # t=4: i=0,j=1,k=1
        # t=5: i=0,j=1,k=2
        # t=6: i=1,j=0,k=0
        # t=7: i=1,j=0,k=1
        # t=8: i=1,j=0,k=2
        # t=9: i=1,j=1,k=0
        # t=10: i=1,j=1,k=1
        # t=11: i=1,j=1,k=2
        
        for t in range(12):
            i = t // 6
            rem = t % 6
            j = rem // 3
            k = rem % 3
            
            # local_u[i,j] = 1
            # local_v[j,k] = 1
            # local_w[i,k] = 1
            
            for a_idx, a_val in a_coeffs.items():
                if a_val == 0: continue
                U[a_idx, col] += a_val
            for b_idx, b_val in b_coeffs.items():
                if b_val == 0: continue
                V[b_idx, col] += b_val
            for c_idx, c_val in c_coeffs.items():
                if c_val == 0: continue
                W[c_idx, col] += c_val
            
            col += 1

    # Define block indices
    A_blocks = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    B_blocks = {
        0: [0, 1, 2, 9, 10, 11],
        1: [3, 4, 5, 12, 13, 14],
        2: [6, 7, 8, 15, 16, 17],
        3: [18, 19, 20, 27, 28, 29],
        4: [21, 22, 23, 30, 31, 32],
        5: [24, 25, 26, 33, 34, 35]
    }
    C_blocks = {
        0: [0, 1, 2, 4, 5, 6],
        1: [12, 16, 20, 13, 17, 21],
        2: [24, 28, 32, 25, 29, 33],
        3: [2, 6, 10, 3, 7, 11],
        4: [14, 18, 22, 15, 19, 23],
        5: [26, 30, 34, 27, 31, 35]
    }

    def get_a_coeffs(blocks):
        coeffs = {}
        for bid, sign in blocks:
            for idx in A_blocks[bid]:
                coeffs[idx] = coeffs.get(idx, 0) + sign
        return coeffs

    def get_b_coeffs(blocks):
        coeffs = {}
        for bid, sign in blocks:
            for idx in B_blocks[bid]:
                coeffs[idx] = coeffs.get(idx, 0) + sign
        return coeffs

    def get_c_coeffs(blocks):
        coeffs = {}
        for bid, sign in blocks:
            for idx in C_blocks[bid]:
                coeffs[idx] = coeffs.get(idx, 0) + sign
        return coeffs

    # Strassen M's
    # M1 = (A11+A22)(B11+B21)
    a1 = get_a_coeffs([(0, 1), (3, 1)])
    b1 = get_b_coeffs([(0, 1), (3, 1)])
    c1 = get_c_coeffs([(0, 1), (3, 1)]) # C11 + C21
    add_block_mult(a1, b1, c1)

    # M2 = (A21+A22)B11
    a2 = get_a_coeffs([(2, 1), (3, 1)])
    b2 = get_b_coeffs([(0, 1)])
    c2 = get_c_coeffs([(3, 1)]) # C21
    add_block_mult(a2, b2, c2)

    # M3 = A11(B12-B22)
    a3 = get_a_coeffs([(0, 1)])
    b3 = get_b_coeffs([(1, 1), (4, -1)])
    c3 = get_c_coeffs([(1, 1)]) # C12
    add_block_mult(a3, b3, c3)

    # M4 = A22(B21-B11)
    a4 = get_a_coeffs([(3, 1)])
    b4 = get_b_coeffs([(3, 1), (0, -1)])
    c4 = get_c_coeffs([(0, 1)]) # C11
    add_block_mult(a4, b4, c4)

    # M5 = (A11+A12)B22
    a5 = get_a_coeffs([(0, 1), (1, 1)])
    b5 = get_b_coeffs([(4, 1)])
    c5 = get_c_coeffs([(1, 1)]) # C12
    add_block_mult(a5, b5, c5)

    # M6 = (A21-A11)(B11+B12)
    a6 = get_a_coeffs([(2, 1), (0, -1)])
    b6 = get_b_coeffs([(0, 1), (1, 1)])
    c6 = get_c_coeffs([(3, 1)]) # C21
    add_block_mult(a6, b6, c6)

    # M7 = (A12-A22)(B21+B22)
    a7 = get_a_coeffs([(1, 1), (3, -1)])
    b7 = get_b_coeffs([(3, 1), (4, 1)])
    c7 = get_c_coeffs([(0, 1)]) # C11
    add_block_mult(a7, b7, c7)

    # Schoolbook for C13, C23
    # C13 = A11 B13 + A12 B23
    a_c13_1 = get_a_coeffs([(0, 1)])
    b_c13_1 = get_b_coeffs([(2, 1)])
    c_c13_1 = get_c_coeffs([(2, 1)])
    add_block_mult(a_c13_1, b_c13_1, c_c13_1)

    a_c13_2 = get_a_coeffs([(1, 1)])
    b_c13_2 = get_b_coeffs([(5, 1)])
    c_c13_2 = get_c_coeffs([(2, 1)])
    add_block_mult(a_c13_2, b_c13_2, c_c13_2)

    # C23 = A21 B13 + A22 B23
    a_c23_1 = get_a_coeffs([(2, 1)])
    b_c23_1 = get_b_coeffs([(2, 1)])
    c_c23_1 = get_c_coeffs([(5, 1)])
    add_block_mult(a_c23_1, b_c23_1, c_c23_1)

    a_c23_2 = get_a_coeffs([(3, 1)])
    b_c23_2 = get_b_coeffs([(5, 1)])
    c_c23_2 = get_c_coeffs([(5, 1)])
    add_block_mult(a_c23_2, b_c23_2, c_c23_2)

    return {"U": U, "V": V, "W": W}
