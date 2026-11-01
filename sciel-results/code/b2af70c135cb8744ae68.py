import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 4x4 is 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 blocks
    # We map the 4x4 matrix to 2x2 blocks of 2x2 matrices.
    # A[0:2, 0:2] is A11, A[0:2, 2:4] is A12, etc.
    
    # The 7 Strassen products for the outer 2x2 block structure
    # Each product is of two 2x2 matrices.
    # We need the rank-7 decomposition for 2x2 matrix multiplication.
    # Let's define the 2x2 Strassen decomposition terms.
    # For 2x2, indices i,j,k in 0..1.
    # U_2x2 is 4x7, V_2x2 is 4x7, W_2x2 is 4x7.
    
    # Standard Strassen 2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We can construct the 2x2 U, V, W matrices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for n=m=p=2.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: (A11+A22)(B11+B22)
    # A11 (i=0,j=0), A22 (i=1,j=1)
    # B11 (j=0,k=0), B22 (j=1,k=1)
    # U[0,0] += 1, U[1,1] += 1
    # V[0,0] += 1, V[1,1] += 1
    # W[0,0] += 1, W[1,1] += 1 ? No, W is for output C.
    # C11 gets M1, C22 gets M1.
    # So W[0,0] += 1, W[1,1] += 1.
    # But wait, the decomposition is a sum of rank 1 tensors.
    # The term is U_c \otimes V_c \otimes W_c.
    # For M1, we have one term? No, M1 is a product of sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 terms in schoolbook, but Strassen combines them into 1 "product" in the recursive step.
    # However, for the tensor decomposition, we need to express the whole thing.
    # The rank of 2x2 is 7.
    # The 7 terms correspond to the 7 M's.
    # Each M is a product of two linear combinations.
    # So each M corresponds to ONE rank-1 term in the 2x2 decomposition?
    # Yes, because M1 = U1 \cdot V1, where U1 = A11+A22, V1 = B11+B22.
    # So U1 is a vector in A-space, V1 in B-space.
    # And W1 is the contribution to C.
    
    # Let's define the 2x2 U, V, W for the 7 terms.
    # Indices for 2x2: i,j,k in {0,1}.
    # U index i*2+j, V index j*2+k, W index k*2+i.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U: A11 (0,0) and A22 (1,1). So U[0] = 1, U[3] = 1.
    # V: B11 (0,0) and B22 (1,1). So V[0] = 1, V[3] = 1.
    # W: C11 (0,0) and C22 (1,1). So W[0] = 1, W[3] = 1.
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: M2 = (A21+A22)B11
    # U: A21 (1,0) and A22 (1,1). U[2] = 1, U[3] = 1.
    # V: B11 (0,0). V[0] = 1.
    # W: C21 (0,1) and C22 (1,1). W[1] = 1, W[3] = 1.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = 1
    
    # Term 3: M3 = A11(B12-B22)
    # U: A11 (0,0). U[0] = 1.
    # V: B12 (0,1) and B22 (1,1). V[1] = 1, V[3] = -1.
    # W: C12 (1,0) and C22 (1,1)? No.
    # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
    # So M3 contributes to C12 (index 1,0 in W? W[k*n+i] -> k=0,i=1 -> W[1]) and C22 (k=1,i=1 -> W[3]).
    # Wait, W index k*2+i.
    # C12 is row 0, col 1. k=0, i=1. W[0*2+1] = W[1].
    # C22 is row 1, col 1. k=1, i=1. W[1*2+1] = W[3].
    # So W[1] = 1, W[3] = 1.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4: M4 = A22(B21-B11)
    # U: A22 (1,1). U[3] = 1.
    # V: B21 (1,0) and B11 (0,0). V[2] = 1, V[0] = -1.
    # W: C11 (0,0) and C21 (0,1).
    # C11: W[0]. C21: W[1].
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # Term 5: M5 = (A11+A12)B22
    # U: A11 (0,0) and A12 (0,1). U[0] = 1, U[1] = 1.
    # V: B22 (1,1). V[3] = 1.
    # W: C11 (0,0) and C12 (0,1).
    # C11: W[0]. C12: W[1].
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # U: A21 (1,0) and A11 (0,0). U[2] = 1, U[0] = -1.
    # V: B11 (0,0) and B12 (0,1). V[0] = 1, V[1] = 1.
    # W: C22 (1,1). W[3] = 1.
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # U: A12 (0,1) and A22 (1,1). U[1] = 1, U[3] = -1.
    # V: B21 (1,0) and B22 (1,1). V[2] = 1, V[3] = 1.
    # W: C11 (0,0). W[0] = 1.
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Now combine for 4x4.
    # The 4x4 matrix is composed of 2x2 blocks.
    # A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right).
    # We apply the 7 Strassen formulas to these blocks.
    # Each formula involves products of blocks.
    # Each block product is a 2x2 matrix multiplication, which we decompose using U2, V2, W2.
    # So we have 7 outer terms, each with 7 inner terms. Total 49.
    
    # Map block indices to 4x4 indices.
    # Block (bi, bj) where bi, bj in {0,1}.
    # A block (bi, bj) covers rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    # A index i*4+j.
    # For a term in U2 at index u_idx (0..3), corresponding to local (li, lj),
    # the global index is (2*bi + li)*4 + (2*bj + lj).
    
    # We define the 7 outer Strassen terms.
    # Each term specifies:
    # - A blocks involved (with signs)
    # - B blocks involved (with signs)
    # - C blocks involved (with signs)
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A: (0,0) sign 1, (1,1) sign 1
    # B: (0,0) sign 1, (1,1) sign 1
    # C: (0,0) sign 1, (1,1) sign 1
    
    # Term 2: M2 = (A21+A22)B11
    # A: (1,0) sign 1, (1,1) sign 1
    # B: (0,0) sign 1
    # C: (1,0) sign 1, (1,1) sign 1
    
    # Term 3: M3 = A11(B12-B22)
    # A: (0,0) sign 1
    # B: (0,1) sign 1, (1,1) sign -1
    # C: (0,1) sign 1, (1,1) sign 1
    
    # Term 4: M4 = A22(B21-B11)
    # A: (1,1) sign 1
    # B: (1,0) sign 1, (0,0) sign -1
    # C: (0,0) sign 1, (1,0) sign 1
    
    # Term 5: M5 = (A11+A12)B22
    # A: (0,0) sign 1, (0,1) sign 1
    # B: (1,1) sign 1
    # C: (0,0) sign 1, (0,1) sign 1
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # A: (1,0) sign 1, (0,0) sign -1
    # B: (0,0) sign 1, (0,1) sign 1
    # C: (1,1) sign 1
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # A: (0,1) sign 1, (1,1) sign -1
    # B: (1,0) sign 1, (1,1) sign 1
    # C: (0,0) sign 1

    outer_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_i, block_j, sign)
        [([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])], # M1
        [([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)])], # M2
        [([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])], # M3
        [([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])], # M4
        [([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)])], # M5
        [([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])], # M6
        [([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])], # M7
    ]

    col = 0
    for a_blocks, b_blocks, c_blocks in outer_terms:
        # For each combination of A block and B block in this term,
        # we add a rank-7 decomposition.
        # Actually, the term is a product of sums.
        # (sum A)(sum B) = sum_{a in A, b in B} A_a B_b.
        # So we iterate over all pairs (a, b).
        for a_block in a_blocks:
            bi_a, bj_a, sign_a = a_block
            for b_block in b_blocks:
                bi_b, bj_b, sign_b = b_block
                sign = sign_a * sign_b
                # Now we have a product of block (bi_a, bj_a) and (bi_b, bj_b).
                # This is a 2x2 matrix multiplication.
                # We use U2, V2, W2 for this.
                for c_inner in range(7):
                    # Add contribution to U, V, W
                    # U: from a_block and U2
                    # V: from b_block and V2
                    # W: from c_block and W2
                    # We need to map the local indices to global.
                    
                    # Iterate over non-zero entries in U2[:, c_inner]
                    for li in range(2):
                        for lj in range(2):
                            u_val = U2[li*2 + lj][c_inner]
                            if u_val != 0:
                                global_i = 2*bi_a + li
                                global_j = 2*bj_a + lj
                                U[global_i*4 + global_j][col] += sign * u_val
                                
                    for lj in range(2):
                        for lk in range(2):
                            v_val = V2[lj*2 + lk][c_inner]
                            if v_val != 0:
                                global_j = 2*bi_b + lj
                                global_k = 2*bj_b + lk
                                V[global_j*4 + global_k][col] += sign * v_val
                                
                    # W blocks
                    for c_block in c_blocks:
                        bi_c, bj_c, sign_c = c_block
                        sign_w = sign * sign_c
                        for lk in range(2):
                            for li in range(2):
                                w_val = W2[lk*2 + li][c_inner]
                                if w_val != 0:
                                    global_k = 2*bi_c + lk
                                    global_i = 2*bj_c + li
                                    W[global_k*4 + global_i][col] += sign_w * w_val
                                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
