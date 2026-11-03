import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen's algorithm for 2x2 matrix multiplication.
    # A is n x m = 2 x 4. B is m x p = 4 x 7.
    # We can view A as two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3).
    # We can view B as two 2x2 blocks vertically? No, B is 4x7.
    # B rows 0,1 and 2,3.
    # Let B_top = B[0:2, :] and B_bot = B[2:4, :].
    # C = A_left @ B_top + A_right @ B_bot.
    # C is 2 x 7.
    # We can split the 7 columns into 3 blocks of 2, 2, 1?
    # Or just treat the 7 columns as 3.5 blocks?
    # Strassen works for 2x2.
    # Let's decompose the 4x7 part differently.
    # Actually, we can just apply Strassen to the whole 2x4 @ 4x7 by padding?
    # No, rank minimization usually benefits from splitting into powers of 2 if possible,
    # but here dimensions are not powers of 2.
    # Let's try to decompose 2x4 @ 4x7 directly using Strassen on 2x2 sub-blocks.
    
    # A = [A11 A12; A21 A22] where A11, A12, A21, A22 are 1x2? No, A is 2x4.
    # A = [A11 A12] where A11 is 2x2, A12 is 2x2.
    # B is 4x7.
    # B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    # C = A11 B1 + A12 B2.
    # Now we have two products of 2x2 and 2x7.
    # We can further decompose the 2x7 part.
    # Let B1 = [B11 B12 B13] where B11, B12 are 2x2, B13 is 2x1.
    # Let B2 = [B21 B22 B23] where B21, B22 are 2x2, B23 is 2x1.
    # C = A11 [B11 B12 B13] + A12 [B21 B22 B23]
    # C = [A11 B11 + A12 B21, A11 B12 + A12 B22, A11 B13 + A12 B23]
    # Each term is a sum of two 2x2 products (or 2x1).
    # Let's use Strassen for 2x2 @ 2x2.
    # For 2x2 @ 2x1, we can just do 4 multiplications (or 3 with Strassen variant?).
    # Standard Strassen is for 2x2 @ 2x2.
    # Let's handle the 2x2 @ 2x2 blocks first.
    # There are 4 such blocks:
    # 1. A11 B11
    # 2. A11 B12
    # 3. A12 B21
    # 4. A12 B22
    # And 2 blocks of 2x2 @ 2x1:
    # 5. A11 B13
    # 6. A12 B23
    
    # For each 2x2 @ 2x2, Strassen uses 7 multiplications.
    # Total 7 * 4 = 28 multiplications.
    # For each 2x2 @ 2x1, we can use 3 multiplications?
    # C = A b where A is 2x2, b is 2x1.
    # c1 = a11 b1 + a12 b2
    # c2 = a21 b1 + a22 b2
    # We can compute:
    # m1 = (a11+a21) b1
    # m2 = a21 (b2-b1)
    # m3 = a12 (b1+b2)
    # c1 = m1 + m2 - m3/2? No.
    # Let's stick to simple 4 multiplications for 2x2 @ 2x1 to be safe, or find a 3-mult scheme.
    # Actually, 2x2 @ 2x1 is equivalent to 2x2 @ 2x2 with the second column zero.
    # Rank of 2x2 @ 2x1 is 3?
    # Let's check.
    # If we use 4 mults: rank 4.
    # Total rank = 28 + 4 + 4 = 36.
    # Current rank is 56. 36 is better.
    
    # Let's try to be smarter.
    # Can we combine the blocks?
    # C = A11 B1 + A12 B2.
    # This is a 2x4 @ 4x7 product.
    # We can view this as (A11 A12) @ [B1; B2].
    # This is exactly the original problem.
    # Strassen on 2x4 @ 4x7?
    # Pad to 2x8 @ 8x8? No, that increases rank.
    # Pad to 4x4 @ 4x8?
    # Let's stick to the block decomposition.
    
    # We need to construct U, V, W.
    # The tensor is T(i, j, k) = delta_{i, j+k}? No.
    # T(A, B, C) such that C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Indices:
    # i in 0..n-1 (0..1)
    # j in 0..m-1 (0..3)
    # k in 0..p-1 (0..6)
    
    # U has shape (n*m, rank) = (8, R)
    # V has shape (m*p, rank) = (28, R)
    # W has shape (p*n, rank) = (14, R)
    # Wait, W shape in task is (p*n, rank).
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is (k*n + i).
    # k is column of C, i is row of C.
    
    # Let's build the Strassen blocks.
    # We will generate the coefficients for each multiplication.
    # Each multiplication corresponds to a rank-1 term in the tensor decomposition.
    # Term r: U[:, r], V[:, r], W[:, r].
    # The product is sum_r U[:, r] V[:, r] W[:, r] (outer products summed).
    # Actually, the tensor T is decomposed as sum_r u_r v_r w_r.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    
    # Let's define the Strassen formulas for 2x2 @ 2x2.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p, q], [r, s]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g+h)
    # m5 = (a-c)(g+h)
    # m6 = (b+c)(e-g)
    # m7 = (b-d)(f+h)
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    
    # We need to map these to the global indices.
    # Block 1: A11 B11
    # A11 = A[0:2, 0:2] -> indices (0,0), (0,1), (1,0), (1,1) in A.
    # B11 = B[0:2, 0:2] -> indices (0,0), (0,1), (1,0), (1,1) in B.
    # Result C11 = C[0:2, 0:2] -> indices (0,0), (0,1), (1,0), (1,1) in C.
    
    # Let's write a helper to add a Strassen block.
    # inputs:
    # a_idx: list of 4 indices (i,j) for A block.
    # b_idx: list of 4 indices (j,k) for B block.
    # c_idx: list of 4 indices (i,k) for C block.
    # offset: current column in U, V, W.
    
    def add_strassen_block(a_idx, b_idx, c_idx, offset):
        # a_idx: [(i1,j1), (i1,j2), (i2,j1), (i2,j2)] corresponding to a,b,c,d
        # b_idx: [(j1,k1), (j1,k2), (j2,k1), (j2,k2)] corresponding to e,f,g,h
        # c_idx: [(i1,k1), (i1,k2), (i2,k1), (i2,k2)] corresponding to p,q,r,s
        
        # We will create 7 terms.
        # Each term is a vector in U, V, W.
        # We return the new offset.
        
        # m1 = (a+d)(e+h)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: p=1, s=1
        # Wait, the term is u * v * w.
        # So for m1, U has 1s at a_idx[0] and a_idx[3].
        # V has 1s at b_idx[0] and b_idx[3].
        # W has 1s at c_idx[0] (p) and c_idx[3] (s).
        # But wait, p = m1 + ... and s = m1 + ...
        # So W for m1 contributes to p and s.
        
        # Let's define the coefficients for W for each m.
        # m1: p=1, s=1
        # m2: r=1, s=-1
        # m3: q=1, s=1
        # m4: p=1, r=1
        # m5: p=-1, q=1
        # m6: s=1
        # m7: p=1, q=1
        
        # Wait, let's re-verify the Strassen formulas.
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        
        # Coefficients for W (c_idx 0,1,2,3 -> p,q,r,s):
        # m1: [1, 0, 0, 1]
        # m2: [0, 0, 1, -1]
        # m3: [0, 1, 0, 1]
        # m4: [1, 0, 1, 0]
        # m5: [-1, 1, 0, 0]
        # m6: [0, 0, 0, 1]
        # m7: [1, 1, 0, 0]
        
        # Coefficients for U (a_idx 0,1,2,3 -> a,b,c,d):
        # m1: [1, 0, 0, 1] (a+d)
        # m2: [0, 0, 1, 1] (c+d)
        # m3: [1, 0, 0, 0] (a)
        # m4: [0, 0, 0, 1] (d)
        # m5: [1, 0, -1, 0] (a-c)
        # m6: [0, 1, 1, 0] (b+c)
        # m7: [0, 1, 0, -1] (b-d)
        
        # Coefficients for V (b_idx 0,1,2,3 -> e,f,g,h):
        # m1: [1, 0, 0, 1] (e+h)
        # m2: [1, 0, 0, 0] (e)
        # m3: [0, 1, 0, -1] (f-h)
        # m4: [0, 0, 1, 1] (g+h)
        # m5: [0, 0, 1, 1] (g+h)
        # m6: [1, 0, -1, 0] (e-g)
        # m7: [0, 1, 0, 1] (f+h)
        
        u_coefs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, -1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, -1]
        ]
        v_coefs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, -1, 0],
            [0, 1, 0, 1]
        ]
        w_coefs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0]
        ]
        
        for t in range(7):
            for idx, val in enumerate(u_coefs[t]):
                if val != 0:
                    U[a_idx[idx]][offset + t] += val
            for idx, val in enumerate(v_coefs[t]):
                if val != 0:
                    V[b_idx[idx]][offset + t] += val
            for idx, val in enumerate(w_coefs[t]):
                if val != 0:
                    W[c_idx[idx]][offset + t] += val
        
        return offset + 7

    # Initialize U, V, W
    # n=2, m=4, p=7
    # U: (8, R)
    # V: (28, R)
    # W: (14, R)
    
    # We don't know R yet. Let's estimate.
    # 4 blocks of 2x2@2x2 -> 28
    # 2 blocks of 2x2@2x1 -> 8 (if 4 mults each)
    # Total 36.
    # Let's use 36.
    
    R = 36
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    offset = 0
    
    # Block 1: A11 B11
    # A11: rows 0,1; cols 0,1.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    a11_idx = [0, 1, 4, 5]
    # B11: rows 0,1; cols 0,1.
    # B indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 7, 8
    b11_idx = [0, 1, 7, 8]
    # C11: rows 0,1; cols 0,1.
    # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    c11_idx = [0, 1, 2, 3]
    offset = add_strassen_block(a11_idx, b11_idx, c11_idx, offset)
    
    # Block 2: A11 B12
    # A11: same.
    # B12: rows 0,1; cols 2,3.
    # B indices: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 9, 10
    b12_idx = [2, 3, 9, 10]
    # C12: rows 0,1; cols 2,3.
    # C indices: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 4, 5
    c12_idx = [2, 3, 4, 5]
    offset = add_strassen_block(a11_idx, b12_idx, c12_idx, offset)
    
    # Block 3: A12 B21
    # A12: rows 0,1; cols 2,3.
    # A indices: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    a12_idx = [2, 3, 6, 7]
    # B21: rows 2,3; cols 0,1.
    # B indices: (2,0), (2,1), (3,0), (3,1) -> 14, 15, 21, 22
    b21_idx = [14, 15, 21, 22]
    # C21: rows 0,1; cols 0,1.
    # C indices: same as C11 -> 0, 1, 2, 3
    c21_idx = [0, 1, 2, 3]
    offset = add_strassen_block(a12_idx, b21_idx, c21_idx, offset)
    
    # Block 4: A12 B22
    # A12: same.
    # B22: rows 2,3; cols 2,3.
    # B indices: (2,2), (2,3), (3,2), (3,3) -> 16, 17, 23, 24
    b22_idx = [16, 17, 23, 24]
    # C22: rows 0,1; cols 2,3.
    # C indices: same as C12 -> 2, 3, 4, 5
    c22_idx = [2, 3, 4, 5]
    offset = add_strassen_block(a12_idx, b22_idx, c22_idx, offset)
    
    # Now the 2x2 @ 2x1 blocks.
    # Block 5: A11 B13
    # A11: 2x2.
    # B13: rows 0,1; col 6. (B is 4x7, so col 6 is the last one).
    # B indices: (0,6), (1,6) -> 6, 13.
    # We need to compute A11 @ B13.
    # Let's use 4 multiplications for simplicity.
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # Terms:
    # 1. a00 b0 -> c0
    # 2. a01 b1 -> c0
    # 3. a10 b0 -> c1
    # 4. a11 b1 -> c1
    
    # U: a00, a01, a10, a11
    # V: b0, b1
    # W: c0, c1
    
    # C13: rows 0,1; col 6.
    # C indices: (0,6), (1,6) -> 6, 7.
    
    # Term 1: a00 b0 -> c0
    U[0][offset] = 1
    V[6][offset] = 1
    W[6][offset] = 1
    offset += 1
    
    # Term 2: a01 b1 -> c0
    U[1][offset] = 1
    V[13][offset] = 1
    W[6][offset] = 1
    offset += 1
    
    # Term 3: a10 b0 -> c1
    U[4][offset] = 1
    V[6][offset] = 1
    W[7][offset] = 1
    offset += 1
    
    # Term 4: a11 b1 -> c1
    U[5][offset] = 1
    V[13][offset] = 1
    W[7][offset] = 1
    offset += 1
    
    # Block 6: A12 B23
    # A12: 2x2.
    # B23: rows 2,3; col 6.
    # B indices: (2,6), (3,6) -> 20, 27.
    # C23: rows 0,1; col 6.
    # C indices: (0,6), (1,6) -> 6, 7.
    
    # Term 1: a00 b0 -> c0
    U[2][offset] = 1
    V[20][offset] = 1
    W[6][offset] = 1
    offset += 1
    
    # Term 2: a01 b1 -> c0
    U[3][offset] = 1
    V[27][offset] = 1
    W[6][offset] = 1
    offset += 1
    
    # Term 3: a10 b0 -> c1
    U[6][offset] = 1
    V[20][offset] = 1
    W[7][offset] = 1
    offset += 1
    
    # Term 4: a11 b1 -> c1
    U[7][offset] = 1
    V[27][offset] = 1
    W[7][offset] = 1
    offset += 1
    
    # Total offset should be 28 + 4 + 4 = 36.
    # R = 36.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
