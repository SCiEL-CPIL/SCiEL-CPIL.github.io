import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build a Strassen-based rank-7 decomposition for 2x2 x 2x2
    # and lift it to 4x4 x 4x8.

    # 4x4 * 4x8 can be seen as computing, for each of the 8 output columns,
    # a 4x4 matrix product. We can batch the 8 columns.

    # Strategy: Split the 4x4 matrices into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B (4x8) = [[B11, B12], [B21, B22]] where each block is 2x4

    # We want C = A * B, which is 4x8.
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]

    # Each 2x2 * 2x4 product can be computed using a Strassen-like approach
    # that handles rectangular blocks. But it's simpler to think of each
    # 2x2 * 2x4 as two independent 2x2 * 2x2 products (columns 0-1 and 2-3).

    # So the whole problem reduces to 8 independent 4x4 * 4x4 products,
    # but with the B side being 4x8 instead of 4x4.

    # Actually, let me just use a direct approach with Strassen for the
    # 4x4 part and schoolbook for the p=8 dimension.

    # 4x4 * 4x8: For each of the 8 columns of B, compute a 4x4 * 4x1 product.
    # A 4x4 * 4x1 product can be done in rank 16 (schoolbook) or less.

    # Let me use a simpler approach: exploit that 4x4*4x8 has the same
    # structure as two 4x4*4x4 products side by side.
    # The rank of 4x4*4x4 is at most 49 (known optimal), and we can do
    # both with the same A factor, so total rank <= 2*49 = 98.

    # But I don't have the explicit rank-49 factors memorized. Let me use
    # Strassen-based factors which I can construct explicitly.

    # Strassen for 2x2: rank 7
    # For 4x4: split into 2x2 blocks, apply Strassen at each level.
    # 4x4x4 via Strassen: rank 7^2 = 49.

    # Let me construct the 4x4x4 Strassen decomposition explicitly.

    # Strassen 2x2 factors (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For 4x4x4, I'll build the rank-49 factors from two levels of Strassen.

    # Level 1: Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x2
    # C = [[C00, C01], [C10, C11]] each 2x2

    # Apply Strassen at the 4x4 level:
    # M1 = (A00+A11)(B00+B11) -> 2x2*2x2 -> 7 mults
    # M2 = (A10+A11)B00 -> 2x2*2x2 -> 7 mults
    # M3 = A00(B01-B11) -> 2x2*2x2 -> 7 mults
    # M4 = A11(B10-B00) -> 2x2*2x2 -> 7 mults
    # M5 = (A00+A01)B11 -> 2x2*2x2 -> 7 mults
    # M6 = (A10-A00)(B00+B01) -> 2x2*2x2 -> 7 mults
    # M7 = (A01-A11)(B10+B11) -> 2x2*2x2 -> 7 mults

    # Total: 7 * 7 = 49 multiplications of scalars.

    # Now for 4x4 * 4x8: B is 4x8, split into two 4x4 halves.
    # B = [B_left, B_right] each 4x4.
    # C = A*B = [A*B_left, A*B_right]

    # We can compute both A*B_left and A*B_right using the same 49 scalar mults
    # but with different B factors. Actually, the A-side factors are shared,
    # but the B-side factors differ. So we need 49 + 49 = 98 scalar mults total.

    # Let me construct the explicit factors.

    # First, define the Strassen 2x2 factors as coefficient vectors.
    # For 2x2 * 2x2, the U, V, W factors (each 4x7) are:

    def strassen_2x2():
        # U[i*2+j, t] where i,j in 0..1, t in 0..6
        # V[i*2+j, t]
        # W[i*2+j, t] (output C[i,j] = sum_t U[i,j,t]*V[j',k',t]*W[k,i,t]... )
        # Actually the convention is: C[k*n+i, r] for W
        # For 2x2x2: n=m=p=2
        # U has shape (4,7), V has shape (4,7), W has shape (4,7)
        # U index: i*2+j (A element)
        # V index: j*2+k (B element)  
        # W index: k*2+i (C element, trace convention)

        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00+=1, C11+=1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1

        # M2 = (A10+A11)B00
        # U: A10=1, A11=1; V: B00=1; W: C10+=1, C11-=1
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] -= 1

        # M3 = A00(B01-B11)
        # U: A00=1; V: B01=1, B11=-1; W: C01+=1, C11+=1
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1

        # M4 = A11(B10-B00)
        # U: A11=1; V: B10=1, B00=-1; W: C00+=1, C10+=1
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[2][3] += 1

        # M5 = (A00+A01)B11
        # U: A00=1, A01=1; V: B11=1; W: C00-=1, C01+=1
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1

        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11+=1
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1

        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00+=1
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1

        return U, V, W

    # Now build 4x4x4 Strassen factors (rank 49)
    def build_4x4x4_strassen():
        sU, sV, sW = strassen_2x2()
        # 4x4x4: n=m=p=4
        # U shape (16, 49), V shape (16, 49), W shape (16, 49)
        # U index: i*4+j, V index: j*4+k, W index: k*4+i
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]

        # The 7 Strassen multiplications at the 4x4 level:
        # Each involves 2x2 blocks of A and B.
        # A block (bi, bj) where bi, bj in {0,1} covers A[2*bi:2*bi+2, 2*bj:2*bj+2]
        # B block (bk, bl) covers B[2*bk:2*bk+2, 2*bl:2*bl+2]
        # C block (ci, cj) covers C[2*ci:2*ci+2, 2*cj:2*cj+2]

        # For each of the 7 Strassen mults, we have a 2x2*2x2 product.
        # The 2x2*2x2 Strassen has 7 scalar mults.
        # So total 49.

        # Strassen mult t (0..6) involves:
        # A blocks: determined by t
        # B blocks: determined by t
        # C blocks: determined by t

        # Let me define the block-level Strassen:
        # M1: A blocks (0,0) and (1,1), B blocks (0,0) and (1,1), C blocks (0,0) and (1,1)
        # M2: A blocks (1,0) and (1,1), B block (0,0), C blocks (1,0) and (1,1)
        # M3: A block (0,0), B blocks (0,1) and (1,1), C blocks (0,1) and (1,1)
        # M4: A block (1,1), B blocks (1,0) and (0,0), C blocks (0,0) and (1,0)
        # M5: A blocks (0,0) and (0,1), B block (1,1), C blocks (0,0) and (0,1)
        # M6: A blocks (1,0) and (0,0), B blocks (0,0) and (0,1), C block (1,1)
        # M7: A blocks (0,1) and (1,1), B blocks (1,0) and (1,1), C block (0,0)

        # For each Strassen mult t, the 2x2*2x2 sub-product uses sU, sV, sW.
        # The scalar mult index is t*7 + s where s is 0..6.

        # Define block contributions for each Strassen mult:
        # (a_blocks, b_blocks, c_blocks, signs)
        # a_blocks: list of (bi, bj, sign)
        # b_blocks: list of (bk, bl, sign)  
        # c_blocks: list of (ci, cj, sign)

        block_defs = [
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]

        for t, (a_blocks, b_blocks, c_blocks) in enumerate(block_defs):
            for s in range(7):
                col = t * 7 + s
                for (bi, bj, sa) in a_blocks:
                    for (si, sj) in [(0,0),(0,1),(1,0),(1,1)]:
                        a_idx = (2*bi+si)*4 + (2*bj+sj)
                        U4[a_idx][col] += sa * sU[si*2+sj][s]

                for (bk, bl, sb) in b_blocks:
                    for (si, sj) in [(0,0),(0,1),(1,0),(1,1)]:
                        b_idx = (2*bk+si)*4 + (2*bl+sj)
                        V4[b_idx][col] += sb * sV[si*2+sj][s]

                for (ci, cj, sc) in c_blocks:
                    for (si, sj) in [(0,0),(0,1),(1,0),(1,1)]:
                        # C index in 4x4x4: k*4+i where C[k,i]
                        # C block (ci,cj) means C rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
                        # So k = 2*cj+sj, i = 2*ci+si
                        c_idx = (2*cj+sj)*4 + (2*ci+si)
                        W4[c_idx][col] += sc * sW[si*2+sj][s]

        return U4, V4, W4

    # Now build 4x4 * 4x8 using two 4x4*4x4 Strassen decompositions
    U4, V4, W4 = build_4x4x4_strassen()

    # For 4x4 * 4x8:
    # B is 4x8, split into B_left (4x4) and B_right (4x4)
    # C = A*B = [A*B_left, A*B_right] (each 4x4)
    
    # U side: same for both (A is 4x4)
    # V side: V_left uses V4 with B_left indices, V_right uses V4 with B_right indices
    # W side: W_left uses W4 with C_left indices, W_right uses W4 with C_right indices

    # Shapes: U (16, 98), V (32, 98), W (32, 98)
    # U index: i*4+j for i,j in 0..3
    # V index: j*8+k for j in 0..3, k in 0..7
    # W index: k*4+i for k in 0..7, i in 0..3

    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]

    # First 49: A * B_left
    for r in range(16):
        for col in range(49):
            U[r][col] = U4[r][col]
    for r in range(16):
        for col in range(49):
            # V index: j*4+k for B_left (j in 0..3, k in 0..3)
            # In the 4x8 B, B_left occupies columns 0..3
            # V4 index: j*4+k
            V[r][col] = V4[r][col]
    for r in range(16):
        for col in range(49):
            # W index: k*4+i for C_left (k in 0..3, i in 0..3)
            # In the 4x8 C, C_left occupies columns 0..3
            W[r][col] = W4[r][col]

    # Last 49: A * B_right
    for r in range(16):
        for col in range(49):
            U[r][col + 49] = U4[r][col]
    for r in range(16):
        for col in range(49):
            # V4 index: j*4+k corresponds to B_right element B[j, 4+k]
            # V index in 4x8: j*8 + (4+k)
            j, k = divmod(r, 4)
            V[j*8 + 4 + k][col + 49] = V4[r][col]
    for r in range(16):
        for col in range(49):
            # W4 index: k*4+i corresponds to C_right element C[k, 4+i]
            # W index in 4x8 C: (4+k)*4 + i
            k, i = divmod(r, 4)
            W[(4+k)*4 + i][col + 49] = W4[r][col]

    return {"U": U, "V": V, "W": W}
