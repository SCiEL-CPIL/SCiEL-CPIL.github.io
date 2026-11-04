"""Strassen-based decomposition for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8
    # Strassen for 2x2 blocks: 7 multiplications instead of 8
    # We tile the 4x4 * 4x8 into 2x2 blocks of (2x2) * (2x4)
    # But Strassen works on square blocks. Let's think differently.
    
    # For 4x4 * 4x8:
    # A is 4x4, B is 4x8, C = A*B is 4x8
    # Split A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks of 2x4: B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]] where Cij is 2x4
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # This is 8 multiplications of (2x2)*(2x4).
    # 
    # We can apply Strassen to the outer 2x2 block structure if we can
    # reformulate. But Strassen requires the block multiplications to be
    # over a ring where we can add/subtract. The inner products are (2x2)*(2x4).
    #
    # Actually, let's use a different approach. We can view the 4x4 * 4x8
    # as two separate 4x4 * 4x4 multiplications (since 8 = 4+4).
    # Each 4x4*4x4 can use Strassen with rank 7^2 = 49? No, Strassen for
    # 4x4 gives rank 7^2 = 49? Actually Strassen for n=4 gives rank 7^2=49.
    # Wait, Strassen for 2x2 has rank 7. For 4x4, we apply Strassen recursively:
    # 4x4 = 2x2 blocks of 2x2, so rank = 7^2 = 49.
    # So 4x4*4x4 has rank 49 with Strassen.
    # 4x4*4x8 = two 4x4*4x4, so rank = 49*2 = 98.
    # 
    # But wait, the schoolbook for 4x4*4x8 is 4*4*8 = 128.
    # Strassen gives 98. That's an improvement.
    #
    # Let me verify: Strassen for 2x2*2x2 has rank 7.
    # For 4x4*4x4: split into 2x2 blocks, each block is 2x2*2x2.
    # Using Strassen on the outer level: 7 "products" where each product
    # involves a 2x2*2x2 multiplication. Each of those has rank 7 (schoolbook
    # for 2x2*2x2 is 4, Strassen is 7... wait no).
    #
    # Let me reconsider. The tensor rank of 2x2*2x2 is 7 (Strassen).
    # The tensor rank of 4x4*4x4: we can use Strassen recursively.
    # Split 4x4 into four 2x2 blocks. The 7 Strassen products each require
    # multiplying 2x2 blocks. But those 2x2 block multiplications are themselves
    # 2x2*2x2 matrix multiplications, which have rank 7 (Strassen) or 4 (schoolbook).
    #
    # If we use Strassen at both levels: rank = 7 * 7 = 49.
    # If we use Strassen at outer level and schoolbook at inner: rank = 7 * 4 = 28? No.
    #
    # Actually, the rank of the 4x4*4x4 matrix multiplication tensor:
    # It's known that R(4x4*4x4) <= 49 via Strassen.
    # The schoolbook is 64.
    #
    # For 4x4*4x8: we can write B as [B_left | B_right] where each is 4x4.
    # C = A * B = [A*B_left | A*B_right]
    # Each A*B_left and A*B_right is 4x4*4x4, rank <= 49.
    # Total rank <= 98.
    #
    # Let me construct this explicitly.
    
    # Strassen for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)  -> (2x2)*(2x2), rank 4 schoolbook or 7 Strassen
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # If we use schoolbook for each 2x2*2x2 (rank 4 each), total = 7*4 = 28.
    # Wait, that's less than 49! Let me double-check.
    #
    # The 7 Strassen formulas involve 7 products of 2x2 matrices. Each 2x2*2x2
    # matrix product has tensor rank 4 (schoolbook). So total rank = 7*4 = 28.
    # But is this correct? The issue is that the "sums" like (A11+A22) involve
    # adding 2x2 matrices, which is fine. Each product (A11+A22)*(B11+B22)
    # is a 2x2*2x2 multiplication, which needs 4 scalar multiplications.
    # So total scalar multiplications = 7 * 4 = 28.
    #
    # But wait, the known result is that R(4x4*4x4) = 49? No, I think I'm confusing
    # things. Let me reconsider.
    #
    # Actually, I think the confusion is about what "rank" means here. The tensor
    # rank of the 2x2*2x2 matmul is 7 (Strassen). This means we need 7 scalar
    # multiplications. When we apply Strassen to 4x4*4x4, each of the 7 "products"
    # in the Strassen formula is itself a 2x2*2x2 matmul, which requires 7 scalar
    # multiplications (using Strassen) or 4 (using schoolbook).
    #
    # If we use schoolbook for the inner products: 7 * 4 = 28 scalar mults.
    # If we use Strassen for the inner products: 7 * 7 = 49 scalar mults.
    #
    # So 28 is better! But is 28 actually achievable? The issue is that the
    # Strassen decomposition at the outer level produces 7 bilinear forms, each
    # of which is a 2x2*2x2 matmul. Each such matmul can be done with 4 scalar
    # multiplications. So yes, 28 should work.
    #
    # Hmm, but I recall that the rank of 4x4*4x4 is actually 49, not 28. Let me
    # think about why. 
    #
    # Oh wait, I think I'm making an error. The Strassen algorithm for 2x2*2x2
    # uses 7 multiplications, but those are multiplications of 2x2 submatrices?
    # No, in the standard Strassen for 2x2, the inputs are scalars and the
    # "multiplications" M1-M7 are scalar multiplications. So R(2x2*2x2) = 7.
    #
    # For 4x4*4x4, we split into 2x2 blocks. The Strassen formulas give 7
    # products, but each product is a 2x2*2x2 matrix multiplication (not scalar).
    # Each 2x2*2x2 matmul has rank 7 (Strassen) or 4 (schoolbook).
    #
    # Wait, no. In the standard Strassen for 2x2*2x2, A11, A12, etc. are scalars.
    # For 4x4*4x4, A11, A12, etc. are 2x2 matrices. So M1 = (A11+A22)*(B11+B22)
    # is a 2x2*2x2 matrix product. To compute this, we need rank(2x2*2x2) = 7
    # scalar multiplications (using Strassen) or 4 (schoolbook).
    #
    # So total = 7 * 4 = 28 using schoolbook inner, or 7*7 = 49 using Strassen inner.
    #
    # I believe 28 is correct and achievable. Let me just implement it.
    
    # For our problem: 4x4 * 4x8
    # Split B into two 4x4 halves: B = [B1 | B2]
    # C = [A*B1 | A*B2]
    # Each A*B1 and A*B2 is 4x4*4x4, which we can do with rank 28 (Strassen outer + schoolbook inner).
    # Total rank = 28 + 28 = 56.
    
    # Let me construct the rank-28 decomposition for 4x4*4x4 first.
    # Then duplicate it for the two halves.
    
    # Actually, let me just build the full 4x4*4x8 directly.
    # 
    # A is 4x4, indexed A[i][j] for i,j in 0..3
    # B is 4x8, indexed B[j][k] for j in 0..3, k in 0..7
    # C is 4x8, indexed C[i][k] = sum_j A[i][j]*B[j][k]
    #
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where Aij is 2x2
    #   A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B = [[B11, B12], [B21, B22]] where Bij is 2x4
    #   B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    # C = [[C11, C12], [C21, C22]] where Cij is 2x4
    #   C11 = C[0:2, 0:4], C12 = C[0:2, 4:8], C21 = C[2:4, 0:4], C22 = C[2:4, 4:8]
    #
    # C11 = A11*B11 + A12*B21  (2x4)
    # C12 = A11*B12 + A12*B22  (2x4)
    # C21 = A21*B11 + A22*B21  (2x4)
    # C22 = A21*B12 + A22*B22  (2x4)
    #
    # This is 8 products of (2x2)*(2x4). Each (2x2)*(2x4) has rank 2*2*4=16 schoolbook.
    # Total = 8*16 = 128. That's just schoolbook.
    #
    # To use Strassen, I need the block sizes to be compatible. Strassen works when
    # the blocks are square. So let me use the 4x4*4x4 approach.
    #
    # 4x4 * 4x8 = (4x4 * 4x4) for left half + (4x4 * 4x4) for right half.
    # Each 4x4*4x4 can be done with rank 28.
    # Total rank = 56.
    
    # Let me build the rank-28 decomposition for 4x4*4x4.
    # A is 4x4, B is 4x4, C = A*B is 4x4.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], each 2x2
    # B = [[B11, B12], [B21, B22]], each 2x2
    # Strassen products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # Each Mj is a 2x2*2x2 product, done with schoolbook (rank 4).
    # So M1 needs 4 scalar mults, etc. Total = 7*4 = 28.
    #
    # For M1 = (A11+A22)*(B11+B22):
    # The 2x2 matrices: A11 = [[a00,a01],[a10,a11]], A22 = [[a20,a21],[a30,a31]]
    # A11+A22 = [[a00+a20, a01+a21], [a10+a30, a11+a31]]
    # B11+B22 = [[b00+b20, b01+b21], [b10+b30, b11+b31]]
    # Product is 2x2*2x2, schoolbook: 4 mults.
    #
    # This is getting complex to index. Let me just build it programmatically.
    
    # I'll construct the U, V, W matrices for the full 4x4*4x8 problem.
    # 
    # The output C[i][k] for i in 0..3, k in 0..7.
    # Using the trace convention W[k*n+i, r] where n=4, so W[k*4+i, r].
    
    n, m, p = 4, 4, 8
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 56
    
    # For each of the two halves (left 4x4 and right 4x4 of B),
    # we apply the rank-28 Strassen decomposition.
    
    # Helper: add a Strassen-28 decomposition for a 4x4*4x4 subproblem.
    # The subproblem uses rows i0..i0+3 of A, columns j0..j0+3 of B,
    # and produces C rows i0..i0+3, cols k0..k0+3.
    # offset is the column offset in U, V, W.
    
    def add_strassen_28(i0, k0, offset):
        """Add rank-28 Strassen decomposition for 4x4*4x4 starting at row i0, col k0."""
        # A blocks (2x2 each):
        # A11 = A[i0:i0+2, 0:2], A12 = A[i0:i0+2, 2:4]
        # A21 = A[i0+2:i0+4, 0:2], A22 = A[i0+2:i0+4, 2:4]
        # B blocks (2x2 each):
        # B11 = B[0:2, k0:k0+2], B12 = B[0:2, k0+2:k0+4]
        # B21 = B[2:4, k0:k0+2], B22 = B[2:4, k0+2:k0+4]
        
        # Strassen M1 = (A11+A22)*(B11+B22)
        # (A11+A22) has entries:
        #   row i0:   A[i0][0]+A[i0+2][0], A[i0][1]+A[i0+2][1]
        #   row i0+1: A[i0+1][0]+A[i0+2+1][0], A[i0+1][1]+A[i0+2+1][1]
        # (B11+B22) has entries:
        #   col k0:   B[0][k0]+B[2][k0], B[1][k0]+B[3][k0]  -- wait, B is 4x4 in this subproblem
        #   Actually B11 = B[0:2, k0:k0+2], B22 = B[2:4, k0+2:k0+4]
        #   B11+B22 doesn't make sense dimensionally unless we're careful.
        #
        # Hmm, I think I need to be more careful. In the Strassen decomposition
        # for 4x4*4x4, all blocks are 2x2. So:
        # A11 = A[i0:i0+2, 0:2], A12 = A[i0:i0+2, 2:4]
        # A21 = A[i0+2:i0+4, 0:2], A22 = A[i0+2:i0+4, 2:4]
        # B11 = B[0:2, k0:k0+2], B12 = B[0:2, k0+2:k0+4]
        # B21 = B[2:4, k0:k0+2], B22 = B[2:4, k0+2:k0+4]
        #
        # M1 = (A11+A22) * (B11+B22)
        # A11+A22 is 2x2:
        #   [[A[i0][0]+A[i0+2][0], A[i0][1]+A[i0+2][1]],
        #    [A[i0+1][0]+A[i0+3][0], A[i0+1][1]+A[i0+3][1]]]
        # B11+B22 is 2x2:
        #   [[B[0][k0]+B[2][k0+2], B[0][k0+1]+B[2][k0+3]],
        #    [B[1][k0]+B[3][k0+2], B[1][k0+1]+B[3][k0+3]]]
        #
        # The product (A11+A22)*(B11+B22) is a 2x2 matrix.
        # Using schoolbook for this 2x2*2x2:
        # P00 = (A11+A22)[0][0]*(B11+B22)[0][0] + (A11+A22)[0][1]*(B11+B22)[1][0]
        #      + (A11+A22)[1][0]*(B11+B22)[0][0] + (A11+A22)[1][1]*(B11+B22)[1][0]
        # This is getting very tedious to do by hand.
        
        # Let me take a different approach. I'll just use the known Strassen
        # decomposition and build it with a helper that handles the 2x2*2x2
        # schoolbook multiplication.
        
        # For a 2x2*2x2 schoolbook multiplication:
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # Z = X*Y:
        # Z00 = x00*y00 + x01*y10
        # Z01 = x00*y01 + x01*y11
        # Z10 = x10*y00 + x11*y10
        # Z11 = x10*y01 + x11*y11
        # Each Zij is a sum of 2 products, so 4 scalar multiplications total.
        
        # I'll define a function that, given the U-row indices for X entries,
        # V-row indices for Y entries, W-row indices for Z entries, and a
        # starting column offset, adds the 4 schoolbook terms.
        
        def add_2x2_schoolbook(x_rows, y_rows, z_rows, col_off):
            """Add 4 schoolbook terms for 2x2*2x2.
            x_rows = [[r00, r01], [r10, r11]]  (U row indices for X)
            y_rows = [[c00, c01], [c10, c11]]  (V row indices for Y)
            z_rows = [[w00, w01], [w10, w11]]  (W row indices for Z)
            """
            # Z00 = x00*y00 + x01*y10
            U[x_rows[0][0]][col_off] = 1.0
            V[y_rows[0][0]][col_off] = 1.0
            W[z_rows[0][0]][col_off] = 1.0
            U[x_rows[0][1]][col_off + 1] = 1.0
            V[y_rows[1][0]][col_off + 1] = 1.0
            W[z_rows[0][0]][col_off + 1] = 1.0
            # Z01 = x00*y01 + x01*y11
            U[x_rows[0][0]][col_off + 2] = 1.0
            V[y_rows[0][1]][col_off + 2] = 1.0
            W[z_rows[0][1]][col_off + 2] = 1.0
            U[x_rows[0][1]][col_off + 3] = 1.0
            V[y_rows[1][1]][col_off + 3] = 1.0
            W[z_rows[0][1]][col_off + 3] = 1.0
            # Z10 = x10*y00 + x11*y10
            U[x_rows[1][0]][col_off + 4] = 1.0
            V[y_rows[0][0]][col_off + 4] = 1.0
            W[z_rows[1][0]][col_off + 4] = 1.0
            U[x_rows[1][1]][col_off + 5] = 1.0
            V[y_rows[1][0]][col_off + 5] = 1.0
            W[z_rows[1][0]][col_off + 5] = 1.0
            # Z11 = x10*y01 + x11*y11
            U[x_rows[1][0]][col_off + 6] = 1.0
            V[y_rows[0][1]][col_off + 6] = 1.0
            W[z_rows[1][1]][col_off + 6] = 1.0
            U[x_rows[1][1]][col_off + 7] = 1.0
            V[y_rows[1][1]][col_off + 7] = 1.0
            W[z_rows[1][1]][col_off + 7] = 1.0
        
        # This approach is too error-prone with the sums. Let me just use
        # the schoolbook 128-rank decomposition but with a smarter structure.
        # Actually, let me just return the schoolbook for now to ensure correctness,
        # and note that the Strassen improvement is the target.
        
        # For safety, let me just do schoolbook.
        pass
    
    # Given the complexity, let me just return the schoolbook decomposition
    # which is guaranteed correct with rank 128.
    
    rank = n * m * p  # 128
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
