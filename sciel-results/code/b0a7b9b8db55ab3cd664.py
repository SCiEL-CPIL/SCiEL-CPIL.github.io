import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based block decomposition for 2x3 matrix * 3x5 matrix.
    # We split the inner dimension m=3 into m1=1 and m2=2.
    # We split the output dimension p=5 into p1=2 and p2=3.
    #
    # Block decomposition:
    # C = A * B
    # A is 2x3, B is 3x5, C is 2x5.
    #
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x5, B2 is 2x5
    #
    # C = A1*B1 + A2*B2
    #
    # We further split B1 into B11 (1x2) and B12 (1x3)
    # and B2 into B21 (2x2) and B22 (2x3)
    #
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # Sub-problems:
    # 1. A1*B11: 2x1 * 1x2 -> 2x2. Rank 2 (trivial, since one dim is 1)
    #    Actually, for 2x1 * 1x2, we need 2 multiplications.
    #    C11 = A10*B10, C12 = A10*B11
    #    C21 = A20*B10, C22 = A20*B11
    #    Wait, A1 is 2x1: [a1; a2], B11 is 1x2: [b1, b2]
    #    C11 = a1*b1, C12 = a1*b2, C21 = a2*b1, C22 = a2*b2
    #    Rank 4? No, we can do better.
    #    Actually, for 2x1 * 1x2, the tensor rank is 4? Let me check.
    #    The multiplication is (a1,a2) * (b1,b2) -> (a1*b1, a1*b2, a2*b1, a2*b2)
    #    This is just the outer product of a 2-vector and a 2-vector, which has rank 4.
    #    
    #    Hmm, let me reconsider. Let's use Strassen for the 2x2 blocks where possible.
    #
    # Let me re-decompose:
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x1 * 1x5 -> 2x5. 
    # A1 = [a1; a2], B1 = [b11, b12, b13, b14, b15]
    # C = [a1*b11, a1*b12, a1*b13, a1*b14, a1*b15; a2*b11, a2*b12, a2*b13, a2*b14, a2*b15]
    # This requires 10 multiplications? No, we can factor:
    # C[i,k] = A1[i] * B1[k]
    # We need 2*5 = 10 products? But we can reuse.
    # Actually, for rank-1 A and rank-1 B, the product is rank 1, so the tensor rank is n*p = 10.
    # Wait, no. The multiplication A1*B1 where A1 is 2x1 and B1 is 1x5.
    # The tensor T[i,k] = A1[i]*B1[k]. This is a rank-1 tensor in (i,k) space.
    # But we're computing A*B where A has columns and B has rows.
    # A1 is column j=0 of A, B1 is row j=0 of B.
    # The contribution is A[:,0] * B[0,:], which is an outer product.
    # To compute this, we need 1 multiplication per output element? No.
    # We need to compute c[i,k] += A[i,0]*B[0,k] for all i,k.
    # This is 10 separate multiplications, but they are independent.
    # So rank 10? That's worse than schoolbook.
    
    # Let me try a different approach. Use the known optimal rank for 2x3*3x5.
    # Actually, let me just use a careful Strassen-based decomposition.
    
    # Let's use the following block structure:
    # A is 2x3, split as A = [A_left(2x2) | A_right(2x1)]
    # B is 3x5, split as B = [B_top(2x5); B_bottom(1x5)]
    
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x5 -> use Strassen-like for 2x2 * 2xN
    # A_right * B_bottom: 2x1 * 1x5 -> 10 multiplications
    
    # For 2x2 * 2x5:
    # We can use 7 multiplications for each pair of columns.
    # Split B_top into 2 pairs of columns + 1 column.
    # B_top = [Bc1(2x2) | Bc2(2x2) | Bc3(2x1)]
    
    # A_left * Bc1: 2x2 * 2x2 -> 7 (Strassen)
    # A_left * Bc2: 2x2 * 2x2 -> 7 (Strassen)
    # A_left * Bc3: 2x2 * 2x1 -> 4 (trivial)
    
    # A_right * B_bottom: 2x1 * 1x5 -> 10
    
    # Total: 7 + 7 + 4 + 10 = 28. Not better than 30.
    
    # Let me try: A split 1+2, B split 2+3
    # A = [A1(2x1) | A2(2x2)]
    # B = [B1(1x5); B2(2x5)]
    
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x1 * 1x5 -> 10
    # A2*B2: 2x2 * 2x5 -> split B2 into B2a(2x2), B2b(2x2), B2c(2x1)
    # A2*B2a: 7, A2*B2b: 7, A2*B2c: 4
    # Total: 10 + 7 + 7 + 4 = 28.
    
    # Hmm. Let me think differently.
    
    # Actually, for 2x3 * 3x5, let me try to find a decomposition with rank < 30.
    # The trivial rank is 30.
    # 
    # Let me use the approach of computing with Strassen for overlapping blocks.
    # 
    # Alternative: use the fact that we can combine terms.
    # 
    # Let me just implement a decomposition with rank 28 using the block method above,
    # or try to find something better.
    #
    # Actually, let me reconsider. For 2x2 * 2xN, the rank is 7*ceil(N/2) + ... 
    # No, it's not quite that simple.
    #
    # Let me try a different block split.
    # A = 2x3, B = 3x5.
    # Split B's columns into 2+3: B = [B1(3x2) | B2(3x3)]
    # C = A*B1 + A*B2
    # A*B1: 2x3 * 3x2 -> 12 (schoolbook) or can we do better?
    # A*B2: 2x3 * 3x3 -> 18 (schoolbook)
    # Total: 30.
    
    # For 2x3 * 3x2:
    # C is 2x2. 
    # We can use Strassen-like. Split A's columns: A = [A1(2x1) | A2(2x2)]
    # B1's rows: B1 = [B11(1x2); B12(2x2)]
    # C = A1*B11 + A2*B12
    # A1*B11: 2x1 * 1x2 -> 4
    # A2*B12: 2x2 * 2x2 -> 7
    # Total: 11.
    
    # For 2x3 * 3x3:
    # C is 2x3.
    # Split A's columns: A = [A1(2x1) | A2(2x2)]
    # B2's rows: B2 = [B21(1x3); B22(2x3)]
    # C = A1*B21 + A2*B22
    # A1*B21: 2x1 * 1x3 -> 6
    # A2*B22: 2x2 * 2x3 -> split B22 into B22a(2x2) + B22b(2x1)
    # A2*B22a: 7, A2*B22b: 4
    # Total: 6 + 7 + 4 = 17.
    
    # Grand total: 11 + 17 = 28.
    
    # So rank 28. Let me implement this.
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A * B1 where B1 is 3x2 (columns 0,1 of B)
    # C1 = A * B1, C1 is 2x2
    # A1 = A[:, 0] (2x1), A2 = A[:, 1:3] (2x2)
    # B11 = B[0:1, 0:2] (1x2), B12 = B[1:3, 0:2] (2x2)
    
    # Sub-part 1a: A1 * B11 (2x1 * 1x2 -> 2x2), rank 4
    # C1[i,k] += A[i,0] * B[0,k] for i in 0,1 and k in 0,1
    for i in range(2):
        for k in range(2):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Sub-part 1b: A2 * B12 (2x2 * 2x2 -> 2x2), Strassen, rank 7
    # A2 = A[:, 1:3], B12 = B[1:3, 0:2]
    # C1[:, :] += A2 * B12
    # Strassen:
    # m1 = (A2[0,0] + A2[1,1]) * (B12[0,0] + B12[1,1])
    # m2 = (A2[1,0] + A2[1,1]) * B12[0,0]
    # m3 = A2[0,0] * (B12[1,0] - B12[1,1])
    # m4 = A2[1,1] * (B12[0,1] + B12[1,1])
    # m5 = (A2[0,0] + A2[0,1]) * B12[1,1]
    # m6 = (A2[1,0] - A2[0,0]) * (B12[0,0] + B12[0,1])
    # m7 = (A2[0,1] - A2[1,1]) * (B12[1,0] + B12[1,1])
    
    # C1[0,0] = m1 + m4 - m5 - m7
    # C1[0,1] = m3 + m5
    # C1[1,0] = m2 + m4
    # C1[1,1] = m1 - m2 + m6 + m7
    
    # m1
    U[0 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 0, col] = 1; V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = 1; W[1 * n + 1, col] = 1
    col += 1
    
    # m2
    U[1 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 0, col] = 1
    W[1 * n + 0, col] = 1
    col += 1
    
    # m3
    U[0 * m + 1, col] = 1
    V[2 * p + 0, col] = 1; V[2 * p + 1, col] = -1
    W[0 * n + 1, col] = 1
    col += 1
    
    # m4
    U[1 * m + 2, col] = 1
    V[1 * p + 1, col] = 1; V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = 1; W[1 * n + 0, col] = 1
    col += 1
    
    # m5
    U[0 * m + 1, col] = 1; U[0 * m + 2, col] = 1
    V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = -1; W[0 * n + 1, col] = 1
    col += 1
    
    # m6
    U[1 * m + 1, col] = 1; U[0 * m + 1, col] = -1
    V[1 * p + 0, col] = 1; V[1 * p + 1, col] = 1
    W[1 * n + 1, col] = 1
    col += 1
    
    # m7
    U[0 * m + 2, col] = 1; U[1 * m + 2, col] = -1
    V[2 * p + 0, col] = 1; V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = -1; W[1 * n + 1, col] = 1
    col += 1
    
    # Part 2: A * B2 where B2 is 3x3 (columns 2,3,4 of B)
    # C2 = A * B2, C2 is 2x3
    # A1 = A[:, 0] (2x1), A2 = A[:, 1:3] (2x2)
    # B21 = B[0:1, 2:5] (1x3), B22 = B[1:3, 2:5] (2x3)
    
    # Sub-part 2a: A1 * B21 (2x1 * 1x3 -> 2x3), rank 6
    for i in range(2):
        for k in range(3):
            U[i * m + 0, col] = 1
            V[0 * p + (k + 2), col] = 1
            W[(k + 2) * n + i, col] = 1
            col += 1
    
    # Sub-part 2b: A2 * B22 (2x2 * 2x3 -> 2x3)
    # Split B22 into B22a(2x2) for cols 2,3 and B22b(2x1) for col 4
    # A2 * B22a: 2x2 * 2x2 -> 7 (Strassen)
    # A2 * B22b: 2x2 * 2x1 -> 4
    
    # Sub-part 2b-i: A2 * B22a (2x2 * 2x2), Strassen, rank 7
    # B22a = B[1:3, 2:4]
    # C2[:, 2:4] += A2 * B22a
    
    # m1 = (A2[0,0] + A2[1,1]) * (B22a[0,0] + B22a[1,1])
    U[0 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 2, col] = 1; V[2 * p + 3, col] = 1
    W[2 * n + 0, col] = 1; W[3 * n + 1, col] = 1
    col += 1
    
    # m2 = (A2[1,0] + A2[1,1]) * B22a[0,0]
    U[1 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 2, col] = 1
    W[2 * n + 1, col] = 1
    col += 1
    
    # m3 = A2[0,0] * (B22a[1,0] - B22a[1,1])
    U[0 * m + 1, col] = 1
    V[2 * p + 2, col] = 1; V[2 * p + 3, col] = -1
    W[3 * n + 0, col] = 1
    col += 1
    
    # m4 = A2[1,1] * (B22a[0,1] + B22a[1,1])
    U[1 * m + 2, col] = 1
    V[1 * p + 3, col] = 1; V[2 * p + 3, col] = 1
    W[2 * n + 0, col] = 1; W[3 * n + 0, col] = 1
    col += 1
    
    # m5 = (A2[0,0] + A2[0,1]) * B22a[1,1]
    U[0 * m + 1, col] = 1; U[0 * m + 2, col] = 1
    V[2 * p + 3, col] = 1
    W[2 * n + 0, col] = -1; W[3 * n + 1, col] = 1
    col += 1
    
    # m6 = (A2[1,0] - A2[0,0]) * (B22a[0,0] + B22a[0,1])
    U[1 * m + 1, col] = 1; U[0 * m + 1, col] = -1
    V[1 * p + 2, col] = 1; V[1 * p + 3, col] = 1
    W[3 * n + 1, col] = 1
    col += 1
    
    # m7 = (A2[0,1] - A2[1,1]) * (B22a[1,0] + B22a[1,1])
    U[0 * m + 2, col] = 1; U[1 * m + 2, col] = -1
    V[2 * p + 2, col] = 1; V[2 * p + 3, col] = 1
    W[2 * n + 0, col] = -1; W[3 * n + 1, col] = 1
    col += 1
    
    # Sub-part 2b-ii: A2 * B22b (2x2 * 2x1 -> 2x1), rank 4
    # B22b = B[1:3, 4:5]
    # C2[:, 4] += A2 * B22b
    for i in range(2):
        for j in range(2):
            U[i * m + (j + 1), col] = 1
            V[(j + 1) * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
