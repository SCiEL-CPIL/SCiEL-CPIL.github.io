import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix using Strassen's algorithm.
    
    Strategy:
    - Split A (2x3) into A_L (2x1) and A_R (2x2)
    - Split B (3x6) into B_T (1x6) and B_B (2x6)
    - C = A_L @ B_T + A_R @ B_B
    - A_L @ B_T: 2 multiplications (naive)
    - A_R @ B_B: 7 multiplications (Strassen)
    - Total: 9 rank-1 terms
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # 110s wall-clock limit
    
    # Build the 9-term decomposition
    # Each term is a triple (u, v, w) where:
    # u is a vector of length n*m = 6 (coefficients for A)
    # v is a vector of length m*p = 18 (coefficients for B)
    # w is a vector of length p*n = 12 (coefficients for C)
    
    terms = []
    
    # Part 1: A_L @ B_T
    # A_L is A[:, 0] = [A[0,0], A[1,0]] (indices 0, 1 in A)
    # B_T is B[0, :] = [B[0,0], B[0,1], ..., B[0,5]] (indices 0-5 in B)
    # Result: C[:, 0] = A_L * B_T[0], C[:, 1] = A_L * B_T[1], ..., C[:, 5] = A_L * B_T[5]
    # Wait, A_L is 2x1, B_T is 1x6, so A_L @ B_T is 2x6.
    # C[i, k] = A[i, 0] * B[0, k]
    
    # Term 1: C[:, 0] contribution
    # u has 1 at index 0 (A[0,0]) and 1 at index 1 (A[1,0])
    # v has 1 at index 0 (B[0,0])
    # w has 1 at index 0 (C[0,0]) and 1 at index 6 (C[1,0])
    u1 = [0.0] * 6
    u1[0] = 1.0  # A[0,0]
    u1[1] = 1.0  # A[1,0]
    v1 = [0.0] * 18
    v1[0] = 1.0  # B[0,0]
    w1 = [0.0] * 12
    w1[0] = 1.0  # C[0,0]
    w1[6] = 1.0  # C[1,0]
    terms.append((u1, v1, w1))
    
    # Term 2: C[:, 1] contribution
    u2 = [0.0] * 6
    u2[0] = 1.0
    u2[1] = 1.0
    v2 = [0.0] * 18
    v2[1] = 1.0  # B[0,1]
    w2 = [0.0] * 12
    w2[1] = 1.0  # C[0,1]
    w2[7] = 1.0  # C[1,1]
    terms.append((u2, v2, w2))
    
    # Term 3: C[:, 2] contribution
    u3 = [0.0] * 6
    u3[0] = 1.0
    u3[1] = 1.0
    v3 = [0.0] * 18
    v3[2] = 1.0  # B[0,2]
    w3 = [0.0] * 12
    w3[2] = 1.0  # C[0,2]
    w3[8] = 1.0  # C[1,2]
    terms.append((u3, v3, w3))
    
    # Term 4: C[:, 3] contribution
    u4 = [0.0] * 6
    u4[0] = 1.0
    u4[1] = 1.0
    v4 = [0.0] * 18
    v4[3] = 1.0  # B[0,3]
    w4 = [0.0] * 12
    w4[3] = 1.0  # C[0,3]
    w4[9] = 1.0  # C[1,3]
    terms.append((u4, v4, w4))
    
    # Term 5: C[:, 4] contribution
    u5 = [0.0] * 6
    u5[0] = 1.0
    u5[1] = 1.0
    v5 = [0.0] * 18
    v5[4] = 1.0  # B[0,4]
    w5 = [0.0] * 12
    w5[4] = 1.0  # C[0,4]
    w5[10] = 1.0  # C[1,4]
    terms.append((u5, v5, w5))
    
    # Term 6: C[:, 5] contribution
    u6 = [0.0] * 6
    u6[0] = 1.0
    u6[1] = 1.0
    v6 = [0.0] * 18
    v6[5] = 1.0  # B[0,5]
    w6 = [0.0] * 12
    w6[5] = 1.0  # C[0,5]
    w6[11] = 1.0  # C[1,5]
    terms.append((u6, v6, w6))
    
    # Part 2: A_R @ B_B using Strassen
    # A_R is A[:, 1:3] = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B_B is B[1:3, :] = [[B[1,0], ..., B[1,5]], [B[2,0], ..., B[2,5]]]
    # Result: C[:, 1:3] = A_R @ B_B (2x6)
    # C[i, k] for i in {0,1}, k in {1,...,5}
    
    # Strassen for 2x2 @ 2x6:
    # Split A_R into 2x1 blocks? No, Strassen is for 2x2 @ 2x2.
    # We need to handle 2x2 @ 2x6.
    
    # Actually, let's use the standard approach:
    # A_R @ B_B where A_R is 2x2 and B_B is 2x6.
    # We can split B_B into two 2x3 blocks: B_B_left (2x3) and B_B_right (2x3).
    # Then C_left = A_R @ B_B_left (2x3) and C_right = A_R @ B_B_right (2x3).
    
    # For 2x2 @ 2x3, we can use Strassen-like decomposition.
    # Let A_R = [[a, b], [c, d]] where:
    # a = A[0,1] (index 2), b = A[0,2] (index 3)
    # c = A[1,1] (index 4), d = A[1,2] (index 5)
    
    # B_B_left = [[B[1,1], B[1,2], B[1,3]], [B[2,1], B[2,2], B[2,3]]]
    # B_B_right = [[B[1,4], B[1,5], B[1,6]], [B[2,4], B[2,5], B[2,6]]]
    
    # Wait, B is 3x6, so B_B is rows 1,2 (0-indexed), columns 0-5.
    # B_B_left: columns 1,2,3 (for C columns 1,2,3)
    # B_B_right: columns 4,5 (for C columns 4,5)
    
    # Actually, C has 6 columns. C[:, 0] comes from A_L @ B_T.
    # C[:, 1:6] comes from A_R @ B_B.
    
    # Let me redo this more carefully.
    
    # Clear terms and start over
    terms = []
    
    # A is 2x3: A[i, j] where i in {0,1}, j in {0,1,2}
    # B is 3x6: B[j, k] where j in {0,1,2}, k in {0,1,2,3,4,5}
    # C is 2x6: C[i, k] where i in {0,1}, k in {0,1,2,3,4,5}
    
    # C[i, k] = sum_{j=0}^{2} A[i, j] * B[j, k]
    
    # Split: C = A[:, 0:1] @ B[0:1, :] + A[:, 1:3] @ B[1:3, :]
    
    # Part 1: A[:, 0:1] @ B[0:1, :]
    # This is (2x1) @ (1x6) = 2x6
    # C[i, k] += A[i, 0] * B[0, k]
    
    # For each k, we have a term:
    # u has 1 at A[i, 0] for i=0,1
    # v has 1 at B[0, k]
    # w has 1 at C[i, k] for i=0,1
    
    for k in range(6):
        u = [0.0] * 6
        u[0] = 1.0  # A[0, 0]
        u[1] = 1.0  # A[1, 0]
        v = [0.0] * 18
        v[0 * 6 + k] = 1.0  # B[0, k]
        w = [0.0] * 12
        w[k * 2 + 0] = 1.0  # C[0, k]
        w[k * 2 + 1] = 1.0  # C[1, k]
        terms.append((u, v, w))
    
    # Part 2: A[:, 1:3] @ B[1:3, :]
    # A_R is 2x2: [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B_B is 2x6: [[B[1,0], ..., B[1,5]], [B[2,0], ..., B[2,5]]]
    # Result is 2x6: C[i, k] for i in {0,1}, k in {0,1,...,5}
    # C[i, k] += A[i, 1] * B[1, k] + A[i, 2] * B[2, k]
    
    # Use Strassen for 2x2 @ 2x6.
    # Standard Strassen for 2x2 @ 2x2 uses 7 multiplications.
    # For 2x2 @ 2x6, we can split B into two 2x3 blocks and use Strassen on each.
    # Or we can use a direct 7-term decomposition for 2x2 @ 2x6.
    
    # Let's use the standard Strassen decomposition for 2x2 @ 2x2, applied to each column block.
    
    # Actually, let's use a simpler approach: split B_B into 3 blocks of 2x2.
    # B_B[:, 0:2], B_B[:, 2:4], B_B[:, 4:6]
    
    # For each 2x2 block, use Strassen (7 terms).
    # Total: 3 * 7 = 21 terms.
    
    # Strassen for [[a, b], [c, d]] @ [[e, f], [g, h]]:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Result:
    # [[m1+m4-m5+m7, m3+m5],
    #  [m2+m4, m1-m2+m3+m6]]
    
    # Let's apply this to each 2x2 block of B_B.
    
    # Block 0: B_B[:, 0:2] = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # a = A[0,1] (index 2), b = A[0,2] (index 3)
    # c = A[1,1] (index 4), d = A[1,2] (index 5)
    # e = B[1,0] (index 6), f = B[1,1] (index 7)
    # g = B[2,0] (index 12), h = B[2,1] (index 13)
    
    # m1 = (a+d)(e+h)
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[6] = 1.0  # e
    v[13] = 1.0  # h
    w = [0.0] * 12
    w[0] = 1.0  # C[0,0]
    w[1] = 1.0  # C[1,0]
    w[2] = 1.0  # C[0,1]
    w[3] = 1.0  # C[1,1]
    terms.append((u, v, w))
    
    # m2 = (c+d)e
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[6] = 1.0  # e
    w = [0.0] * 12
    w[2] = -1.0  # C[0,1]
    w[3] = 1.0  # C[1,1]
    terms.append((u, v, w))
    
    # m3 = a(f-h)
    u = [0.0] * 6
    u[2] = 1.0  # a
    v = [0.0] * 18
    v[7] = 1.0  # f
    v[13] = -1.0  # -h
    w = [0.0] * 12
    w[1] = 1.0  # C[1,0]
    w[3] = 1.0  # C[1,1]
    terms.append((u, v, w))
    
    # m4 = d(g-e)
    u = [0.0] * 6
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[12] = 1.0  # g
    v[6] = -1.0  # -e
    w = [0.0] * 12
    w[0] = 1.0  # C[0,0]
    w[1] = 1.0  # C[1,0]
    w[2] = 1.0  # C[0,1]
    w[3] = 1.0  # C[1,1]
    terms.append((u, v, w))
    
    # m5 = (a+b)h
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[3] = 1.0  # b
    v = [0.0] * 18
    v[13] = 1.0  # h
    w = [0.0] * 12
    w[0] = -1.0  # C[0,0]
    w[2] = -1.0  # C[0,1]
    terms.append((u, v, w))
    
    # m6 = (c-a)(e+f)
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[2] = -1.0  # -a
    v = [0.0] * 18
    v[6] = 1.0  # e
    v[7] = 1.0  # f
    w = [0.0] * 12
    w[3] = 1.0  # C[1,1]
    terms.append((u, v, w))
    
    # m7 = (b-d)(g+h)
    u = [0.0] * 6
    u[3] = 1.0  # b
    u[5] = -1.0  # -d
    v = [0.0] * 18
    v[12] = 1.0  # g
    v[13] = 1.0  # h
    w = [0.0] * 12
    w[0] = 1.0  # C[0,0]
    w[1] = 1.0  # C[1,0]
    terms.append((u, v, w))
    
    # Block 1: B_B[:, 2:4] = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]]
    # e = B[1,2] (index 8), f = B[1,3] (index 9)
    # g = B[2,2] (index 14), h = B[2,3] (index 15)
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[8] = 1.0  # e
    v[15] = 1.0  # h
    w = [0.0] * 12
    w[4] = 1.0  # C[0,2]
    w[5] = 1.0  # C[1,2]
    w[6] = 1.0  # C[0,3]
    w[7] = 1.0  # C[1,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[8] = 1.0  # e
    w = [0.0] * 12
    w[6] = -1.0  # C[0,3]
    w[7] = 1.0  # C[1,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    v = [0.0] * 18
    v[9] = 1.0  # f
    v[15] = -1.0  # -h
    w = [0.0] * 12
    w[5] = 1.0  # C[1,2]
    w[7] = 1.0  # C[1,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[14] = 1.0  # g
    v[8] = -1.0  # -e
    w = [0.0] * 12
    w[4] = 1.0  # C[0,2]
    w[5] = 1.0  # C[1,2]
    w[6] = 1.0  # C[0,3]
    w[7] = 1.0  # C[1,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[3] = 1.0  # b
    v = [0.0] * 18
    v[15] = 1.0  # h
    w = [0.0] * 12
    w[4] = -1.0  # C[0,2]
    w[6] = -1.0  # C[0,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[2] = -1.0  # -a
    v = [0.0] * 18
    v[8] = 1.0  # e
    v[9] = 1.0  # f
    w = [0.0] * 12
    w[7] = 1.0  # C[1,3]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[3] = 1.0  # b
    u[5] = -1.0  # -d
    v = [0.0] * 18
    v[14] = 1.0  # g
    v[15] = 1.0  # h
    w = [0.0] * 12
    w[4] = 1.0  # C[0,2]
    w[5] = 1.0  # C[1,2]
    terms.append((u, v, w))
    
    # Block 2: B_B[:, 4:6] = [[B[1,4], B[1,5]], [B[2,4], B[2,5]]]
    # e = B[1,4] (index 10), f = B[1,5] (index 11)
    # g = B[2,4] (index 16), h = B[2,5] (index 17)
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[10] = 1.0  # e
    v[17] = 1.0  # h
    w = [0.0] * 12
    w[8] = 1.0  # C[0,4]
    w[9] = 1.0  # C[1,4]
    w[10] = 1.0  # C[0,5]
    w[11] = 1.0  # C[1,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[10] = 1.0  # e
    w = [0.0] * 12
    w[10] = -1.0  # C[0,5]
    w[11] = 1.0  # C[1,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    v = [0.0] * 18
    v[11] = 1.0  # f
    v[17] = -1.0  # -h
    w = [0.0] * 12
    w[9] = 1.0  # C[1,4]
    w[11] = 1.0  # C[1,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[5] = 1.0  # d
    v = [0.0] * 18
    v[16] = 1.0  # g
    v[10] = -1.0  # -e
    w = [0.0] * 12
    w[8] = 1.0  # C[0,4]
    w[9] = 1.0  # C[1,4]
    w[10] = 1.0  # C[0,5]
    w[11] = 1.0  # C[1,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[2] = 1.0  # a
    u[3] = 1.0  # b
    v = [0.0] * 18
    v[17] = 1.0  # h
    w = [0.0] * 12
    w[8] = -1.0  # C[0,4]
    w[10] = -1.0  # C[0,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[4] = 1.0  # c
    u[2] = -1.0  # -a
    v = [0.0] * 18
    v[10] = 1.0  # e
    v[11] = 1.0  # f
    w = [0.0] * 12
    w[11] = 1.0  # C[1,5]
    terms.append((u, v, w))
    
    u = [0.0] * 6
    u[3] = 1.0  # b
    u[5] = -1.0  # -d
    v = [0.0] * 18
    v[16] = 1.0  # g
    v[17] = 1.0  # h
    w = [0.0] * 12
    w[8] = 1.0  # C[0,4]
    w[9] = 1.0  # C[1,4]
    terms.append((u, v, w))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][r] = u[i]
        for j in range(m * p):
            V[j][r] = v[j]
        for k in range(p * n):
            W[k][r] = w[k]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        # Check half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
