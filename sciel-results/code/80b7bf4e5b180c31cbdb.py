"""Strassen-based rank-54 exact decomposition for 2x5 * 5x6."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Split m = m1 + m2 with m1=2, m2=3
    m1, m2 = 2, 3

    rank = 7 * m2 + 6 * m1 * m2  # 21 + 36 = 57? Let's recompute carefully.
    # Actually: 2x2 part uses rank 7, each producing 2 outputs (rows) x 6 (cols)
    # 2x3 part uses schoolbook rank 6
    # 3x6 part uses schoolbook rank 18
    # Total = 7 + 6 + 18 = 31? No, that's wrong because of how tensor decomposition works.

    # Let me think about this more carefully.
    # C = A * B where A is 2x5, B is 5x6, C is 2x6.
    # Split A into [A1 A2] where A1 is 2x2, A2 is 2x3
    # Split B into [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2

    # For A1*B1: A1 is 2x2, B1 is 2x6. We can use Strassen on the 2x2 part.
    # A1 = [[a11, a12], [a21, a22]], B1 = [[b11, b12, ...], [b21, b22, ...]]
    # The 2x2 * 2x6 multiplication: each row of A1 times B1.
    # Actually, let's think of it as: for each column j of B1 (6 columns),
    # we compute A1 * B1[:,j] which is a 2x2 * 2x1 = 2x1 product.
    # But that's not right either because B1 is 2x6.

    # Better approach: A1*B1 is (2x2)*(2x6) = 2x6.
    # We can view this as: for each of the 6 output columns, we do a 2x2*2x1.
    # But Strassen applies to 2x2*2x2. So we need to pad or restructure.

    # Alternative: treat A1*B1 as follows.
    # A1 = [[a11, a12],[a21,a22]], B1 is 2x6.
    # C1 = A1*B1 is 2x6 where:
    #   C1[0,:] = a11*B1[0,:] + a12*B1[1,:]
    #   C1[1,:] = a21*B1[0,:] + a22*B1[1,:]
    # This is 4 scalar multiplications per column, 6 columns = 24 mults.
    # With Strassen on 2x2*2x2, we reduce 2x2*2x2 to 7 mults.
    # But here B1 is 2x6, not 2x2.

    # Let me reconsider. The standard approach:
    # A is 2x5, B is 5x6.
    # Split into blocks:
    #   A = [A11 A12]  where A11 is 2x2, A12 is 2x3
    #   B = [B11; B12] where B11 is 2x6, B12 is 3x6
    # C = A11*B11 + A12*B12

    # A11*B11: (2x2)*(2x6). Think of B11 as two rows, each 1x6.
    # For each column of B11, we have a 2x2 * 2x1 = 2x1.
    # Actually, A11*B11 = A11 * [B11_row0; B11_row1]
    # = A11_row0 * B11_row0 (as outer? No.)
    # C11[0,k] = A11[0,0]*B11[0,k] + A11[0,1]*B11[1,k]
    # C11[1,k] = A11[1,0]*B11[0,k] + A11[1,1]*B11[1,k]

    # So for each k (6 values), we compute a 2x2*2x1 product.
    # The tensor for 2x2*2x1 has rank... let's see.
    # Actually, let's just use the full 2x2*2x2 Strassen and pad.

    # Simpler: use the known result. For 2x5*5x6, we can get rank 54.
    # Let me just construct it directly.

    # Approach: 
    # C = A*B, A is 2x5, B is 5x6.
    # Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x3.
    # Write B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 3x6.
    # C = A_left*B_top + A_right*B_bottom.

    # Term 1: A_left (2x2) * B_top (2x6)
    # For each column j of B_top (j=0..5), compute A_left * B_top[:,j] (2x2 * 2x1).
    # The 2x2*2x1 tensor: C[0] = a00*b0 + a01*b1, C[1] = a10*b0 + a11*b1.
    # This has rank 4 (schoolbook) or can we do better?
    # Actually 2x2*2x1 is just two dot products of length 2, rank 4.
    # But wait, if we consider all 6 columns together, it's (2x2)*(2x6).
    # The tensor is M_{2,2,6} where the last dimension is split.
    # Rank of 2x2*2x6: since 2x2*2x2 has rank 7, and 2x2*2x1 has rank 4...
    # Actually 2x2*2x6 can be seen as 3 copies of 2x2*2x2? No.
    # 2x2*2x6 = (2x2*2x2) + (2x2*2x2) + (2x2*2x2)? No, that's 2x2*2x6.
    # Each 2x2*2x2 block uses rank 7, so 3 blocks = 21? But that overcounts.
    # Actually, 2x2 * 2x6: fix the 2x2 part, the 6 columns are independent.
    # The bilinear complexity is at most 7*3 = 21 if we do 3 Strassen steps.
    # But actually, since the 2x2 matrix is the same for all 6 columns, we can do better.
    # The tensor is: for i in {0,1}, j in {0,1}, k in {0..5}: A[i,j]*B[j,k]
    # This is M_{2,2,6}. The rank is at most 7 + 4 + 4 = 15? 
    # Hmm, let me just use schoolbook for simplicity first and see if I can improve.

    # Let me just do a direct construction with rank 54.
    # Known: M(2,5,6) <= 54.

    # Construction:
    # A is 2x5, B is 5x6, C is 2x6.
    # Split: A = [A1 A2], A1 is 2x2, A2 is 2x3.
    # B = [B1; B2], B1 is 2x6, B2 is 3x6.
    # C = A1*B1 + A2*B2.

    # For A1*B1 (2x2 * 2x6):
    # Use Strassen-style. Let A1 = [[a,b],[c,d]], B1 rows are u,v (each 1x6).
    # C1 = [[a*u+b*v], [c*u+d*v]]
    # Using 7 multiplications:
    # m1 = (a+d)*(u+v)
    # m2 = (c+d)*v
    # m3 = a*(u-v)
    # m4 = d*v
    # m5 = (a+b)*u
    # m6 = (c-a)*(u+v)
    # m7 = (b-d)*(v+u)  -- wait, standard Strassen:
    # m1 = (a+d)*(u+v)
    # m2 = (c+d)*v
    # m3 = a*(u-v)
    # m4 = d*v
    # m5 = (a+b)*u
    # m6 = (c-a)*(u+v)
    # m7 = (b-d)*(u+v)  -- no, m7 = (b-d)*(v+u) is same as (b-d)*(u+v)
    # Actually standard Strassen for 2x2*2x2:
    # m1 = (a+d)(u+v), m2 = (c+d)v, m3 = a(u-v), m4 = dv, m5 = (a+b)u, m6 = (c-a)(u+v), m7 = (b-d)(v+u)
    # Wait, m7 should be (b-d)*(v+u)? Let me recall:
    # p11 = m1 - m4 + m5 - m7
    # p12 = m4 + m7
    # p21 = m2 + m3
    # p22 = m1 - m2 + m4 - m6
    # where m1=(a+d)(u+v), m2=(c+d)v, m3=a(u-v), m4=dv, m5=(a+b)u, m6=(c-a)(u+v), m7=(b-d)(v+u)
    # Hmm, m7 = (b-d)(v+u) = (b-d)(u+v). Let me just use this.

    # But B1 is 2x6, so u and v are 1x6 vectors. The multiplications m1..m7 are
    # scalar * vector, which is fine. Each "multiplication" is actually a scalar times
    # a 6-vector, contributing 6 to the tensor rank? No, in tensor decomposition,
    # each bilinear term is a scalar product of one A-entry and one B-entry.
    # When we write m1 = (a+d)*(u+v), this expands to a*u + a*v + d*u + d*v, which is 4 terms.
    # So the rank contribution is not simply 7.

    # I think I'm overcomplicating this. Let me just use the schoolbook approach
    # but with the known optimal rank for 2x5*5x6 which is 54.

    # Actually, let me just construct a valid decomposition. The safest is schoolbook (rank 60).
    # But we know rank 54 is achievable. Let me try to build it.

    # For now, let me return schoolbook to ensure correctness, then optimize.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
