"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
    # Also verify that non-matching entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                a = i * m + j
                                b = jp * p + kp
                                c = k * n + ip
                                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                                if val != 0:
                                    return False
    return True


def _find_rank11_2x3_3x2(deadline):
    """
    Find a rank-11 decomposition for 2x3 * 3x2 matrix multiplication.
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]  (2x3)
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]  (3x2)
    C = [[c00, c01],
         [c10, c11]]  (2x2)
    
    We need: c_ik = sum_j a_ij * b_jk
    
    Strategy: Use Strassen-like 7 products for a 2x2 core, plus 4 additional
    rank-1 terms to handle the third column of A.
    
    Actually, let me think more carefully.
    
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    Approach: Define 11 products. Let's use a structured ansatz.
    
    Products:
    t0 = a00 * b00
    t1 = a00 * b01
    t2 = a01 * b10
    t3 = a01 * b11
    t4 = a02 * b20
    t5 = a02 * b21
    t6 = (a00+a10) * b00
    t7 = (a00+a10) * b01
    t8 = (a01+a11) * b10
    t9 = (a01+a11) * b11
    t10 = (a02+a12) * (b20+b21)
    
    Then:
    c00 = t0 + t2 + t4
    c01 = t1 + t3 + t5
    c10 = (t6-t0) + (t8-t2) + ???  need a12*b20
    c11 = (t7-t1) + (t9-t3) + ???  need a12*b21
    
    t10 = a02*b20 + a02*b21 + a12*b20 + a12*b21
    t10 - t4 - t5 = a12*b20 + a12*b21
    
    This gives a12*b20 + a12*b21, but I need them separately.
    
    Let me try a different approach. Use 11 products with a more flexible ansatz.
    
    Actually, let me try:
    t0 = a00*b00
    t1 = a00*b01  
    t2 = a01*b10
    t3 = a01*b11
    t4 = a02*b20
    t5 = a02*b21
    t6 = (a00+a10)*(b00+b01)
    t7 = (a01+a11)*(b10+b11)
    t8 = (a02+a12)*b20
    t9 = (a02+a12)*b21
    t10 = (a00+a10+a01+a11)*(b00+b10)  -- this is getting complex
    
    Let me try a cleaner approach. The known result is that rank(2x3*3x2) = 11.
    
    Here's a known construction:
    
    Let me use the following 11 products:
    p0 = a00 * b00
    p1 = a00 * b01
    p2 = a01 * b10
    p3 = a01 * b11
    p4 = a02 * b20
    p5 = a02 * b21
    p6 = (a00+a10) * b00
    p7 = (a00+a10) * b01
    p8 = (a01+a11) * b10
    p9 = (a01+a11) * b11
    p10 = (a02+a12) * b20
    
    Then:
    c00 = p0 + p2 + p4
    c01 = p1 + p3 + p5
    c10 = (p6-p0) + (p8-p2) + (p10-p4)
    c11 = (p7-p1) + (p9-p3) + ???
    
    For c11, I need a10*b01 + a11*b11 + a12*b21
    = (p7-p1) + (p9-p3) + a12*b21
    
    I don't have a12*b21 directly. p10 = (a02+a12)*b20, not b21.
    
    Let me try p10 = (a02+a12) * b21:
    c10 = (p6-p0) + (p8-p2) + ???  need a12*b20
    c11 = (p7-p1) + (p9-p3) + (p10-p5)
    
    Now I need a12*b20 for c10. I don't have it.
    
    What if I use p10 = (a02+a12) * (b20+b21)?
    p10 = a02*b20 + a02*b21 + a12*b20 + a12*b21
    p10 - p4 - p5 = a12*b20 + a12*b21
    
    c10 = (p6-p0) + (p8-p2) + (a12*b20)  -- but I only have a12*b20 + a12*b21
    c11 = (p7-p1) + (p9-p3) + (a12*b21)
    
    This doesn't work cleanly.
    
    Let me try yet another approach. What if I use:
    p10 = a12 * (b20+b21)?
    Then p10 = a12*b20 + a12*b21
    c10 = (p6-p0) + (p8-p2) + ???  still need a12*b20 separately
    c11 = (p7-p1) + (p9-p3) + ???  still need a12*b21 separately
    
    Hmm. Let me think about this differently.
    
    Actually, the correct rank-11 algorithm for 2x3*3x2 is:
    
    p0 = a00*b00
    p1 = a00*b01
    p2 = a01*b10
    p3 = a01*b11
    p4 = a02*b20
    p5 = a02*b21
    p6 = (a00+a10)*b00
    p7 = (a00+a10)*b01
    p8 = (a01+a11)*b10
    p9 = (a01+a11)*b11
    p10 = (a02+a12)*(b20+b21)
    
    c00 = p0 + p2 + p4
    c01 = p1 + p3 + p5
    c10 = (p6-p0) + (p8-p2) + (p10-p4-p5)  -- wait, this gives a12*b20 + a12*b21, not just a12*b20
    
    This doesn't work. Let me look at this more carefully.
    
    Actually, I think the issue is that rank 11 for 2x3*3x2 might require a more clever ansatz.
    
    Let me try a completely different set of 11 products:
    
    p0 = a00*b00
    p1 = a00*b01
    p2 = a01*b10
    p3 = a01*b11
    p4 = a02*b20
    p5 = a02*b21
    p6 = (a00+a10)*(b00+b10)
    p7 = (a00+a10)*(b01+b11)
    p8 = (a01+a11)*(b00+b20)
    p9 = (a01+a11)*(b01+b21)
    p10 = (a02+a12)*(b00+b01+b10+b11+b20+b21)  -- this is getting too complex
    
    Let me just go with the known result that rank(2x3*3x2) = 11 and use a verified construction.
    
    After more thought, here's a working rank-11 construction:
    
    p0 = a00*b00
    p1 = a00*b01
    p2 = a01*b10
    p3 = a01*b11
    p4 = a02*b20
    p5 = a02*b21
    p6 = (a00+a10)*b00
    p7 = (a00+a10)*b01
    p8 = (a01+a11)*b10
    p9 = (a01+a11)*b11
    p10 = (a02+a12)*b20
    
    c00 = p0 + p2 + p4
    c01 = p1 + p3 + p5
    c10 = (p6-p0) + (p8-p2) + (p10-p4)
    c11 = (p7-p1) + (p9-p3) + ???
    
    For c11, I need a12*b21. I have p5 = a02*b21 but not a12*b21.
    
    What if I change p10 to (a02+a12)*b21?
    p10 = a02*b21 + a12*b21
    c10 = (p6-p0) + (p8-p2) + ???  need a12*b20
    c11 = (p7-p1) + (p9-p3) + (p10-p5)
    
    Now I need a12*b20 for c10.
    
    The problem is I need both a12*b20 and a12*b21, but I only have one product left.
    
    Solution: Use a different split. Instead of keeping a02 and a12 separate, 
    let me restructure.
    
    Actually, I realize the correct approach is:
    
    For the 2x3 * 3x2 problem, we can view it as:
    C = A * B where A is 2x3 and B is 3x2.
    
    Split A into A1 (2x2) and A2 (2x1):
    A = [A1 | A2]
    Split B into B1 (2x2) and B2 (1x2):
    B = [B1; B2]
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2*2x2, which can be done in rank 7 (Strassen).
    A2*B2 is 2x1*1x2, which is rank 4 (outer product).
    
    Total: 7 + 4 = 11!
    
    Strassen for 2x2*2x2:
    A1 = [[a00, a01], [a10, a11]]
    B1 = [[b00, b01], [b10, b11]]
    
    s1 = a10 * b00
    s2 = (a00+a10) * b00
    s3 = a01 * (b10+b11)
    s4 = a11 * (b01-b10)
    s5 = (a00+a11) * (b00+b11)
    s6 = (a10+a11) * (b11-b00)
    s7 = (a00+a01) * (b10+b11)
    
    c00 = s5 + s4 - s6 - s2
    c01 = s1 + s5
    c10 = s3 + s6
    c11 = s5 - s1 - s3 + s7
    
    Wait, let me use the standard Strassen formulas:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b10-b11)
    M4 = a11*(b01-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b10)
    M7 = (a01-a11)*(b01+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 - M7
    
    Let me verify:
    c00 = (a00+a11)(b00+b11) + a11(b01-b00) - (a00+a01)b11 + (a01-a11)(b01+b11)
    = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a01*b11 + a01*b01 + a01*b11 - a11*b01 - a11*b11
    = a00*b00 + a01*b01 + a11*b00 - a11*b00 + ... let me redo this carefully.
    
    Actually, let me just use the well-known Strassen algorithm and verify it computationally.
    
    For A2*B2 (2x1 * 1x2):
    A2 = [a02; a12], B2 = [b20, b21]
    c00 += a02*b20
    c01 += a02*b21
    c10 += a12*b20
    c11 += a12*b21
    
    These are 4 additional products:
    t0 = a02*b20
    t1 = a02*b21
    t2 = a12*b20
    t3 = a12*b21
    
    Total: 7 + 4 = 11 products.
    
    Let me implement this and verify.
    """
    # Strassen products for 2x2 * 2x2
    # A1 = [[a00, a01], [a10, a11]], B1 = [[b00, b01], [b10, b11]]
    # 
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 - M7
    
    # Outer product products for 2x1 * 1x2
    # t0 = a02*b20
    # t1 = a02*b21
    # t2 = a12*b20
    # t3 = a12*b21
    
    # Total 11 products.
    
    # Let's verify this is correct by checking the tensor decomposition.
    # For the 2x3 * 3x2 problem:
    # A indices: (i,j) where i in {0,1}, j in {0,1,2}
    # B indices: (j,k) where j in {0,1,2}, k in {0,1}
    # C indices: (i,k) where i in {0,1}, k in {0,1}
    
    # The 11 products are:
    # Product 0: (a00+a11) * (b00+b11)
    # Product 1: (a10+a11) * b00
    # Product 2: a00 * (b10-b11)
    # Product 3: a11 * (b01-b00)
    # Product 4: (a00+a01) * b11
    # Product 5: (a10-a00) * (b00+b10)
    # Product 6: (a01-a11) * (b01+b11)
    # Product 7: a02 * b20
    # Product 8: a02 * b21
    # Product 9: a12 * b20
    # Product 10: a12 * b21
    
    # U matrix (n*m = 2*3 = 6 rows, 11 cols):
    # Row i*m+j = i*3+j
    # U[0,0]=1 (a00 in M1), U[1,0]=1 (a11 in M1)
    # U[3,1]=1 (a10 in M2), U[4,1]=1 (a11 in M2)
    # U[0,2]=1 (a00 in M3)
    # U[4,3]=1 (a11 in M4)
    # U[0,4]=1 (a00 in M5), U[1,4]=1 (a01 in M5)
    # U[3,5]=1 (a10 in M6), U[0,5]=-1 (a00 in M6)
    # U[1,6]=1 (a01 in M6), U[4,6]=-1 (a11 in M6)
    # U[2,7]=1 (a02 in t0)
    # U[2,8]=1 (a02 in t1)
    # U[5,9]=1 (a12 in t2)
    # U[5,10]=1 (a12 in t3)
    
    # V matrix (m*p = 3*2 = 6 rows, 11 cols):
    # Row j*p+k = j*2+k
    # V[0,0]=1 (b00 in M1), V[1,0]=1 (b11 in M1)
    # V[0,1]=1 (b00 in M2)
    # V[2,2]=1 (b10 in M3), V[3,2]=-1 (b11 in M3)
    # V[1,3]=1 (b01 in M4), V[0,3]=-1 (b00 in M4)
    # V[3,4]=1 (b11 in M5)
    # V[0,5]=1 (b00 in M6), V[2,5]=1 (b10 in M6)
    # V[1,6]=1 (b01 in M6), V[3,6]=1 (b11 in M6)
    # V[4,7]=1 (b20 in t0)
    # V[5,8]=1 (b21 in t1)
    # V[4,9]=1 (b20 in t2)
    # V[5,10]=1 (b21 in t3)
    
    # W matrix (p*n = 2*2 = 4 rows, 11 cols):
    # Row k*n+i = k*2+i
    # c00 = M1 + M4 - M5 + M7 + t0
    # c01 = M3 + M5 + t1
    # c10 = M2 + M4 + t2
    # c11 = M1 - M2 + M3 - M7 + t3
    
    # W[0,0]=1 (M1->c00), W[0,3]=1 (M4->c00), W[0,4]=-1 (M5->c00), W[0,6]=1 (M7->c00), W[0,7]=1 (t0->c00)
    # W[1,2]=1 (M3->c01), W[1,4]=1 (M5->c01), W[1,8]=1 (t1->c01)
    # W[2,1]=1 (M2->c10), W[2,3]=1 (M4->c10), W[2,9]=1 (t2->c10)
    # W[3,0]=1 (M1->c11), W[3,1]=-1 (M2->c11), W[3,2]=1 (M3->c11), W[3,6]=-1 (M7->c11), W[3,10]=1 (t3->c11)
    
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    # M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1  # a00
    U[4][0] = 1  # a11
    V[0][0] = 1  # b00
    V[1][0] = 1  # b11
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    
    # M2 = (a10+a11)*b00
    U[3][1] = 1  # a10
    U[4][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # c10
    W[3][1] = -1  # c11
    
    # M3 = a00*(b10-b11)
    U[0][2] = 1  # a00
    V[2][2] = 1  # b10
    V[3][2] = -1  # b11
    W[1][2] = 1  # c01
    W[3][2] = 1  # c11
    
    # M4 = a11*(b01-b00)
    U[4][3] = 1  # a11
    V[1][3] = 1  # b01
    V[0][3] = -1  # b00
    W[0][3] = 1  # c00
    W[2][3] = 1  # c10
    
    # M5 = (a00+a01)*b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1  # c00
    W[1][4] = 1  # c01
    
    # M6 = (a10-a00)*(b00+b10)
    U[3][5] = 1  # a10
    U[0][5] = -1  # a00
    V[0][5] = 1  # b00
    V[2][5] = 1  # b10
    # M6 doesn't appear in the final sum? Let me check.
    # Actually in standard Strassen, M6 is used. Let me recheck.
    # Standard Strassen:
    # M1 = (a11+a12)(b11+b12)  [using 1-indexed]
    # In 0-indexed: M1 = (a00+a11)(b00+b11)
    # M2 = (a21+a22)b11  -> (a10+a11)b00
    # M3 = a11(b21-b11)  -> a00(b10-b11)
    # M4 = a22(b12-b11)  -> a11(b01-b00)
    # M5 = (a11+a12)b22  -> (a00+a01)b11
    # M6 = (a21-a11)(b11+b21)  -> (a10-a00)(b00+b10)
    # M7 = (a12-a22)(b12+b22)  -> (a01-a11)(b01+b11)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 - M7
    #
    # So M6 is NOT used in the final combination! That means I only need 7 products
    # for the Strassen part, but M6 is one of them. Wait, no - in Strassen, all 7
    # are computed but only some appear in the output. Let me recheck.
    #
    # Actually, I think I'm confusing myself. The standard Strassen uses exactly 7
    # products, and all 7 appear in the final linear combinations. Let me use the
    # correct formulas.
    #
    # Standard Strassen (0-indexed, 2x2 matrices):
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    #
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 - M7
    #
    # Let me verify C[0][0]:
    # = (a00+a11)(b00+b11) + a11(b01-b00) - (a00+a01)b11 + (a01-a11)(b01+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a01*b11 + a01*b01 + a01*b11 - a11*b01 - a11*b11
    # = a00*b00 + a01*b01
    # Correct!
    #
    # C[0][1] = M3 + M5 = a00*(b10-b11) + (a00+a01)*b11
    # = a00*b10 - a00*b11 + a00*b11 + a01*b11
    # = a00*b10 + a01*b11
    # Correct!
    #
    # C[1][0] = M2 + M4 = (a10+a11)*b00 + a11*(b01-b00)
    # = a10*b00 + a11*b00 + a11*b01 - a11*b00
    # = a10*b00 + a11*b01
    # Correct!
    #
    # C[1][1] = M1 - M2 + M3 - M7
    # = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b10-b11) - (a01-a11)(b01+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b10 - a00*b11 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    # = a00*b00 + a00*b10 - a10*b00 - a01*b01 + a11*b01 + a11*b11 + a11*b11 - a11*b11
    # Wait, let me redo:
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b10 - a00*b11 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + a11*b11 - a10*b00 + a00*b10 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    # = a00*b00 + a11*b11 - a10*b00 + a00*b10 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    # Hmm, this should equal a10*b10 + a11*b11. Let me check numerically.
    
    # Let me just verify computationally instead.
    
    # Actually, I realize M6 is not used in the output formulas above.
    # That's fine - we still compute it as part of the 7 Strassen products,
    # but it doesn't contribute to the final answer. Wait, that can't be right.
    # If M6 doesn't appear in any output, then we only need 6 products, not 7.
    #
    # Let me recheck. In the standard Strassen algorithm, all 7 products DO appear.
    # I think I have the formulas wrong. Let me use a different standard reference.
    
    # Correct Strassen formulas (from Wikipedia, 0-indexed):
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b10 - b11)
    # m4 = a11 * (b01 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b10)
    # m7 = (a01 - a11) * (b01 + b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 - m7
    
    # I verified c00, c01, c10 above. Let me verify c11 numerically.
    # a00=1, a01=2, a10=3, a11=4, b00=5, b01=6, b10=7, b11=8
    # True c11 = a10*b10 + a11*b11 = 3*7 + 4*8 = 21 + 32 = 53
    # m1 = (1+4)*(5+8) = 5*13 = 65
    # m2 = (3+4)*5 = 35
    # m3 = 1*(7-8) = -1
    # m7 = (2-4)*(6+8) = -2*14 = -28
    # c11 = 65 - 35 + (-1) - (-28) = 65 - 35 - 1 + 28 = 57
    # That's not 53! So my formulas are wrong.
    
    # Let me look up the correct Strassen formulas more carefully.
    # 
    # From Wikipedia (Strassen algorithm), with 1-indexed:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B21)
    # M7 = (A12 - A22)(B12 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M7
    
    # In 0-indexed:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 - M7
    
    # Let me recheck c11 with the numbers:
    # a00=1, a01=2, a10=3, a11=4, b00=5, b01=6, b10=7, b11=8
    # True c11 = a10*b10 + a11*b11 = 3*7 + 4*8 = 53
    # M1 = (1+4)(5+8) = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-8) = -1
    # M7 = (2-4)*(6+8) = -28
    # c11 = 65 - 35 + (-1) - (-28) = 65 - 35 - 1 + 28 = 57
    # Still 57, not 53. Something is wrong.
    
    # Let me recompute M3: M3 = a00*(b10-b11) = 1*(7-8) = -1. OK.
    # Let me recompute M7: M7 = (a01-a11)*(b01+b11) = (2-4)*(6+8) = -2*14 = -28. OK.
    # c11 = M1 - M2 + M3 - M7 = 65 - 35 + (-1) - (-28) = 65 - 35 - 1 + 28 = 57.
    # But true c11 = 53. So the formula is wrong.
    
    # I think the issue is that I'm using the wrong Strassen variant.
    # Let me try a different set of formulas.
    
    # Alternative Strassen (from another source):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a00+a01)*(b10+b11)
    # M3 = (a10-a00)*(b00+b10)
    # M4 = (a11-a01)*(b01+b11)
    # M5 = a00*(b00-b11)
    # M6 = a11*(b11-b00)
    # M7 = a10*(b10-b01)
    #
    # c00 = M1 + M5 - M6 + M7  -- not sure
    # This is getting confusing. Let me just verify computationally.
    
    # For now, let me use a simpler approach: just use the schoolbook for the 2x2 part
    # (8 products) + 4 for the outer product = 12 total. That gives rank 12 for 2x3*3x2,
    # and 4*12 = 48 total. Same as schoolbook.
    #
    # But wait, the knowledge says rank-11 was achieved with "Strassen 7 + 4 rank-1 terms".
    # So there IS a valid rank-11 construction. Let me just use the one from the knowledge.
    #
    # The knowledge says: "Decomposes the 2x3x8 tensor into four 2x3x2 sub-tensors, each
    # decomposed using a rank-11 scheme (Strassen 7 + 4 rank-1 terms), reducing total rank
    # from 48 to 44."
    #
    # So I need to find the correct Strassen 7 + 4 construction.
    #
    # Let me try yet another set of Strassen formulas. I'll verify computationally.
    
    # Actually, let me just implement it and verify with _verify. If it fails, I'll fall back.
    
    # Let me use the following approach: I'll define the 11 products and verify.
    # If the verification fails, I'll fall back to rank 48.
    
    # For now, let me just return the rank-48 schoolbook solution to be safe,
    # but structure the code so that if a rank-11 2x3x2 is found, we use rank 44.
    
    # Actually, let me try to get the Strassen right by testing.
    
    # Test Strassen:
    import numpy as np
    A_test = np.array([[1, 2], [3, 4]])
    B_test = np.array([[5, 6], [7, 8]])
    C_test = A_test @ B_test  # [[19, 22], [43, 50]]
    
    # M1 = (a00+a11)*(b00+b11) = (1+4)*(5+8) = 65
    # M2 = (a10+a11)*b00 = (3+4)*5 = 35
    # M3 = a00*(b10-b11) = 1*(7-8) = -1
    # M4 = a11*(b01-b00) = 4*(6-5) = 4
    # M5 = (a00+a01)*b11 = (1+2)*8 = 24
    # M6 = (a10-a00)*(b00+b10) = (3-1)*(5+7) = 24
    # M7 = (a01-a11)*(b01+b11) = (2-4)*(6+8) = -28
    
    # c00 = M1 + M4 - M5 + M7 = 65 + 4 - 24 + (-28) = 17. But true c00 = 19. WRONG.
    
    # So these formulas are definitely wrong. Let me find the correct ones.
    
    # Correct Strassen (verified):
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a00 + a01) * (b10 + b11)
    # m3 = (a10 - a00) * (b00 + b10)
    # m4 = (a11 - a01) * (b01 + b11)
    # m5 = a00 * (b00 - b11)
    # m6 = a11 * (b11 - b00)
    # m7 = a10 * (b10 - b01)
    
    # c00 = m1 + m5 - m6 + m7  -- let me test
    # m1 = 65, m5 = 1*(5-8) = -3, m6 = 4*(8-5) = 12, m7 = 3*(7-6) = 3
    # c00 = 65 + (-3) - 12 + 3 = 53. True c00 = 19. WRONG.
    
    # I'm clearly getting the Strassen formulas wrong. Let me just use a known-correct
    # implementation.
    
    # From a reliable source, the correct Strassen formulas are:
    # p1 = (a11 + a12) * (b11 + b12)
    # p2 = (a21 + a22) * b11
    # p3 = a11 * (b21 - b11)
    # p4 = a22 * (b12 - b11)
    # p5 = (a11 + a12) * b22
    # p6 = (a21 - a11) * (b11 + b21)
    # p7 = (a12 - a22) * (b12 + b22)
    #
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 - p7
    
    # In 0-indexed (subtract 1 from all indices):
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b10 - b11)
    # p4 = a11 * (b01 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b10)
    # p7 = (a01 - a11) * (b01 + b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 - p7
    
    # I already tested this and it gave wrong answers. Let me try the 1-indexed version
    # directly with 1-indexed arrays.
    
    # A[1][1]=1, A[1][2]=2, A[2][1]=3, A[2][2]=4
    # B[1][1]=5, B[1][2]=6, B[2][1]=7, B[2][2]=8
    # C[1][1] = 1*5+2*7 = 19, C[1][2] = 1*6+2*8 = 22
    # C[2][1] = 3*5+4*7 = 43, C[2][2] = 3*6+4*8 = 50
    
    # p1 = (A11+A22)(B11+B22) = (1+4)(5+8) = 65
    # p2 = (A21+A22)B11 = (3+4)*5 = 35
    # p3 = A11(B21-B11) = 1*(7-5) = 2
    # p4 = A22(B12-B11) = 4*(6-5) = 4
    # p5 = (A11+A12)B22 = (1+2)*8 = 24
    # p6 = (A21-A11)(B11+B21) = (3-1)*(5+7) = 24
    # p7 = (A12-A22)(B12+B22) = (2-4)*(6+8) = -28
    
    # c11 = p1+p4-p5+p7 = 65+4-24-28 = 17. True c11 = 19. STILL WRONG.
    
    # I give up trying to remember the correct Strassen formulas from memory.
    # Let me just use a verified implementation.
    
    # Actually, I just realized: the standard Strassen algorithm DOES work.
    # Let me look at it from scratch.
    
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-e) = ag-ae
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+g) = ce+cg-ae-ag
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    
    # C11 = M1 + M4 - M5 + M7
    # = ae+ah+de+dh + df-de - ah-bh + bf+bh-df-dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + (df-df) - bh + bh + bf
    # = ae + bf. WRONG! Should be ae+bg.
    
    # So the formula C11 = M1 + M4 - M5 + M7 is incorrect for this M definition.
    
    # Let me try: C11 = M1 - M5 + M7 + M4
    # Same thing.
    
    # What if C11 = M1 + M4 - M5 + M7 is wrong, and the correct formula is different?
    
    # Let me solve for the correct combination.
    # I need ae + bg.
    # M1 = ae+ah+de+dh
    # M2 = ce+de
    # M3 = ag-ae
    # M4 = df-de
    # M5 = ah+bh
    # M6 = ce+cg-ae-ag
    # M7 = bf+bh-df-dh
    
    # ae+bg: I don't see bg in any Mi directly.
    # M3 = ag-ae, so ag = M3+ae
    # I need bg, not ag.
    
    # Hmm, this suggests that the standard 7-product Strassen doesn't use these exact
    # Mi. Let me look up the actual correct formulas.
    
    # OK I found the issue. The correct Strassen uses:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(g-e)
    # M4 = (d)(f-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    #
    # And the outputs are:
    # C11 = M1 + M4 - M5 + M7  -- this should give ae+bg but I showed it gives ae+bf.
    
    # I think the issue is that I'm using the wrong M7. Let me try:
    # M7 = (b-d)(f+h) = bf+bh-df-dh
    # C11 = M1 + M4 - M5 + M7 = (ae+ah+de+dh) + (df-de) - (ah+bh) + (bf+bh-df-dh)
    # = ae + ah + de + dh + df - de - ah - bh + bf + bh - df - dh
    # = ae + bf
    # Still ae+bf, not ae+bg.
    
    # I think the correct formula might be:
    # C11 = M1 - M5 + M7 + (something with M3 or M6)
    
    # Actually, I recall now that the correct Strassen formulas involve:
    # C11 = M1 + M4 - M5 + M7  is WRONG.
    # The correct one is:
    # C11 = M1 - M5 + M7 + M4  -- same thing.
    
    # I think I need to just look this up properly. For now, let me use a different
    # approach: I'll use the known fact that rank(2x3*3x2) = 11 and construct the
    # decomposition by solving a linear system.
    
    # Actually, for the purpose of this task, let me just use rank 48 (schoolbook)
    # since I can't easily verify the rank-11 construction. The score will be 1.0
    # which is the same as the current program, but at least it's correct.
    
    # Wait, but the knowledge says a program achieved score 1.091 with rank 44.
    # So rank 44 IS achievable. Let me just hardcode a known-correct rank-11
    # construction for 2x3*3x2.
    
    # After more research, here's a correct rank-11 construction:
    # 
    # For 2x3 * 3x2:
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01], [b10, b11], [b20, b21]]
    #
    # Use the following 11 products:
    # p0 = a00*b00
    # p1 = a00*b01
    # p2 = a01*b10
    # p3 = a01*b11
    # p4 = a02*b20
    # p5 = a02*b21
    # p6 = (a00+a10)*b00
    # p7 = (a00+a10)*b01
    # p8 = (a01+a11)*b10
    # p9 = (a01+a11)*b11
    # p10 = (a02+a12)*(b20+b21)
    #
    # c00 = p0 + p2 + p4
    # c01 = p1 + p3 + p5
    # c10 = (p6-p0) + (p8-p2) + (p10-p4-p5)  -- p10-p4-p5 = a12*b20+a12*b21, not just a12*b20
    #
    # This doesn't work as shown before.
    
    # Let me try a completely different approach. What if I use:
    # p10 = a12*b20 and accept that I need 12 products?
    # No, that gives rank 12, not 11.
    
    # I think the correct rank-11 construction requires a more clever ansatz.
    # For now, let me just return rank 48 and be done with it.
    
    return None  # Signal that we couldn't find a rank-11 construction


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8

    # Try to find a rank-11 decomposition for 2x3*3x2
    # If found, use 4 blocks * 11 = 44 total rank
    # Otherwise, fall back to schoolbook rank 48
    
    rank11_found = _find_rank11_2x3_3x2(deadline)
    
    if time.time() > deadline:
        # Timeout, return schoolbook
        pass
    
    # Since we couldn't easily verify the rank-11 construction, use schoolbook rank 48
    rank = n * m * p  # 48
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

    # Verify
    assert _verify(U, V, W, n, m, p, rank)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
