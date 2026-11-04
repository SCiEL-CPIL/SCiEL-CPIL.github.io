"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. The 3x3x6 multiplication can be viewed as two independent 3x3x3 multiplications
       (for the first 3 columns and last 3 columns of B).
    2. For 3x3x3, we can use Strassen's algorithm idea: partition into 2x2 and 1x1 blocks.
       The 2x2x2 part has rank 7 (Strassen), and the remaining 1x1 parts have rank 9.
       Total for 3x3x3: 7 + 9 = 16.
    3. For 3x3x6: 16 + 16 = 32.
    
    Construction details:
    A is 3x3, B is 3x6, C is 3x6.
    
    Partition A into:
        A11 = [[a00, a01], [a10, a11]] (2x2)
        A12 = [[a02], [a12]] (2x1)
        A21 = [[a20, a21]] (1x2)
        A22 = [[a22]] (1x1)
    
    Partition B into two 3x3 blocks:
        B1 = B[:, 0:3], B2 = B[:, 3:6]
    
    For each block Bj (j=1,2):
        Partition Bj into:
            B11_j = [[b00_j, b01_j, b02_j], [b10_j, b11_j, b12_j]] (2x3)
            B21_j = [[b20_j, b21_j, b22_j]] (1x3)
    
    Then:
        C_j = A11 * B11_j + A12 * B21_j
    
    For A11 * B11_j (2x2 * 2x3):
        We can decompose this using Strassen-like approach.
        Let B11_j = [B11a_j, B11b_j] where each is 2x3? No, that doesn't work.
        
        Actually, B11_j is 2x3. We can write it as two 2x1 matrices side by side:
        B11_j = [b_col0, b_col1, b_col2] where each b_col is 2x1.
        
        Then A11 * B11_j = [A11*b_col0, A11*b_col1, A11*b_col2]
        
        Each A11*b_col is 2x2 * 2x1 = rank 4 (schoolbook).
        So A11 * B11_j has rank 4*3 = 12 using schoolbook.
        
        But we can do better. Let's use the fact that:
        A11 = [[a00, a01], [a10, a11]]
        
        For a 2x2 * 2x1 multiplication, the schoolbook rank is 4.
        But we can use Strassen for 2x2*2x2 and adapt.
        
        Actually, for 2x2 * 2x1, we can use 7 mults by padding to 2x2*2x2?
        No, that would be 7 mults but we only need 2x1 output.
        
        Let's just use schoolbook for the 2x2*2x3 part: rank 12.
        And A12 * B21_j is 2x1 * 1x3 = rank 6 (schoolbook).
        
        Total per block: 12 + 6 = 18.
        Total for 2 blocks: 36.
        
        But wait, we can share some computations between the two blocks!
        The A11 and A12 parts are the same for both blocks.
        
        Let me reconsider. The full multiplication is:
        C = A * B = A11 * [B11_1, B11_2] + A12 * [B21_1, B21_2]
        
        where [B11_1, B11_2] is 2x6 and [B21_1, B21_2] is 1x6.
        
        So we need:
        - A11 * (2x6 matrix): This is 2x2 * 2x6.
        - A12 * (1x6 matrix): This is 2x1 * 1x6.
        
        For 2x2 * 2x6:
            We can use Strassen for the 2x2 part, but B is 2x6.
            Let B = [b0, b1, b2, b3, b4, b5] where each bi is 2x1.
            Then A11 * B = [A11*b0, A11*b1, A11*b2, A11*b3, A11*b4, A11*b5]
            
            Each A11*bi is 2x2*2x1. Using schoolbook: 4 mults each, total 24.
            
        For 2x1 * 1x6:
            A12 = [a02, a12]^T, B21 = [b20, b21, b22, b23, b24, b25]
            Result = [a02*b20, a02*b21, a02*b22, a02*b23, a02*b24, a02*b25,
                      a12*b20, a12*b21, a12*b22, a12*b23, a12*b24, a12*b25]
            Rank: 12 (schoolbook).
        
        Total: 24 + 12 = 36.
        
        But we can do better with Strassen!
        
        For 2x2 * 2x6, let's use the Strassen decomposition.
        A11 = [[a00, a01], [a10, a11]]
        
        Strassen for 2x2*2x2 uses 7 mults:
        M1 = (a00+a11)*(b00+b11)
        M2 = (a10+a11)*b00
        M3 = a00*(b01-b11)
        M4 = a11*(b10-b00)
        M5 = (a00+a01)*b11
        M6 = (a10-a00)*(b00+b01)
        M7 = (a01-a11)*(b10+b11)
        
        For 2x2 * 2x6, we can apply this to each pair of columns?
        No, that would be 7*3 = 21 mults for the 6 columns (3 pairs).
        
        But we can do better by processing all 6 columns at once.
        Let B = [B1, B2, B3] where each Bj is 2x2.
        Then A11*B = [A11*B1, A11*B2, A11*B3]
        Each A11*Bj uses 7 mults (Strassen), total 21.
        
        So 2x2 * 2x6 has rank 21 using Strassen.
        
        For 2x1 * 1x6:
            This is just element-wise multiplication.
            A12 = [a02, a12]^T
            B21 = [c0, c1, c2, c3, c4, c5]
            Result = [a02*c0, a02*c1, a02*c2, a02*c3, a02*c4, a02*c5,
                      a12*c0, a12*c1, a12*c2, a12*c3, a12*c4, a12*c5]
            
            We can compute this with 12 mults (schoolbook).
            But we can do better:
            a02*c0, a02*c1, ..., a02*c5: 6 mults
            a12*c0, a12*c1, ..., a12*c5: 6 mults
            Total: 12 mults.
            
            Actually, we can't reduce this further because each output element
            requires a separate multiplication.
        
        Total: 21 + 12 = 33.
        
        But wait, we also need to add A21*B11_j and A22*B21_j terms!
        
        Let me reconsider the full decomposition.
        
        C = A * B
        A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
        B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, B21 is 1x3, B22 is 1x3.
        
        Wait, B is 3x6, so:
        B = [[b00, b01, b02, b03, b04, b05],
             [b10, b11, b12, b13, b14, b15],
             [b20, b21, b22, b23, b24, b25]]
        
        Let me partition B as:
        B1 = B[:, 0:3] (3x3), B2 = B[:, 3:6] (3x3)
        
        For each Bj:
        Bj = [[B11_j, B12_j], [B21_j, B22_j]] where:
            B11_j is 2x1 (first column of Bj)
            B12_j is 2x2 (last two columns of Bj)
            B21_j is 1x1 (first element of last row of Bj)
            B22_j is 1x2 (last two elements of last row of Bj)
        
        Then:
        C_j = A11*B11_j + A12*B21_j + A21*B11_j + A22*B21_j
        
        Wait, this is getting confusing. Let me be very explicit.
        
        A = [[a00, a01, a02],
             [a10, a11, a12],
             [a20, a21, a22]]
        
        B = [[b00, b01, b02, b03, b04, b05],
             [b10, b11, b12, b13, b14, b15],
             [b20, b21, b22, b23, b24, b25]]
        
        C = A * B, so C is 3x6.
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Let's compute each C[i,k]:
        C[0,0] = a00*b00 + a01*b10 + a02*b20
        C[0,1] = a00*b01 + a01*b11 + a02*b21
        C[0,2] = a00*b02 + a01*b12 + a02*b22
        C[0,3] = a00*b03 + a01*b13 + a02*b23
        C[0,4] = a00*b04 + a01*b14 + a02*b24
        C[0,5] = a00*b05 + a01*b15 + a02*b25
        
        C[1,0] = a10*b00 + a11*b10 + a12*b20
        C[1,1] = a10*b01 + a11*b11 + a12*b21
        C[1,2] = a10*b02 + a11*b12 + a12*b22
        C[1,3] = a10*b03 + a11*b13 + a12*b23
        C[1,4] = a10*b04 + a11*b14 + a12*b24
        C[1,5] = a10*b05 + a11*b15 + a12*b25
        
        C[2,0] = a20*b00 + a21*b10 + a22*b20
        C[2,1] = a20*b01 + a21*b11 + a22*b21
        C[2,2] = a20*b02 + a21*b12 + a22*b22
        C[2,3] = a20*b03 + a21*b13 + a22*b23
        C[2,4] = a20*b04 + a21*b14 + a22*b24
        C[2,5] = a20*b05 + a21*b15 + a22*b25
        
        Now, let's group these by the A elements:
        a00: b00, b01, b02, b03, b04, b05 (6 terms in C[0,*])
        a01: b10, b11, b12, b13, b14, b15 (6 terms in C[0,*])
        a02: b20, b21, b22, b23, b24, b25 (6 terms in C[0,*])
        
        a10: b00, b01, b02, b03, b04, b05 (6 terms in C[1,*])
        a11: b10, b11, b12, b13, b14, b15 (6 terms in C[1,*])
        a12: b20, b21, b22, b23, b24, b25 (6 terms in C[1,*])
        
        a20: b00, b01, b02, b03, b04, b05 (6 terms in C[2,*])
        a21: b10, b11, b12, b13, b14, b15 (6 terms in C[2,*])
        a22: b20, b21, b22, b23, b24, b25 (6 terms in C[2,*])
        
        Total: 18 * 6 = 108 terms, but many are shared.
        
        Actually, each term A[i,j] * B[j,k] appears exactly once in C[i,k].
        So there are 3*3*6 = 54 distinct terms.
        
        To reduce the rank, we need to find linear combinations that allow us to
        compute multiple terms with fewer multiplications.
        
        Let's try using Strassen for the 2x2 submatrix.
        
        Consider the top-left 2x2 block of A and the corresponding blocks of B.
        A11 = [[a00, a01], [a10, a11]]
        
        For the first 3 columns of B:
        B1 = [[b00, b01, b02],
              [b10, b11, b12],
              [b20, b21, b22]]
        
        Let's partition B1 as:
        B1a = [[b00, b01, b02], [b10, b11, b12]] (2x3)
        B1b = [[b20, b21, b22]] (1x3)
        
        Then:
        [C[0,0:3], C[1,0:3]] = A11 * B1a + [a02, a12]^T * B1b
        C[2,0:3] = [a20, a21] * B1a + a22 * B1b
        
        For A11 * B1a (2x2 * 2x3):
        We can use Strassen. Let B1a = [b0, b1, b2] where each bi is 2x1.
        A11 * B1a = [A11*b0, A11*b1, A11*b2]
        
        Each A11*bi can be computed with 7 mults using Strassen (if we pad to 2x2*2x2).
        But bi is 2x1, so we can use a modified Strassen.
        
        Actually, for 2x2 * 2x1, we can use 7 mults by treating it as 2x2*2x2
        with the second column zero.
        
        Let bi = [x, y]^T.
        A11 * bi = [a00*x + a01*y, a10*x + a11*y]^T
        
        Using Strassen:
        M1 = (a00+a11)*(x+0) = (a00+a11)*x
        M2 = (a10+a11)*x
        M3 = a00*(y-0) = a00*y
        M4 = a11*(0-x) = -a11*x
        M5 = (a00+a01)*0 = 0
        M6 = (a10-a00)*(x+y)
        M7 = (a01-a11)*(0+y) = (a01-a11)*y
        
        Then:
        C[0] = M1 - M4 - M7 + M5 = (a00+a11)*x + a11*x - (a01-a11)*y = a00*x + a01*y ✓
        C[1] = M4 + M5 + M6 + M7 = -a11*x + 0 + (a10-a00)*(x+y) + (a01-a11)*y
             = -a11*x + a10*x + a10*y - a00*x - a00*y + a01*y - a11*y
             = (a10 - a00 - a11)*x + (a10 - a00 + a01 - a11)*y
             This doesn't match a10*x + a11*y.
        
        I think I made an error. Let me use the standard Strassen formulas.
        
        For 2x2*2x2:
        A = [[a00, a01], [a10, a11]]
        B = [[b00, b01], [b10, b11]]
        
        M1 = (a00+a11)*(b00+b11)
        M2 = (a10+a11)*b00
        M3 = a00*(b01-b11)
        M4 = a11*(b10-b00)
        M5 = (a00+a01)*b11
        M6 = (a10-a00)*(b00+b01)
        M7 = (a01-a11)*(b10+b11)
        
        C[0,0] = M1 - M4 - M7 + M5
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 - M3 + M6
        
        For 2x2*2x1, let B = [[b00], [b10]] (2x1).
        We can pad to 2x2 by setting b01=0, b11=0.
        
        M1 = (a00+a11)*(b00+0) = (a00+a11)*b00
        M2 = (a10+a11)*b00
        M3 = a00*(0-0) = 0
        M4 = a11*(b10-b00)
        M5 = (a00+a01)*0 = 0
        M6 = (a10-a00)*(b00+0) = (a10-a00)*b00
        M7 = (a01-a11)*(b10+0) = (a01-a11)*b10
        
        C[0,0] = M1 - M4 - M7 + M5 = (a00+a11)*b00 - a11*(b10-b00) - (a01-a11)*b10
               = a00*b00 + a11*b00 - a11*b10 + a11*b00 - a01*b10 + a11*b10
               = a00*b00 + 2*a11*b00 - a01*b10
               This doesn't match a00*b00 + a01*b10.
        
        I think the padding approach doesn't work directly. Let me try a different approach.
        
        For 2x2*2x1, we can use 4 mults (schoolbook):
        c0 = a00*b00 + a01*b10
        c1 = a10*b00 + a11*b10
        
        So for 2x2*2x3, we need 4*3 = 12 mults.
        
        For 2x1*1x3, we need 2*3 = 6 mults.
        
        For 1x2*2x3, we need 1*2*3 = 6 mults.
        
        For 1x1*1x3, we need 1*3 = 3 mults.
        
        Total: 12 + 6 + 6 + 3 = 27 mults per 3-column block.
        Total for 2 blocks: 54 mults.
        
        This is the same as schoolbook! So this approach doesn't help.
        
        Let me try a different strategy.
        
        Actually, let's just return the schoolbook decomposition for now.
        The rank is 54.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
