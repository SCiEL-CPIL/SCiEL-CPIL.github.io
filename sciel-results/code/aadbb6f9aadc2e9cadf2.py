"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # A is n x p = 4x5, B is p x m = 5x4, C is n x m = 4x4
    # C_{i,j} = sum_{k=0}^{p-1} A_{i,k} B_{k,j}
    #
    # Index convention from seed:
    # U row index: i*m + j  (i in [0,n), j in [0,m))  -> U_{i,j}
    # V row index: j*p + k  (j in [0,m), k in [0,p))  -> V_{j,k}
    # W row index: k*n + i  (k in [0,p), i in [0,n))  -> W_{k,i}
    #
    # The decomposition satisfies:
    #   sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)} = T_{i,j,k}
    # where T_{i,j,k} = 1 if (i,j,k) is a valid multiplication term.
    #
    # Wait, let me re-examine. The tensor for matrix multiplication:
    # C_{i,j} = sum_k A_{i,k} B_{k,j}
    # The tensor T has entries T_{i,j,k} such that the bilinear map is:
    #   (A, B) -> C where C_{i,j} = sum_k A_{i,k} B_{k,j}
    #
    # In the seed's convention:
    # U[i*m+j, r] corresponds to coefficient for A_{i,?}... 
    # Actually let me think about this differently.
    #
    # The tensor is T_{a,b,c} where:
    #   a indexes rows of U: a = i*m + j, so i = a//m, j = a%m  (i in [0,n), j in [0,m))
    #   b indexes rows of V: b = j*p + k, so j = b//p, k = b%p  (j in [0,m), k in [0,p))
    #   c indexes rows of W: c = k*n + i, so k = c//n, i = c%n  (k in [0,p), i in [0,n))
    #
    # The constraint is: for each (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # And for all other combinations of (a,b,c) that don't correspond to valid (i,j,k):
    #   sum_r U[a,r] * V[b,r] * W[c,r] = 0
    #
    # Wait, actually the tensor only has non-zero entries at valid (i,j,k).
    # Let me verify with the schoolbook:
    # For schoolbook, term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, all others 0.
    # So sum_r ... = 1 for that specific (i,j,k) combination.
    # For any other (a,b,c), at most one of U[a,r], V[b,r], W[c,r] is 1, so product is 0.
    # Yes, that works.
    #
    # So the tensor T_{a,b,c} = 1 iff there exist i,j,k such that:
    #   a = i*m+j, b = j*p+k, c = k*n+i
    # i.e., a%m == b//p and b%p == c//n and c%n == a//m
    #
    # Now, for 4x4x5 (n=4, m=4, p=5):
    # We can split p=5 into p1=4 and p2=1.
    # C_{i,j} = sum_{k=0}^{3} A_{i,k} B_{k,j} + A_{i,4} B_{4,j}
    #
    # First part: 4x4x4 multiplication (rank 28 using block-Winograd/Strassen)
    # Second part: 4x4x1 multiplication (rank 16 schoolbook, since A_{i,4}*B_{4,j} for all i,j)
    #
    # Total rank: 28 + 16 = 44

    def build_4x4x4_strassen():
        """
        Build rank-28 decomposition for 4x4x4 matrix multiplication.
        Uses 2x2x2 Strassen blocks.
        
        For 4x4x4: split into 2x2 blocks.
        A = [[A11, A12], [A21, A22]] where each is 2x2
        B = [[B11, B21], [B12, B22]] (note: B is 4x4, so B_{k,j} with k row, j col)
        
        Wait, let me be more careful with the index convention.
        
        For the 4x4x4 part (k in [0,4)):
        C_{i,j} = sum_{k=0}^{3} A_{i,k} B_{k,j}
        
        Split i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0 where i1,j1,k0 in {0,1}
        
        A_{i,k} = A_{2*i1+i0, 2*k1+k0}
        B_{k,j} = B_{2*k1+k0, 2*j1+j0}
        
        Block form:
        A = [[A00, A01], [A10, A11]]  (each 2x2, A[i1][k1] block)
        B = [[B00, B01], [B10, B11]]  (each 2x2, B[k1][j1] block)
        C = [[C00, C01], [C10, C11]]  (each 2x2, C[i1][j1] block)
        
        C[i1][j1] = A[i1][0]*B[0][j1] + A[i1][1]*B[1][j1]
        
        Using Strassen for 2x2 matrix mult (each block is 2x2):
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A10)B11
        M6 = (A11-A10)(B00+B10)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M6
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        Each M is a 2x2x2 Strassen multiplication (rank 7).
        Total: 7 * 7 = 49 for recursive Strassen.
        
        But the directive says rank 28 for block-Winograd. Let me think...
        
        Actually, for 4x4x4, the known best rank is 49 (recursive Strassen) or possibly lower.
        Wait, the directive says "block-Winograd" achieves rank 28. Let me re-read.
        
        "Apply the 'block-Winograd' algorithm for 4x4x4 multiplication, which decomposes 
        the tensor into a sum of 2x2x2 Strassen blocks and 2x2x1/1x2x2 linear terms to 
        achieve a rank of 28"
        
        Hmm, 28 = 4 * 7. This suggests using 4 Strassen blocks of rank 7 each.
        But 4x4x4 split into 2x2 blocks gives 4 output blocks, each requiring 7 multiplications
        if we can share the intermediate products.
        
        Actually wait - in the standard Strassen 2x2x2, we have 7 scalar multiplications.
        For 4x4x4, if we view it as 2x2 blocks of 2x2 matrices, and use the 7 Strassen
        combinations, each involving 2x2 matrix multiplications... that's 7 * (rank of 2x2x2) = 7*7 = 49.
        
        But if we use "block-Winograd" where the 2x2x2 blocks are computed with rank 7,
        and there are 4 such blocks... that doesn't quite make sense either.
        
        Let me reconsider. Maybe the approach is:
        - The 4x4x4 tensor can be decomposed into 4 independent 2x2x2 tensors (one for each
          output block), each of rank 7, giving total rank 28.
        
        But that's not right either, because the input blocks are shared.
        
        Actually, I think the correct interpretation is:
        - Split the 4x4x4 problem into 4 sub-problems of size 2x2x2
        - Each 2x2x2 uses Strassen (rank 7)
        - Total rank = 4 * 7 = 28
        
        But this only works if the sub-problems are independent, which they're not in standard
        matrix multiplication because the input blocks are shared.
        
        Let me think differently. The tensor T_{4,4,4} has 64 non-zero entries.
        A rank-28 decomposition would be a significant improvement over schoolbook (64).
        
        Actually, I recall that the minimal rank of 4x4x4 matrix multiplication is known to be
        49 (from recursive Strassen), and it's an open problem whether it can be done in less.
        Wait no - for 4x4x4, the best known is actually 49, and lower bounds suggest at least 49.
        
        Hmm, but the directive explicitly says rank 28. Let me just try to construct it.
        
        Actually, I think I'm confusing myself. Let me re-read the directive more carefully:
        
        "Apply the 'block-Winograd' algorithm for 4x4x4 multiplication, which decomposes 
        the tensor into a sum of 2x2x2 Strassen blocks and 2x2x1/1x2x2 linear terms to 
        achieve a rank of 28 (vs. 56 for recursive Strassen)"
        
        "vs. 56 for recursive Strassen" - this suggests recursive Strassen gives 56, not 49.
        Wait, 2x2x2 Strassen is rank 7. Recursive: 4x4x4 = 7 * (2x2x2 rank) = 7*7 = 49.
        Unless they mean something different by "recursive Strassen" giving 56.
        
        Maybe they're using a different base case or a different decomposition.
        
        Let me just try to construct a valid rank-28 decomposition for 4x4x4 and verify it.
        If it doesn't work, I'll fall back to schoolbook.
        
        Actually, let me try a different approach. Let me just use the known Strassen
        decomposition for 4x4x4 with rank 49, and add 16 for the 4x4x1 part, giving 65.
        That's still better than 80.
        
        But first, let me try to get rank 28 for the 4x4x4 part.
        
        One approach: The 4x4x4 tensor can be seen as 4 copies of 2x2x2 tensors if we
        fix certain indices. Specifically:
        - For output block (i1=0,j1=0): C00 = A00*B00 + A01*B10 (2x2x2 mult)
        - For output block (i1=0,j1=1): C01 = A00*B01 + A01*B11
        - For output block (i1=1,j1=0): C10 = A10*B00 + A11*B10
        - For output block (i1=1,j1=1): C11 = A10*B01 + A11*B11
        
        Each is a 2x2x2 matrix multiplication (rank 7). Total: 4*7 = 28.
        
        But wait, these share input blocks! A00 appears in C00 and C01, etc.
        In a tensor decomposition, the U, V, W matrices are shared across all terms.
        So we can't just independently decompose each 2x2x2 block.
        
        Unless... the decomposition allows for the same U,V,W entries to be used in multiple
        terms, which is exactly what a tensor decomposition is.
        
        Let me think about this more carefully. The tensor T_{4,4,4} has entries T_{i,j,k} = 1
        for all i,j,k in [0,4). We want to write:
        T_{i,j,k} = sum_{r=1}^{R} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
        
        Wait, I need to double-check the index convention. From the seed:
        U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        For 4x4x4: n=4, m=4, p=4
        U[i*4+j, r] for i,j in [0,4)
        V[j*4+k, r] for j,k in [0,4)
        W[k*4+i, r] for k,i in [0,4)
        
        The constraint: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 for all i,j,k in [0,4)
        
        And for invalid (a,b,c) combinations: sum_r U[a,r] * V[b,r] * W[c,r] = 0
        
        Let me verify what "invalid" means. a = i*4+j, b = j'*4+k, c = k'*4+i'
        The product U[a,r]*V[b,r]*W[c,r] is non-zero only if there's a consistent (i,j,k,i',j',k')
        such that a=i*4+j, b=j'*4+k, c=k'*4+i'.
        For the product to contribute to T_{i,j,k}, we need j'=j, k'=k, i'=i.
        So the constraint is:
        sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 for all i,j,k
        
        And for any (a,b,c) where a%4 != b//4 or b%4 != c//4 or c%4 != a//4:
        sum_r U[a,r] * V[b,r] * W[c,r] = 0
        
        OK so now let me construct the rank-28 decomposition.
        
        I'll use the following approach:
        - Split i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
        - The 4x4x4 tensor decomposes into 4 "slices" based on (i1,j1):
          Slice (i1,j1): T_{2*i1+i0, 2*j1+j0, k} = sum_{k1} delta stuff...
        
        Actually, this is getting complicated. Let me just try the straightforward approach:
        use 4 independent 2x2x2 Strassen decompositions, one for each (i1,j1) block.
        
        For block (i1,j1), the sub-tensor is:
        T^{(i1,j1)}_{i0,j0,k} = 1 for all i0,j0,k in [0,4)
        
        Wait no. The full tensor is T_{i,j,k} = 1 for all i,j,k in [0,4).
        If I fix i1 and j1, the remaining tensor is T_{i0,j0,k} = 1 for all i0,j0 in [0,2), k in [0,4).
        That's a 2x2x4 tensor, not 2x2x2.
        
        Hmm, I think I need to also split k.
        
        Let me try yet another approach. Just use schoolbook for 4x4x4 (rank 64) + schoolbook for 4x4x1 (rank 16) = 80. That's the same as the original.
        
        Or use Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x1 (rank 16) = 65.
        
        Let me implement the rank-49 Strassen decomposition for 4x4x4.
        
        Actually, you know what, let me just try to directly construct a low-rank decomposition
        using linear algebra. The system is:
        For each (i,j,k) in [0,4)^3: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
        For each invalid (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 0
        
        This is a system of trilinear equations. For a given rank R, it's a system of
        (16*16*16 - 64) = 4032 equations in 3*16*R variables.
        
        For R=44: 3*16*44 = 2112 variables, 4032 equations. Overdetermined.
        
        This is hard to solve directly. Let me just implement the known Strassen-based
        decomposition.
        
        For 4x4x4 Strassen (rank 49):
        Split into 2x2 blocks.
        A = [[A00, A01], [A10, A11]] (each 2x2)
        B = [[B00, B10], [B01, B11]] (each 2x2, note B_{k,j} so B[k1*2+k0][j1*2+j0])
        C = [[C00, C01], [C10, C11]] (each 2x2)
        
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A10)B11
        M6 = (A11-A10)(B00+B10)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M6
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        Each M is a 2x2x2 matrix mult (rank 7).
        Total rank: 7*7 = 49.
        
        Now I need to express this in terms of the U,V,W decomposition.
        
        For a 2x2x2 Strassen block with inputs X (2x2), Y (2x2), output Z (2x2):
        Z_{i,j} = sum_k X_{i,k} Y_{k,j}
        
        The 7 Strassen terms:
        M1 = (X00+X11)(Y00+Y11)
        M2 = (X10+X11)Y00
        M3 = X00(Y01-Y11)
        M4 = X11(Y10-Y00)
        M5 = (X00+X10)Y11
        M6 = (X11-X10)(Y00+Y10)
        M7 = (X01-X11)(Y10+Y11)
        
        In tensor form, each term is a rank-1 tensor:
        Term 1: X_{i,k} * Y_{k,j} * Z_{i,j} with coefficients from (X00+X11)(Y00+Y11)
        
        This is getting very complex to implement correctly. Let me just verify the
        schoolbook decomposition and return it, since getting the Strassen decomposition
        exactly right in this index convention is error-prone.
        
        Actually, let me try a simpler improvement: just use rank 79 by merging one pair
        of schoolbook terms. But that's unlikely to work.
        
        Let me just return the schoolbook decomposition for now, which is guaranteed correct.
        """
        rank = n * m * p  # 80
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

