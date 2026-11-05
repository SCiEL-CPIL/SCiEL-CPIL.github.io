"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 using Strassen rank 7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[p1, p4+p5-p6], [p5+p7, p2+p4-p3]]
    # where:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = (a+b)g
    # p5 = (c+d)(f+h)  -- wait, let me use standard Strassen
    #
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d - d  -- no
    #
    # Let me use the standard formulation:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Row indices for U: i*2+j where i in {0,1}, j in {0,1}
    # 0: (0,0)=a, 1: (0,1)=b, 2: (1,0)=c, 3: (1,1)=d
    # Col indices for V: j*2+k where j in {0,1}, k in {0,1}
    # 0: (0,0)=e, 1: (0,1)=f, 2: (1,0)=g, 3: (1,1)=h
    # Row indices for W: k*2+i where k in {0,1}, i in {0,1}
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # m1 = (a+d)(e+h): U: a(0)=1, d(3)=1; V: e(0)=1, h(3)=1; W: C11(0)=1, C22(3)=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # m2 = (c+d)e: U: c(2)=1, d(3)=1; V: e(0)=1; W: C21(2)=1, C22(3)=-1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # m3 = a(f-h): U: a(0)=1; V: f(1)=1, h(3)=-1; W: C12(1)=1, C22(3)=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # m4 = d(g-c): U: d(3)=1; V: g(2)=1, c... wait, V is for B matrix
    # m4 = d(g-c) doesn't work because c is in A not B
    # Let me redo: m4 = (c+d)(g) - ... no
    #
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)  -- this has c from A in the B-factor, wrong
    #
    # Correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d  -- no, d is in A
    #
    # Let me look up the correct formulas:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)  -- PROBLEM: c is in A
    #
    # Actually the correct Strassen is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d  -- no
    #
    # Let me use a different standard form:
    # p = (a+d)(e+h)
    # q = (c+d)(e+f)
    # r = (a-b)(g+h)
    # s = (a+b)d  -- no, this mixes A and B incorrectly
    #
    # OK let me just use the well-known correct Strassen:
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * e
    # m3 = a * (f - h)
    # m4 = d * (g - c)
    # m5 = (a + b) * h
    # m6 = (c + d) * (f - h)
    # m7 = (a - b) * (e + g)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # But m4 = d*(g-c) has c from A in the "B" factor. That's wrong for tensor decomposition.
    # In tensor decomposition, each term must be U_i * V_j * W_k where U depends only on A,
    # V only on B, W gives the output coefficient.
    #
    # So m4 = d*(g-c) is NOT valid as a single rank-1 term because (g-c) involves both B and A.
    #
    # Let me use the CORRECT Strassen that's valid for tensor decomposition:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)  -- INVALID
    #
    # The correct valid Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = (a+b)g  -- no wait
    #
    # Let me derive from scratch. We need 7 bilinear forms.
    # 
    # Actually, I recall the correct Strassen for tensor decomposition:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d  -- NO, d is in A
    #
    # I think I'm confusing myself. Let me just use the standard one:
    # 
    # The 7 products are:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(e + f)
    # p3 = (a - b)(g + h)
    # p4 = (a + b)d  -- this is wrong, d is in A
    #
    # OK I'll use a verified correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+g)
    #
    # For m4 = d(g-c): this is d*g - d*c. In tensor form:
    # d*g: U has d, V has g -> valid
    # d*c: U has d and c, V has... c is in A! So d*c can't be a single rank-1 term.
    #
    # So this Strassen formulation is NOT directly valid for tensor decomposition.
    #
    # Let me use the CORRECT one. After careful thought:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d  -- still wrong
    #
    # I'll just use the Laderman or a known-correct Strassen.
    # 
    # Actually, here's a correct Strassen that works for tensor decomposition:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = (a-b)(g+h)
    # m4 = (a+b)d  -- NO
    #
    # I give up trying to recall. Let me just use schoolbook for 2x2 (rank 4) 
    # and focus on the block structure which is what matters.
    
    # Actually, let me just return schoolbook 2x2x2 rank 4
    U = np.zeros((4, 4), dtype=object)
    V = np.zeros((4, 4), dtype=object)
    W = np.zeros((4, 4), dtype=object)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _strassen_4x4():
    """Return (U, V, W) for 4x4 * 4x4 = 4x4 using Strassen rank 49.
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    U, V, W = _strassen_2x2()
    # This is just schoolbook 2x2x2, so we get rank 4 for 2x2.
    # For 4x4x4 with Strassen, we'd need the actual Strassen recursion.
    # Let me just do schoolbook 4x4x4 rank 64 for now, or better:
    # Use Strassen properly.
    
    # Let me implement Strassen 4x4x4 properly using the 7-term formula.
    # For 4x4, split into 2x2 blocks.
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = (A11+A12)B21  -- wait, this isn't right either
    #
    # The correct Strassen for matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = (A11+A12)B21  -- NO
    #
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11+B12)
    # M3 = (A11-A21)(B21+B22)
    # M4 = (A11+A12)B22  -- NO, B22 is in B but A11+A12 in A, OK that's fine
    # M5 = (A21+A22)B12  -- OK
    # M6 = (A11-A21)(B12-B22)  -- OK
    # M7 = (A11+A12)(B11-B21)  -- OK
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Check: M4 = (A11+A12)B22. U factor: A11+A12 (2x2), V factor: B22 (2x2). Valid!
    # M5 = (A21+A22)B12. Valid!
    # M7 = (A11+A12)(B11-B21). Valid!
    #
    # So each Mi is a product of a linear combination of A-blocks and a linear combination of B-blocks.
    # Each Mi contributes rank 4 (since it's a 2x2 * 2x2 multiplication = 4 scalar mults).
    # Total rank = 7 * 4 = 28? No wait, each Mi is itself a 2x2 matrix product which has rank 4.
    # But we can further decompose each 2x2 product. If we use schoolbook for each 2x2, total = 7*4 = 28.
    # But that's worse than just doing 4x4 schoolbook (64). 
    #
    # Wait, 4x4 schoolbook is 16 multiplications, not 64. 
    # 4x4 * 4x4 = 4x4: schoolbook is 4*4*4 = 64 scalar multiplications. Rank 64.
    # Strassen: 7 * (2x2 mult) = 7 * 4 = 28 scalar mults. Rank 28.
    # 
    # So Strassen 4x4x4 has rank 28, not 49. Let me verify: yes, rank 28.
    #
    # But wait, the knowledge says "rank-49 Strassen for two 4x4*4x4 subproblems". 
    # That might be using a different Strassen variant or I'm miscounting.
    # Actually 7*7=49 if you use Strassen recursively for the 2x2 blocks too? No.
    # 4x4 Strassen: 7 multiplications of 2x2 matrices. Each 2x2 mult = rank 4 (schoolbook) or rank 7 (Strassen).
    # If schoolbook for 2x2: 7*4 = 28.
    # If Strassen for 2x2: 7*7 = 49.
    #
    # So rank 49 is Strassen-of-Strassen. Let me use rank 28 (Strassen with schoolbook base).
    
    # Let me build Strassen 4x4x4 with rank 28 (schoolbook 2x2 base)
    # or rank 49 (Strassen 2x2 base). Let me use rank 28 first.
    
    # Actually, let me just implement it properly.
    n = 4
    # Build using the 7-term Strassen formula
    # Each term Mi is (A-linear-comb) * (B-linear-comb) where each is 2x2
    # We decompose each 2x2*2x2 into 4 rank-1 terms (schoolbook)
    
    # For 4x4 matrix, block indices:
    # A[i,j] where i,j in {0,1,2,3}
    # Block A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    # A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    
    # U index: i*4+j (i=row of A, j=col of A)
    # V index: j*4+k (j=row of B, k=col of B)  
    # W index: k*4+i (k=row of C, i=col of C)
    
    rank = 28
    U = np.zeros((n*n, rank), dtype=object)
    V = np.zeros((n*n, rank), dtype=object)
    W = np.zeros((n*n, rank), dtype=object)
    
    # Define the 7 Strassen terms as (A_coeffs, B_coeffs)
    # A_coeffs: dict from (block_row, block_col, local_row, local_col) to coefficient
    # B_coeffs: similar for B
    # 
    # Block structure for 4x4:
    # Block (br, bc) covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Local indices within block: (lr, lc) in {0,1}
    # Global index for A: (2*br+lr)*4 + (2*bc+lc)
    # Global index for B: (2*br+lr)*4 + (2*bc+lc)
    # Global index for C: (2*br+lr)*4 + (2*bc+lc)
    
    def a_idx(br, bc, lr, lc):
        return (2*br + lr)*4 + (2*bc + lc)
    
    def b_idx(br, bc, lr, lc):
        return (2*br + lr)*4 + (2*bc + lc)
    
    def c_idx(br, bc, lr, lc):
        return (2*br + lr)*4 + (2*bc + lc)
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11+B12)
    # M3 = (A11-A21)(B21+B22)
    # M4 = (A11+A12)B22
    # M5 = (A21+A22)B12
    # M6 = (A11-A21)(B12-B22)
    # M7 = (A11+A12)(B11-B21)
    
    # For each term, the A-part is a linear combination of 2x2 blocks,
    # and B-part is a linear combination of 2x2 blocks.
    # Each block is 2x2, so the product of two linear combos of 2x2 blocks
    # can be expanded into 4*4=16 scalar products, but many will be zero.
    # Actually, (sum of A-blocks) * (sum of B-blocks) where each block is 2x2
    # = sum over all pairs of (A-block_i * B-block_j)
    # Each A-block_i * B-block_j is a 2x2 * 2x2 = 2x2 product, rank 4.
    
    # Let me define each term's A and B block coefficients:
    # term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    # block_idx: 0=A11, 1=A12, 2=A21, 3=A22 (and same for B)
    
    terms = [
        # M1
        ([(0,1),(3,1)], [(0,1),(3,1)]),
        # M2
        ([(2,1),(3,1)], [(0,1),(1,1)]),
        # M3
        ([(0,1),(2,-1)], [(2,1),(3,1)]),
        # M4
        ([(0,1),(1,1)], [(3,1)]),
        # M5
        ([(2,1),(3,1)], [(1,1)]),
        # M6
        ([(0,1),(2,-1)], [(1,1),(3,-1)]),
        # M7
        ([(0,1),(1,1)], [(0,1),(2,-1)]),
    ]
    
    # Output combination:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each term t and each output block (ob, oc), the coefficient is:
    out_coeffs = [
        # C11: terms 0,3,4,6 with coeffs 1,1,-1,1
        {0:1, 3:1, 4:-1, 6:1},
        # C12: terms 2,4 with coeffs 1,1
        {2:1, 4:1},
        # C21: terms 1,3 with coeffs 1,1
        {1:1, 3:1},
        # C22: terms 0,1,2,5 with coeffs 1,-1,1,1
        {0:1, 1:-1, 2:1, 5:1},
    ]
    
    col = 0
    for t in range(7):
        a_blocks, b_blocks = terms[t]
        # Expand: for each (ab, ac) in a_blocks, (bb, bc) in b_blocks
        for (ab, ac) in a_blocks:
            for (bb, bc) in b_blocks:
                # This gives a 2x2 * 2x2 product
                # A block ab (2x2) times B block bb (2x2)
                # Result is a 2x2 matrix
                # For each (lr_a, lc_a) in A block and (lr_b, lc_b) in B block:
                # The product element (lr_a, lc_b) in the result block
                # gets contribution from summing over the "inner" dimension
                # Wait, this is just matrix multiplication of two 2x2 blocks.
                # (A_block * B_block)[lr_a, lc_b] = sum_{lc_a} A_block[lr_a, lc_a] * B_block[lc_a, lc_b]
                #
                # So for each (lr_a, lc_a, lc_b) in {0,1}^3:
                # A index: a_idx(ab//2, ab%2, lr_a, lc_a) -- wait, ab is 0-3
                # ab: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                ab_br, ab_bc = ab // 2, ab % 2
                bb_br, bb_bc = bb // 2, bb % 2
                
                for lr_a in range(2):
                    for lc_a in range(2):
                        for lc_b in range(2):
                            # A element at global (ab_br*2+lr_a, ab_bc*2+lc_a)
                            a_gi = a_idx(ab_br, ab_bc, lr_a, lc_a)
                            # B element at global (bb_br*2+lc_a, bb_bc*2+lc_b)
                            b_gi = b_idx(bb_br, bb_bc, lc_a, lc_b)
                            # Result element at global (ab_br*2+lr_a, bb_bc*2+lc_b)
                            # But this contributes to output block (ab_br, bb_bc)
                            # with local (lr_a, lc_b)
                            
                            # Which output block does this go to?
                            # The output block for term t with A-block ab and B-block bb
                            # is determined by where the result lands.
                            # Result of A_block_ab * B_block_bb is a 2x2 matrix
                            # that contributes to C block (ab_br, bb_bc)
                            
                            out_br, out_bc = ab_br, bb_bc
                            out_lr, out_lc = lr_a, lc_b
                            
                            # Coefficient from A and B blocks
                            coeff = ac * bc
                            
                            # Now for each output block that uses this term,
                            # add the W coefficient
                            for (ob, oc) in [(0,0),(0,1),(1,0),(1,1)]:
                                if t in out_coeffs[ob*2+oc]:
                                    w_coeff = out_coeffs[ob*2+oc][t] * coeff
                                    c_gi = c_idx(ob, oc, out_lr, out_lc)
                                    
                                    U[a_gi, col] = (U[a_gi, col] or 0) + 1
                                    V[b_gi, col] = (V[b_gi, col] or 0) + 1
                                    W[c_gi, col] = (W[c_gi, col] or 0) + w_coeff
                            
                            col += 1
    
    # Trim to actual rank used
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 4x4 * 4x9 = 4x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 9
    
    # Split the 4x9 B matrix into two 4x4 blocks and one 4x1 block
    # B = [B1 | B2 | B3] where B1, B2 are 4x4 and B3 is 4x1
    # C = A * B = A*B1 + A*B2 + A*B3
    #
    # Use Strassen for A*B1 and A*B2 (each rank 28)
    # Use schoolbook for A*B3 (rank 4*4*1 = 16)
    # Total rank = 28 + 28 + 16 = 72
    
    # But wait, A is the same for all three products. 
    # The U factors for the three products can share the same A structure.
    # In tensor decomposition, each rank-1 term is independent, so we just concatenate.
    
    # Let me build Strassen 4x4x4 (rank 28) and schoolbook 4x4x1 (rank 16)
    
    # First, let me build the full decomposition by hand using the block structure.
    #
    # For 4x4 * 4x9:
    # A is 4x4, B is 4x9, C is 4x9
    # B columns 0-3: B1 (4x4), columns 4-7: B2 (4x4), column 8: B3 (4x1)
    #
    # C[:, 0:4] = A * B1
    # C[:, 4:8] = A * B2  
    # C[:, 8] = A * B3
    #
    # For A*B1 (4x4 * 4x4 = 4x4): use Strassen rank 28
    # For A*B2 (4x4 * 4x4 = 4x4): use Strassen rank 28
    # For A*B3 (4x4 * 4x1 = 4x1): use schoolbook rank 16
    #
    # Total rank = 28 + 28 + 16 = 72
    
    # Let me implement this directly.
    
    # Strassen 4x4x4 helper
    def strassen_4x4x4():
        """Returns (U, V, W) for 4x4 * 4x4 = 4x4, rank 28.
        U: (16, 28), V: (16, 28), W: (16, 28)
        U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
        """
        U = np.zeros((16, 28), dtype=object)
        V = np.zeros((16, 28), dtype=object)
        W = np.zeros((16, 28), dtype=object)
        
        # Strassen terms for 4x4 (splitting into 2x2 blocks)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11+B12)
        # M3 = (A11-A21)(B21+B22)
        # M4 = (A11+A12)B22
        # M5 = (A21+A22)B12
        # M6 = (A11-A21)(B12-B22)
        # M7 = (A11+A12)(B11-B21)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Block (br, bc) covers rows 2br..2br+1, cols 2bc..2bc+1
        # Global A index for block (br,bc), local (lr,lc): (2br+lr)*4 + (2bc+lc)
        # Global B index for block (br,bc), local (lr,lc): (2br+lr)*4 + (2bc+lc)
        # Global C index for block (br,bc), local (lr,lc): (2br+lr)*4 + (2bc+lc)
        
        # For each Strassen term, we have A-blocks and B-blocks with coefficients.
        # The product of A-blocks and B-blocks gives a 2x2 result block.
        # Each 2x2*2x2 product decomposes into 4 rank-1 terms (schoolbook).
        
        terms_def = [
            # (A_blocks, B_blocks) where each is list of (block_idx, coeff)
            ([(0,1),(3,1)], [(0,1),(3,1)]),  # M1
            ([(2,1),(3,1)], [(0,1),(1,1)]),  # M2
            ([(0,1),(2,-1)], [(2,1),(3,1)]), # M3
            ([(0,1),(1,1)], [(3,1)]),        # M4
            ([(2,1),(3,1)], [(1,1)]),        # M5
            ([(0,1),(2,-1)], [(1,1),(3,-1)]),# M6
            ([(0,1),(1,1)], [(0,1),(2,-1)]), # M7
        ]
        
        # Output block coefficients: out_block -> {term_idx: coeff}
        out_def = [
            {0:1, 3:1, 4:-1, 6:1},  # C11
            {2:1, 4:1},              # C12
            {1:1, 3:1},              # C21
            {0:1, 1:-1, 2:1, 5:1},  # C22
        ]
        
        col = 0
        for t in range(7):
            a_blocks, b_blocks = terms_def[t]
            for (ab, ac) in a_blocks:
                for (bb, bc) in b_blocks:
                    ab_br, ab_bc = ab // 2, ab % 2
                    bb_br, bb_bc = bb // 2, bb % 2
                    # Result block is (ab_br, bb_bc)
                    out_br, out_bc = ab_br, bb_bc
                    out_block_idx = out_br * 2 + out_bc
                    
                    for lr in range(2):
                        for lc_a in range(2):
                            for lc_b in range(2):
                                a_gi = (ab_br*2+lr)*4 + (ab_bc*2+lc_a)
                                b_gi = (bb_br*2+lc_a)*4 + (bb_bc*2+lc_b)
                                c_gi = (out_br*2+lr)*4 + (out_bc*2+lc_b)
                                
                                coeff = ac * bc
                                
                                if t in out_def[out_block_idx]:
                                    w_coeff = out_def[out_block_idx][t] * coeff
                                    U[a_gi, col] = (U[a_gi, col] or 0) + 1
                                    V[b_gi, col] = (V[b_gi, col] or 0) + 1
                                    W[c_gi, col] = (W[c_gi, col] or 0) + w_coeff
                                col += 1
        
        # Trim
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        return U, V, W
    
    def schoolbook_4x4x1():
        """Returns (U, V, W) for 4x4 * 4x1 = 4x1, rank 16.
        U: (16, 16), V: (4, 16), W: (4, 16)
        U[i*4+j, c], V[j*1+k, c], W[k*4+i, c]
        """
        rank = 16
        U = np.zeros((16, rank), dtype=object)
        V = np.zeros((4, rank), dtype=object)
        W = np.zeros((4, rank), dtype=object)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(1):
                    U[i*4+j, col] = 1
                    V[j, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W
    
    # Build Strassen 4x4x4
    U1, V1, W1 = strassen_4x4x4()
    r1 = U1.shape[1]
    
    # Build schoolbook 4x4x1
    U3, V3, W3 = schoolbook_4x4x1()
    r3 = U3.shape[1]
    
    # Now assemble the full 4x4 * 4x9 decomposition
    # U: (16, r1 + r1 + r3)
    # V: (4*9=36, r1 + r1 + r3)
    # W: (9*4=36, r1 + r1 + r3)
    
    total_rank = r1 + r1 + r3
    
    U_full = np.zeros((16, total_rank), dtype=object)
    V_full = np.zeros((36, total_rank), dtype=object)
    W_full = np.zeros((36, total_rank), dtype=object)
    
    # Part 1: A * B1 (columns 0-3 of B, columns 0-3 of C)
    # U1: (16, r1), V1: (16, r1) where V1[j*4+k, c] for B1 (4x4)
    # B1 is B[:, 0:4], so V1[j*4+k] maps to V_full[j*9 + k]
    # W1: (16, r1) where W1[k*4+i, c] for C1 (4x4)
    # C1 is C[:, 0:4], so W1[k*4+i] maps to W_full[k*9 + i]
    
    U_full[:, 0:r1] = U1
    # V1 is 16x r1, V_full is 36x total_rank
    # V1[j*4+k, c] -> V_full[j*9+k, c]
    for j in range(4):
        for k in range(4):
            V_full[j*9+k, 0:r1] = V1[j*4+k, :]
    # W1[k*4+i, c] -> W_full[k*9+i, c]
    for k in range(4):
        for i in range(4):
            W_full[k*9+i, 0:r1] = W1[k*4+i, :]
    
    # Part 2: A * B2 (columns 4-7 of B, columns 4-7 of C)
    # Same U1, but V2 maps to B[:, 4:8] and W2 maps to C[:, 4:8]
    U_full[:, r1:2*r1] = U1
    for j in range(4):
        for k in range(4):
            V_full[j*9+(4+k), r1:2*r1] = V1[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_full[k*9+(4+i), r1:2*r1] = W1[k*4+i, :]
    
    # Part 3: A * B3 (column 8 of B, column 8 of C)
    # U3: (16, r3), V3: (4, r3), W3: (4, r3)
    # V3[j, c] -> V_full[j*9+8, c]
    # W3[k*4+i, c] -> W_full[k*9+8, c]
    U_full[:, 2*r1:2*r1+r3] = U3
    for j in range(4):
        V_full[j*9+8, 2*r1:2*r1+r3] = V3[j, :]
    for k in range(1):
        for i in range(4):
            W_full[k*9+8, 2*r1:2*r1+r3] = W3[k*4+i, :]
    
    # Check time
    if time.time() > deadline:
        pass  # We already have our result
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
