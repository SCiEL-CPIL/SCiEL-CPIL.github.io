"""Matrix multiplication 4x4x8 with Strassen-based decomposition."""
DIMENSIONS = [4, 4, 8]

# Strassen 2x2x2 coefficients
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def _strassen_2x2():
    """Return (a, b, c) for Strassen 2x2x2, each is dict mapping (i,j) or (k) to list of (s, val)."""
    # a[i][j] = list of (s, val) for the s-th term
    a = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}
    b = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}
    c = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}
    
    # M1 = (A00+A11)(B00+B11)
    a[(0,0)].append((0, 1))
    a[(1,1)].append((0, 1))
    b[(0,0)].append((0, 1))
    b[(1,1)].append((0, 1))
    c[(0,0)].append((0, 1))
    c[(1,1)].append((0, 1))
    
    # M2 = (A10+A11)B00
    a[(1,0)].append((1, 1))
    a[(1,1)].append((1, 1))
    b[(0,0)].append((1, 1))
    c[(1,0)].append((1, 1))
    c[(1,1)].append((1, -1))
    
    # M3 = A00(B01-B11)
    a[(0,0)].append((2, 1))
    b[(0,1)].append((2, 1))
    b[(1,1)].append((2, -1))
    c[(0,1)].append((2, 1))
    c[(1,1)].append((2, 1))
    
    # M4 = A11(B10-B00)
    a[(1,1)].append((3, 1))
    b[(1,0)].append((3, 1))
    b[(0,0)].append((3, -1))
    c[(0,0)].append((3, 1))
    c[(1,0)].append((3, 1))
    
    # M5 = (A00+A01)B11
    a[(0,0)].append((4, 1))
    a[(0,1)].append((4, 1))
    b[(1,1)].append((4, 1))
    c[(0,0)].append((4, -1))
    c[(0,1)].append((4, 1))
    
    # M6 = (A10-A00)(B00+B01)
    a[(1,0)].append((5, 1))
    a[(0,0)].append((5, -1))
    b[(0,0)].append((5, 1))
    b[(0,1)].append((5, 1))
    c[(1,1)].append((5, 1))
    
    # M7 = (A01-A11)(B10+B11)
    a[(0,1)].append((6, 1))
    a[(1,1)].append((6, -1))
    b[(1,0)].append((6, 1))
    b[(1,1)].append((6, 1))
    c[(0,0)].append((6, 1))
    
    return a, b, c


def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication
    using Strassen's algorithm.
    
    The 4x4x8 product is split into two 4x4x4 products:
    - C[:, 0:4] = A * B[:, 0:4]  (rank 49 via Strassen on 4x4x4)
    - C[:, 4:8] = A * B[:, 4:8]  (rank 49 via Strassen on 4x4x4)
    
    Total rank = 98.
    
    W convention: W[k*n + i, r] where k is column of B, i is row of A.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get Strassen 2x2 coefficients
    a2, b2, c2 = _strassen_2x2()
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # This combines 4 independent 2x2x2 Strassen blocks
    # A is 4x4, split into 4 blocks of 2x2: A[r1*2+r2, c1*2+c2]
    # B is 4x4, split into 4 blocks: B[r1*2+r2, c1*2+c2]
    # C is 4x4, split into 4 blocks: C[r1*2+r2, c1*2+c2]
    
    # For 4x4x4, we need to handle all pairs of 2x2 blocks
    # The (r1, c1) block of C depends on all 4 pairs of blocks from A and B
    # C[r1,c1] = sum_{r2,c2} A[r1,r2] * B[r2,c1]
    
    # Actually, for the 4x4 case, we can use Strassen recursively:
    # Split A, B, C into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Similarly for B, C
    # Then apply Strassen to the 2x2 "super-matrices"
    
    # Let's build the 4x4x4 decomposition directly
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    
    # For the 4x4x4 Strassen, we use the same 7-term structure but at the block level
    # Then each block-level term expands into 2x2 Strassen terms
    
    # Let me build it more carefully:
    # A is 4x4: index as A[i, j] where i in 0..3, j in 0..3
    # B is 4x4: index as B[j, k] where j in 0..3, k in 0..3
    # C is 4x4: index as C[i, k] where i in 0..3, k in 0..3
    
    # Split into 2x2 blocks:
    # A[r1, c1] where r1 in {0,1}, c1 in {0,1}, each block is 2x2
    # A[r1][c1][r2][c2] = A[r1*2+r2, c1*2+c2]
    
    # The 4x4x4 Strassen uses 7 terms at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_s is a 2x2 matrix product computed via Strassen (7 terms)
    # So total 7*7 = 49 terms
    
    # For each block-level term s (0..6) and inner Strassen term t (0..6):
    # We have a scalar term in the decomposition
    
    # U4[i*4+j, s*7+t] gets the coefficient from a2 for A's block
    # V4[j*4+k, s*7+t] gets the coefficient from b2 for B's block  
    # W4[k*4+i, s*7+t] gets the coefficient from c2 for C's block
    
    # But we need to be careful about which blocks contribute to which terms
    
    # Let me define the block-level Strassen:
    # For each term s (0..6), we have coefficients for A blocks, B blocks, C blocks
    # a_block[s][(r1, c1)] = list of (t, val) where t is inner Strassen index
    # This is getting complex. Let me use a different approach.
    
    # Actually, the simplest way: the 4x4x4 tensor can be decomposed by
    # treating it as 4 independent 2x2x2 multiplications that share structure.
    
    # Let me just build it by combining 4 Strassen 2x2x2 decompositions
    # for the diagonal blocks, but that's not correct because of cross-terms.
    
    # Correct approach: Use the recursive Strassen structure.
    # The 4x4x4 multiplication tensor T4 has components T4[i, j, k] = 1
    # We decompose it as sum over s=0..48 of U4[i*4+j, s] * V4[j*4+k, s] * W4[k*4+i, s]
    
    # Using Strassen recursion:
    # Level 0: 2x2x2, rank 7
    # Level 1: 4x4x4, rank 49
    
    # For 4x4x4, the Strassen terms are:
    # Term (s, t) where s in 0..6 (block level), t in 0..6 (inner level)
    
    # A block (r1, c1) contributes to A with coefficient a_block[s][r1][c1] * a2[r2][c2][t]
    # where r2, c2 are within the 2x2 block
    
    # Let me define the block-level coefficients:
    # a_blks[s][(r1, c1)] = coefficient for A block (r1, c1) in term s
    # b_blks[s][(r1, c1)] = coefficient for B block (r1, c1) in term s
    # c_blks[s][(r1, c1)] = coefficient for C block (r1, c1) in term s
    
    a_blk = {}
    b_blk = {}
    c_blk = {}
    
    # M1 = (A00+A11)(B00+B11)
    a_blk[0] = {(0,0): 1, (1,1): 1}
    b_blk[0] = {(0,0): 1, (1,1): 1}
    c_blk[0] = {(0,0): 1, (1,1): 1}
    
    # M2 = (A10+A11)B00
    a_blk[1] = {(1,0): 1, (1,1): 1}
    b_blk[1] = {(0,0): 1}
    c_blk[1] = {(1,0): 1, (1,1): -1}
    
    # M3 = A00(B01-B11)
    a_blk[2] = {(0,0): 1}
    b_blk[2] = {(0,1): 1, (1,1): -1}
    c_blk[2] = {(0,1): 1, (1,1): 1}
    
    # M4 = A11(B10-B00)
    a_blk[3] = {(1,1): 1}
    b_blk[3] = {(1,0): 1, (0,0): -1}
    c_blk[3] = {(0,0): 1, (1,0): 1}
    
    # M5 = (A00+A01)B11
    a_blk[4] = {(0,0): 1, (0,1): 1}
    b_blk[4] = {(1,1): 1}
    c_blk[4] = {(0,0): -1, (0,1): 1}
    
    # M6 = (A10-A00)(B00+B01)
    a_blk[5] = {(1,0): 1, (0,0): -1}
    b_blk[5] = {(0,0): 1, (0,1): 1}
    c_blk[5] = {(1,1): 1}
    
    # M7 = (A01-A11)(B10+B11)
    a_blk[6] = {(0,1): 1, (1,1): -1}
    b_blk[6] = {(1,0): 1, (1,1): 1}
    c_blk[6] = {(0,0): 1}
    
    # Now build U4, V4, W4 for 4x4x4
    # U4 has shape (16, 49), indexed by U4[i*4+j, s*7+t]
    # V4 has shape (16, 49), indexed by V4[j*4+k, s*7+t]
    # W4 has shape (16, 49), indexed by W4[k*4+i, s*7+t]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            # For each block in A that contributes to term s:
            for (r1, c1), aval in a_blk[s].items():
                # Within block (r1, c1), A[r1*2+r2, c1*2+c2]
                for r2 in range(2):
                    for c2 in range(2):
                        # Find coefficient from a2
                        a_coeff = 0
                        for (idx, val) in a2.get((r2, c2), []):
                            if idx == t:
                                a_coeff += val
                        i = r1*2 + r2
                        j = c1*2 + c2
                        U4[i*4 + j][col] += aval * a_coeff
            
            for (r1, c1), bval in b_blk[s].items():
                # Within block (r1, c1), B[r1*2+r2, c1*2+c2]
                for r2 in range(2):
                    for c2 in range(2):
                        b_coeff = 0
                        for (idx, val) in b2.get((r2, c2), []):
                            if idx == t:
                                b_coeff += val
                        j = r1*2 + r2
                        k = c1*2 + c2
                        V4[j*4 + k][col] += bval * b_coeff
            
            for (r1, c1), cval in c_blk[s].items():
                # Within block (r1, c1), C[r1*2+r2, c1*2+c2]
                for r2 in range(2):
                    for c2 in range(2):
                        c_coeff = 0
                        for (idx, val) in c2.get((r2, c2), []):
                            if idx == t:
                                c_coeff += val
                        i = r1*2 + r2
                        k = c1*2 + c2
                        W4[k*4 + i][col] += cval * c_coeff
    
    # Now for 4x4x8:
    # Split B into two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8]
    # C = A * B = [A * B_left, A * B_right]
    # So we use two independent 4x4x4 Strassen decompositions
    
    rank = 98
    U = [[0]*rank for _ in range(16)]   # n*m = 4*4 = 16
    V = [[0]*rank for _ in range(32)]   # m*p = 4*8 = 32
    W = [[0]*rank for _ in range(32)]   # p*n = 8*4 = 32
    
    # For left half (k in 0..3):
    for i in range(4):
        for j in range(4):
            U[i*4+j][0:49] = U4[i*4+j]
    for j in range(4):
        for k in range(4):
            V[j*8+k][0:49] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W[k*4+i][0:49] = W4[k*4+i]
    
    # For right half (k in 4..7):
    for i in range(4):
        for j in range(4):
            U[i*4+j][49:98] = U4[i*4+j]
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4)][49:98] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i][49:98] = W4[k*4+i]
    
    # Convert to integers (they should already be integers)
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
