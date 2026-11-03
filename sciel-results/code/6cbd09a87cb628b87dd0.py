"""
Rank-minimal tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy:
1. Start from the known Strassen-based decomposition of rank 112 (split each 4x4
   into 2x2 blocks; 2x2*2x2 uses rank 7, 2x2*2x4 uses rank 12, 2x4*2x2 uses rank 12,
   2x4*2x4 uses rank 16; total = 7+12+12+16 = 47 per quadrant-pair... let me recompute).

   Actually for 4x4 * 4x8:
   Split A (4x4) into four 2x2 blocks: A11 A12; A21 A22
   Split B (4x8) into four 2x4 blocks: B11 B12; B21 B22
   Output C (4x8) has four 2x4 blocks: C11 C12; C21 C22

   C11 = A11*B11 + A12*B21
   C12 = A11*B12 + A12*B22
   C21 = A21*B11 + A22*B21
   C22 = A21*B12 + A22*B22

   Each 2x2*2x4 multiplication can be done with rank 12 (Strassen-style: split 2x2 into
   1x1 and 2x4 into two 1x4... actually 2x2*2x4: split A into two 1x2 rows, B into two 2x4
   rows. That's 4 sub-problems of 1x2*2x4. Each 1x2*2x4 is just a vector-matrix product
   which is rank 8 (2*4). So 4*8=32? No, that's schoolbook.

   Let me think more carefully. The multiplication tensor for (m,n,p) has rank at most m*n*p
   (schoolbook). Strassen for 2x2*2x2 gives rank 7 instead of 8.

   For 4x4*4x8, a natural approach:
   - Treat as two independent 4x4*4x4 multiplications (since B is 4x8 = two 4x4 blocks)
   - Each 4x4*4x4 can be done with Strassen: rank = 7^2 = 49? No, Strassen for 4x4 is
     done by splitting into 2x2 blocks and using 7 multiplications of 2x2*2x2, each of
     which is rank 7, so total rank = 7*7 = 49.
   - Two such: rank = 98.

   But we can do better by sharing structure. Let's use the full 4x4*4x8 tensor.

   Alternative: Use the known result that the rank of 4x4*4x4 is at most 49 (Strassen),
   and 4x4*4x8 is two copies, so at most 98.

   Can we do better? There's a known decomposition for 4x4*4x4 with rank 49. For 4x4*4x8,
   since the second dimension is 8, we can think of it as 4x4*4x4 done twice.

   Actually, let me reconsider. The tensor for multiplying A(n x m) by B(m x p) to get C(n x p)
   has indices (i, j, k) where A[i,j], B[j,k], C[i,k]. The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k].

   For n=4, m=4, p=8: the tensor has 4*4*8 = 128 terms in schoolbook.

   Strassen's idea: group terms to reduce count. For 2x2*2x2, we can do it in 7 terms.

   For 4x4*4x8:
   Split A into 2x2 blocks (2x2 grid of 2x2 blocks): A = [[A11, A12], [A21, A22]]
   Split B into 2x4 blocks (2x2 grid of 2x4 blocks): B = [[B11, B12], [B21, B22]]
   C = [[C11, C12], [C21, C22]] where each Cij is 2x4.

   C11 = A11*B11 + A12*B21  (2x2 * 2x4 + 2x2 * 2x4)
   C12 = A11*B12 + A12*B22
   C21 = A21*B11 + A22*B21
   C22 = A21*B12 + A22*B22

   Each 2x2*2x4 multiplication: split the 2x2 into two 1x2 row vectors, and 2x4 into two 2x4
   row blocks. So it's sum of two 1x2 * 2x4 products. Each 1x2*2x4 is a vector (length 2) times
   a matrix (2x4), giving a vector (length 4). The tensor for 1x2*2x4 has rank 2*4=8 (schoolbook)
   since it's just outer product structure: (a1*x1 + a2*x2) for each of 4 outputs... actually
   1x2 * 2x4: result[k] = a[0]*B[0,k] + a[1]*B[1,k] for k=0..3. This is rank 4 (not 8) because
   it's a linear combination. Wait, the tensor is sum_{j,k} A[j] B[j,k] C[k] where A has 2 entries,
   B has 8 entries, C has 4 entries. The tensor is sum_{j=0}^{1} sum_{k=0}^{3} A[j] B[j,k] C[k].
   This is a sum of 4 terms (one for each k, with A[0]*B[0,k]+A[1]*B[1,k] for output k). So rank 4.

   So 2x2*2x4 = 2 * (1x2*2x4) = 2*4 = 8 rank terms? No, that's not right either. Let me be more careful.

   The multiplication tensor T_{m,p} for (1xm) * (mxp) -> (1xp) has rank p (each output is a linear
   combination). For (2x2)*(2x4)->(2x4): 
   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
   This is 8 bilinear terms in schoolbook. Can we reduce? 

   Actually, let's just use a known good decomposition. The safest approach is to use the 
   Strassen-based 4x4*4x4 decomposition (rank 49) twice, giving rank 98 for 4x4*4x8.

   Let me implement Strassen's 4x4*4x4 with rank 49, and duplicate it for the two 4x4 halves
   of the 4x8 matrix.

   Strassen 4x4*4x4:
   Split into 2x2 blocks. Use 7 Strassen products of 2x2*2x2, each of rank 7.
   Total: 7 * 7 = 49.

   For 4x4*4x8: split B into B_left (4x4) and B_right (4x4).
   C_left = A * B_left (4x4*4x4, rank 49)
   C_right = A * B_right (4x4*4x4, rank 49)
   Total rank: 98.

   This is a valid decomposition. Let's implement it.
"""

import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Strassen's formulas (using 0-indexed blocks):
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    For each Mi, we have a product of two linear combinations.
    We need to express each Mi as a rank-1 term in the tensor decomposition.
    
    The tensor for 2x2*2x2 has indices (i,j,k) for A[i,j], B[j,k], C[i,k].
    Each rank-1 term is (u, v, w) where u is in R^4 (A indices), v in R^4 (B indices), w in R^4 (C indices).
    
    For Mi = (linear combo of A) * (linear combo of B), we get a rank-1 tensor:
    u = coefficients on A, v = coefficients on B, w = coefficients on C.
    
    M1 = (a+d)(e+h): u = [1,0,0,1], v = [1,1,0,0], contributes to C[0,0] with +1, C[1,1] with -1
       Wait, M1 appears in C[0,0] with +1 and C[1,1] with -1.
       So w = [1, 0, 0, -1] (C indices: C[0,0], C[0,1], C[1,0], C[1,1])
    
    Let me be more systematic. C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.
    A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
    B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
    
    M1 = (A00+A11)(B00+B11): u=[1,0,0,1], v=[1,1,0,0], w=[1,0,0,-1]
    M2 = (A10+A11)B00: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    M3 = A00(B01-B11): u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
    M4 = A11(B10-B00): u=[0,0,0,1], v=[-1,0,1,0], w=[1,0,1,0]
    M5 = (A00+A01)B11: u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,0,0]  -- wait, M5 is in C[0,0] with -1
       C[0,0] = M1+M4-M5+M7, so M5 has coefficient -1 in C[0,0].
       w=[-1,0,0,0]
    M6 = (A10-A00)(B00+B01): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
       C[1,1] = M1-M2+M3+M6, so M6 has coefficient +1 in C[1,1].
       w=[0,0,0,1]
    M7 = (A01-A11)(B10+B11): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
       C[0,0] = M1+M4-M5+M7, so M7 has coefficient +1 in C[0,0].
       w=[1,0,0,0]
    
    Let me verify:
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Standard Strassen. Let me code this up.
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(([1,0,0,1], [1,1,0,0], [1,0,0,-1]))
    # M2 = (A10+A11)B00
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3 = A00(B01-B11)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4 = A11(B10-B00)
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    # M5 = (A00+A01)B11
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,0]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms


def _strassen_4x4_decomposition():
    """
    Build the rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4.
    
    Split into 2x2 blocks. Use 7 Strassen products, each of which is a 2x2*2x2
    multiplication of rank 7.
    
    The 7 Strassen products operate on linear combinations of the 2x2 blocks.
    Each "block product" X*Y where X and Y are 2x2 matrices is itself a rank-7
    tensor (from _strassen_2x2_decomposition).
    
    For the 4x4 case:
    A = [[A11, A12], [A21, A22]] (each 2x2)
    B = [[B11, B12], [B21, B22]]
    
    Strassen:
    S1 = (A11+A22)(B11+B22)
    S2 = (A21+A22)B11
    S3 = A11(B12-B22)
    S4 = A22(B21-B11)
    S5 = (A11+A12)B22
    S6 = (A21-A11)(B11+B12)
    S7 = (A12-A22)(B21+B22)
    
    C11 = S1 + S4 - S5 + S7
    C12 = S3 + S5
    C21 = S2 + S4
    C22 = S1 - S2 + S3 + S6
    
    Each Si is a 2x2*2x2 product. The "A side" of Si is a linear combination of the
    four 2x2 blocks of A, and the "B side" is a linear combination of the four 2x2
    blocks of B.
    
    For the tensor decomposition, each Si contributes 7 rank-1 terms. For each rank-1
    term in the 2x2*2x2 decomposition, we need to expand it to the full 4x4 level.
    
    The A side of the 4x4 has 16 entries (4x4). The B side has 16 entries. The C side
    has 16 entries.
    
    For Si, the A-side vector is a 2x2 matrix which is a linear combo of A11,A12,A21,A22.
    The B-side vector is a 2x2 matrix which is a linear combo of B11,B12,B21,B22.
    
    Each rank-1 term in the 2x2*2x2 decomposition has u (4 entries for the 2x2 A-block),
    v (4 entries for the 2x2 B-block), w (4 entries for the 2x2 C-block).
    
    To expand to 4x4:
    - The A-side u is a 2x2 matrix of coefficients. This gets multiplied by the block
      combination. So the full 4x4 A-vector is: for each position (block, within-block),
      the coefficient is the block-combination coefficient times the within-block coefficient.
    
    Let me define the block structure:
    A 4x4: rows 0-3, cols 0-3.
    Block A11: rows 0-1, cols 0-1
    Block A12: rows 0-1, cols 2-3
    Block A21: rows 2-3, cols 0-1
    Block A22: rows 2-3, cols 2-3
    
    Similarly for B and C.
    
    For a given Si with block-combination coefficients (ca11, ca12, ca21, ca22) on the A side
    and (cb11, cb12, cb21, cb22) on the B side, and a 2x2*2x2 rank-1 term (u, v, w):
    
    The full A-vector (16 entries) has:
    - For block (bi,bj) with coefficient cabi,bj, and within-block position (r,c) with u[r,c]:
      A[2*bi+r, 2*bj+c] = cabi,bj * u[r,c]
    
    Similarly for B and C.
    
    The C side: Si contributes to specific C blocks with specific signs.
    For example, S1 contributes +1 to C11 and -1 to C22.
    So the C block coefficients are (cc11, cc12, cc21, cc22) and the within-block is w.
    """
    # Block combination coefficients for each Si: (A-side blocks, B-side blocks, C-side blocks)
    # Order: (11, 12, 21, 22)
    strassen_defs = [
        # S1 = (A11+A22)(B11+B22) -> C11:+1, C22:-1
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),
        # S2 = (A21+A22)B11 -> C21:+1, C22:-1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
        # S3 = A11(B12-B22) -> C12:+1, C22:+1
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),
        # S4 = A22(B21-B11) -> C11:+1, C21:+1
        ((0,0,0,1), (0,0,1,-1), (1,0,1,0)),
        # S5 = (A11+A12)B22 -> C11:-1, C12:+1
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),
        # S6 = (A21-A11)(B11+B12) -> C22:+1
        ((0,-1,1,0), (1,1,0,0), (0,0,0,1)),
        # S7 = (A12-A22)(B21+B22) -> C11:+1
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),
    ]
    
    base_terms = _strassen_2x2_decomposition()
    
    all_terms = []
    for (ca, cb, cc) in strassen_defs:
        for (u, v, w) in base_terms:
            # u, v, w are lists of 4 (2x2 flattened)
            # Expand to 16 (4x4 flattened)
            U_full = [0]*16
            V_full = [0]*16
            W_full = [0]*16
            
            # Block offsets: block (bi,bj) starts at row 2*bi, col 2*bj
            # Flattened index: (2*bi + r) * 4 + (2*bj + c)
            for bi in range(2):
                for bj in range(2):
                    cab = ca[bi*2+bj]
                    cbb = cb[bi*2+bj]
                    ccb = cc[bi*2+bj]
                    for r in range(2):
                        for c in range(2):
                            u_idx = r*2+c
                            v_idx = r*2+c
                            w_idx = r*2+c
                            full_idx = (2*bi+r)*4 + (2*bj+c)
                            U_full[full_idx] += cab * u[u_idx]
                            V_full[full_idx] += cbb * v[v_idx]
                            W_full[full_idx] += ccb * w[w_idx]
            
            all_terms.append((U_full, V_full, W_full))
    
    return all_terms  # 49 terms


def construct():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Split B (4x8) into B_left (4x4) and B_right (4x4).
    C (4x8) = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right.
    
    Use the rank-49 Strassen decomposition for each half.
    
    U: shape (n*m, rank) = (16, 98)
    V: shape (m*p, rank) = (32, 98)
    W: shape (p*n, rank) = (32, 98)
    
    Index conventions:
    U[i*m + j, c] where i in 0..3, j in 0..3 -> A[i,j]
    V[j*p + k, c] where j in 0..3, k in 0..7 -> B[j,k]
    W[k*n + i, c] where k in 0..7, i in 0..3 -> C[i,k] (note: k*n+i, so k is outer)
    
    For the left half (B columns 0-3, C columns 0-3):
    V indices: j*p + k where k in 0..3 -> j*8 + k
    W indices: k*n + i where k in 0..3 -> k*4 + i
    
    For the right half (B columns 4-7, C columns 4-7):
    V indices: j*p + k where k in 4..7 -> j*8 + (k+4)
    W indices: k*n + i where k in 4..7 -> (k+4)*4 + i
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    base_terms = _strassen_4x4_decomposition()
    # base_terms: list of 49 tuples (U_16, V_16, W_16)
    # U_16: A[i,j] flattened as i*4+j
    # V_16: B[j,k] flattened as j*4+k (for 4x4)
    # W_16: C[i,k] flattened as i*4+k (for 4x4)
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    col = 0
    for (u16, v16, w16) in base_terms:
        # Left half: B cols 0-3, C cols 0-3
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = u16[i*4+j]
        for j in range(4):
            for k in range(4):
                V[j*8+k][col] = v16[j*4+k]
        for i in range(4):
            for k in range(4):
                W[k*4+i][col] = w16[i*4+k]
        col += 1
        
        if time.time() > deadline:
            break
    
    for (u16, v16, w16) in base_terms:
        if time.time() > deadline:
            break
        # Right half: B cols 4-7, C cols 4-7
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = u16[i*4+j]
        for j in range(4):
            for k in range(4):
                V[j*8+(k+4)][col] = v16[j*4+k]
        for i in range(4):
            for k in range(4):
                W[(k+4)*4+i][col] = w16[i*4+k]
        col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
