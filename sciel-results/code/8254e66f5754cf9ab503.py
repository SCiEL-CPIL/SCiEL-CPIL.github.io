"""Rewritten candidate for matrix multiplication tensor decomposition (3x5x5)."""
import time
import numpy as np

_DIM = (3, 5, 5)


def _strassen_222():
    """Return Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) or (4, 7) with rank 7.
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    
    Strassen's 7 products for 2x2 matrix mult:
    Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    # U rows: A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows: B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = (a+d)(e+h): U contributes a+d to A-part, e+h to B-part
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],    # b: M5, M7
        [0, 1, 0, 0, 0, 1, 0],    # c: M2, M6
        [1, 1, 0, 1, 0, 0, -1],   # d: M1, M2, M4, M7
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # e: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],    # f: M3, M6
        [0, 0, -1, 1, 0, 0, 1],   # g: M3, M4, M7
        [1, 0, 0, 0, 1, 0, 1],    # h: M1, M5, M7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0] = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1] = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0] = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1] = M1 - M2 + M3 + M6
    ]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition using numpy for speed."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    r = Ua.shape[1]
    
    # For each (i,k), check that the bilinear form matches
    # Coefficient of A[a,b]*B[c2,d] in C[i,k] should be delta[a,i]*delta[b,c2]*delta[d,k]
    # This coefficient is sum_r' W[k*n+i, r'] * U[a*m+b, r'] * V[c2*p+d, r']
    
    # Efficient check: for each r', the outer product U[:,r'] (n*m) x V[:,r'] (m*p) 
    # weighted by W[k*n+i, r'] should sum to the correct tensor slice.
    
    # Build expected tensor: T[i,k, a,b,c2,d] = 1 if a==i, b==c2, d==k else 0
    # Reshape: for each (i,k), the matrix M_{i,k}[a*m+b, c2*p+d] = delta[a,i]*delta[b,c2]*delta[d,k]
    
    for i in range(n):
        for k in range(p):
            # Compute the actual bilinear form matrix
            # M_actual[a*m+b, c2*p+d] = sum_r' W[k*n+i, r'] * U[a*m+b, r'] * V[c2*p+d, r']
            M_actual = np.zeros((n * m, m * p))
            for r_idx in range(r):
                w_val = Wa[k * n + i, r_idx]
                if w_val != 0:
                    M_actual += w_val * np.outer(Ua[:, r_idx], Va[:, r_idx])
            
            # Expected: M_expected[a*m+b, c2*p+d] = 1 if a==i, b==c2, d==k
            M_expected = np.zeros((n * m, m * p))
            for b in range(m):
                idx_a = i * m + b
                idx_b = b * p + k
                M_expected[idx_a, idx_b] = 1.0
            
            if not np.allclose(M_actual, M_expected, atol=1e-9):
                return False
    return True


def _schoolbook_factors(n, m, p):
    """Return schoolbook rank-n*m*p decomposition."""
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
    return U, V, W


def _build_block_decomposition(n, m, p):
    """Build a block decomposition using Strassen for 2x2 sub-blocks.
    
    For 3x5x5:
    - Split A (3x5) into rows: top 2 rows form a 2x5 block, bottom 1 row is 1x5
    - Split B (5x5) into cols: left 2 cols form 5x2, right 3 cols form 5x3
    - Also split B rows: top 2, bottom 3
    
    Strategy: Use Strassen for the 2x2 core (rows 0-1 of A, cols 0-1 of B, rows 0-1 of B)
    and schoolbook for the rest.
    
    Let's partition:
    A rows: {0,1} (2 rows) and {2} (1 row)
    B cols: {0,1} (2 cols) and {2,3,4} (3 cols)  
    B rows: {0,1} (2 rows) and {2,3,4} (3 rows)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i in {0,1}, k in {0,1}: use Strassen on A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    For i in {0,1}, k in {2,3,4}: schoolbook on A[0:2, :] * B[:, k] 
    For i = 2, k in {0,1}: schoolbook on A[2, :] * B[:, k]
    For i = 2, k in {2,3,4}: schoolbook on A[2, :] * B[:, k]
    
    Wait, but the Strassen part only handles A[0:2,0:2]*B[0:2,0:2]. We also need:
    A[0:2, 2:5] * B[2:5, 0:2] for C[0:2, 0:2]
    
    So let me be more careful:
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:5] * B[2:5, 0:2]
    
    The first term is 2x2x2 (Strassen, rank 7)
    The second term: A[0:2, 2:5] is 2x3, B[2:5, 0:2] is 3x2, result 2x2. Rank = 2*3*2 = 12 schoolbook.
    
    C[0:2, 2:5] = A[0:2, :] * B[:, 2:5]
    = A[0:2, 0:2] * B[0:2, 2:5] + A[0:2, 2:5] * B[2:5, 2:5]
    First: 2x2 * 2x3 = 2x3, rank 2*2*3=12
    Second: 2x3 * 3x3 = 2x3, rank 2*3*3=18
    Total for this block: 30
    
    C[2, 0:2] = A[2, :] * B[:, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:5]*B[2:5,0:2]
    First: 1x2 * 2x2 = 1x2, rank 4
    Second: 1x3 * 3x2 = 1x2, rank 6
    Total: 10
    
    C[2, 2:5] = A[2, :] * B[:, 2:5] = A[2,0:2]*B[0:2,2:5] + A[2,2:5]*B[2:5,2:5]
    First: 1x2 * 2x3 = 1x3, rank 6
    Second: 1x3 * 3x3 = 1x3, rank 9
    Total: 15
    
    Grand total: 7 + 12 + 30 + 10 + 15 = 74
    
    That's only saving 1! Let me try a different partition.
    
    Alternative: Use Strassen for multiple 2x2 blocks.
    
    Let me think differently. The key is to find 2x2 sub-problems where both A and B have 2x2 sub-matrices.
    
    For A (3x5) and B (5x5):
    A has rows 0,1,2 and cols 0,1,2,3,4
    B has rows 0,1,2,3,4 and cols 0,1,2,3,4
    
    A 2x2 Strassen block requires:
    - 2 rows of A
    - 2 cols of A (which are 2 rows of B)
    - 2 cols of B
    
    So we need to choose:
    - Row pair from A: (0,1) or (0,2) or (1,2)
    - Col pair from A = Row pair from B: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    - Col pair from B: any pair from 5 cols
    
    Each such choice gives a 2x2x2 sub-problem with rank 7 instead of 8.
    
    But these sub-problems overlap! We can't just add them up.
    
    Better approach: partition the computation into non-overlapping blocks.
    
    Partition A rows into {0,1} and {2}
    Partition B cols into {0,1} and {2,3,4}
    Partition B rows into {0,1} and {2,3,4}
    
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:5] * B[2:5, 0:2]
    - Term 1: 2x2 x 2x2 -> Strassen rank 7
    - Term 2: 2x3 x 3x2 -> schoolbook rank 12
    
    C[0:2, 2:5] = A[0:2, 0:2] * B[0:2, 2:5] + A[0:2, 2:5] * B[2:5, 2:5]
    - Term 1: 2x2 x 2x3 -> schoolbook rank 12
    - Term 2: 2x3 x 3x3 -> schoolbook rank 18
    
    C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2:5] * B[2:5, 0:2]
    - Term 1: 1x2 x 2x2 -> schoolbook rank 4
    - Term 2: 1x3 x 3x2 -> schoolbook rank 6
    
    C[2, 2:5] = A[2, 0:2] * B[0:2, 2:5] + A[2, 2:5] * B[2:5, 2:5]
    - Term 1: 1x2 x 2x3 -> schoolbook rank 6
    - Term 2: 1x3 x 3x3 -> schoolbook rank 9
    
    Total: 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74
    
    Only saves 1. Let me try using Strassen on more blocks.
    
    What if I use Strassen on C[0:2, 0:2] term 1 AND also find another 2x2x2?
    
    Actually, the issue is that A only has 3 rows, so I can only form one 2-row block from A.
    And B only has 5 cols, so I can form one 2-col block from B.
    
    But wait - I can also use Strassen differently. What about using the 2x2 structure in a different way?
    
    Let me try: partition B rows into {0,1} and {2,3,4}, and A cols into {0,1} and {2,3,4}.
    Also partition B cols into {0,1} and {2,3,4}.
    
    The only 2x2x2 block is A[0:2, 0:2] * B[0:2, 0:2].
    
    Hmm, let me try a different approach: use Strassen for 2x2x2 and see if I can get rank 71.
    
    The inspiration says rank 71, saving 4. Let me think about how.
    
    Maybe: use Strassen for the 2x2x2 core, and then the remaining terms can be optimized.
    
    Actually, re-reading: "reduces the rank from 75 to 71 by exploiting algebraic structure in the 2x2x2 blocks, saving 4 multiplications."
    
    75 - 71 = 4. So we save 4 multiplications total.
    
    One 2x2x2 Strassen saves 1 (8->7). To save 4, we need 4 such savings.
    
    But with 3x5x5, how many disjoint 2x2x2 blocks can we form?
    
    A is 3x5: we can pick at most 1 pair of rows (since 3 rows, only one 2-row subset that's "clean" in a partition).
    Actually we can pick rows {0,1} leaving row 2, or rows {0,2} leaving row 1, etc.
    
    In a partition sense: A rows partition into {0,1} | {2} (one 2-block)
    B rows partition into {0,1} | {2,3,4} (one 2-block)  
    B cols partition into {0,1} | {2,3,4} (one 2-block)
    
    So only one 2x2x2 block. That saves 1, giving rank 74.
    
    To get 71, we need a smarter approach. Let me look at the inspiration more carefully.
    
    The inspirations define _strassen_patterns, _get_splits, _compute_rank, _build_decomposition.
    This suggests trying multiple split patterns and picking the best.
    
    Let me try: what if we don't require a clean partition? What if we use overlapping Strassen blocks?
    
    No, that doesn't work for tensor decomposition - the blocks must be disjoint in the sense that each (i,j,k) triple is covered exactly once.
    
    Let me reconsider. Maybe the approach is:
    - Use Strassen for 2x2x2: saves 1
    - Then apply a similar trick to another sub-block
    
    Or maybe: use a different algebraic identity for some of the remaining terms.
    
    Actually, let me just implement the straightforward block decomposition with one Strassen block (rank 74) and then see if I can improve.
    
    Wait, the inspirations got 71. Let me think harder.
    
    What if we split differently:
    A rows: {0,1} | {2}
    B rows: {0,1} | {2,3,4}  
    B cols: {0,1,2} | {3,4}  -- different split!
    
    Then:
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    Still one 2x2x2.
    
    What about:
    A rows: {0,1} | {2}
    B rows: {0,1} | {2,3,4}
    B cols: {0,1} | {2,3,4}
    
    And within A[0:2, 2:5] * B[2:5, 0:2] (which is 2x3 * 3x2):
    Can we use Strassen here? No, because it's not 2x2.
    
    Hmm. Let me try yet another approach. What if we use Strassen on a 2x2x2 and then the remaining computation can be done with fewer terms than schoolbook?
    
    Actually, I think the key insight might be: after using Strassen for the 2x2x2 core, some of the remaining "cross terms" can be combined or simplified.
    
    Let me just implement the rank-74 version first and verify it, then try to optimize further.
    
    Actually, let me try a completely different partition:
    
    What if I use Strassen for TWO 2x2x2 blocks by choosing different row/col pairs?
    
    Block 1: A rows {0,1}, A cols {0,1} = B rows {0,1}, B cols {0,1}
    Block 2: A rows {0,1}, A cols {2,3} = B rows {2,3}, B cols {2,3}
    
    But these overlap in A rows {0,1}! In a tensor decomposition, each "rank-1 term" must be independent. The issue is that the U and V matrices would have overlapping support.
    
    Actually, in a tensor decomposition, the blocks don't need to be disjoint in the input indices - they just need to correctly reproduce the output. The rank is the number of terms, and each term is a product of three vectors.
    
    So I CAN use multiple Strassen blocks that share input indices, as long as the total decomposition is correct. But the rank would be the sum of the ranks of all blocks (since each block contributes its own set of rank-1 terms).
    
    So if I use two Strassen 2x2x2 blocks (each rank 7) plus schoolbook for the rest, the total rank would be 7 + 7 + (remaining schoolbook terms).
    
    Let me compute:
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (contributes to C[0,0], C[0,1], C[1,0], C[1,1])
    Block 2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (contributes to C[0,2], C[0,3], C[1,2], C[1,3])
    
    Remaining terms:
    C[0:2, 0:2] also needs: A[0:2, 2:5] * B[2:5, 0:2] (2x3 * 3x2 = 12 schoolbook)
    C[0:2, 0:2] also needs: A[0:2, 4] * B[4, 0:2] -- wait, I already included cols 2,3 in block 2.
    
    Let me be more careful:
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    For i in {0,1}, k in {0,1}:
    = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3,4}} A[i,j]*B[j,k]
    First part: Strassen block 1 (rank 7)
    Second part: schoolbook 2*3*2 = 12
    
    For i in {0,1}, k in {2,3}:
    = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k] + sum_{j=4} A[i,j]*B[j,k]
    First part: 2x2 * 2x2 schoolbook = 8 (or could use Strassen? But B rows {0,1} and B cols {2,3} - this is A[0:2,0:2] * B[0:2,2:4], which is 2x2 * 2x2)
    Second part: Strassen block 2 (rank 7) for A[0:2,2:4] * B[2:4,2:4]
    Third part: 2*1*2 = 4 schoolbook
    
    For i in {0,1}, k in {4}:
    = sum_{j in {0,1}} A[i,j]*B[j,4] + sum_{j in {2,3,4}} A[i,j]*B[j,4]
    First: 2*2*1 = 4
    Second: 2*3*1 = 6
    Total: 10
    
    For i = 2, k in {0,1}:
    = sum_{j in {0,1}} A[2,j]*B[j,k] + sum_{j in {2,3,4}} A[2,j]*B[j,k]
    First: 1*2*2 = 4
    Second: 1*3*2 = 6
    Total: 10
    
    For i = 2, k in {2,3}:
    = sum_{j in {0,1}} A[2,j]*B[j,k] + sum_{j in {2,3}} A[2,j]*B[j,k] + sum_{j=4} A[2,j]*B[j,4]
    First: 1*2*2 = 4
    Second: 1*2*2 = 4
    Third: 1*1*2 = 2
    Total: 10
    
    For i = 2, k in {4}:
    = sum_{j in {0,1}} A[2,j]*B[j,4] + sum_{j in {2,3,4}} A[2,j]*B[j,4]
    First: 1*2*1 = 2
    Second: 1*3*1 = 3
    Total: 5
    
    Grand total: 7 + 12 + 8 + 7 + 4 + 10 + 10 + 10 + 5 = 73
    
    Still not 71. Let me try to optimize further.
    
    What if for "A[0:2,0:2] * B[0:2,2:4]" I also use Strassen? That's another 2x2x2!
    
    Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    
    But wait, this uses the same A sub-block as Block 1. In the tensor decomposition, this is fine - we just add more rank-1 terms.
    
    New count:
    Block 1: A[0:2,0:2] * B[0:2,0:2] -> rank 7
    Block 2: A[0:2,2:4] * B[2:4,2:4] -> rank 7
    Block 3: A[0:2,0:2] * B[0:2,2:4] -> rank 7
    
    Remaining:
    C[0:2, 0:2] remaining: A[0:2, 2:5] * B[2:5, 0:2] = 2*3*2 = 12
    C[0:2, 2:4] remaining: A[0:2, 4] * B[4, 2:4] = 2*1*2 = 4
    C[0:2, 4]: A[0:2, 0:2]*B[0:2,4] + A[0:2, 2:4]*B[2:4,4] + A[0:2,4]*B[4,4]
    = 2*2*1 + 2*2*1 + 2*1*1 = 4+4+2 = 10
    C[2, 0:2]: A[2,0:2]*B[0:2,0:2] + A[2,2:5]*B[2:5,0:2] = 1*2*2 + 1*3*2 = 4+6 = 10
    C[2, 2:4]: A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4] + A[2,4]*B[4,2:4]
    = 1*2*2 + 1*2*2 + 1*1*2 = 4+4+2 = 10
    C[2, 4]: A[2,0:2]*B[0:2,4] + A[2,2:4]*B[2:4,4] + A[2,4]*B[4,4]
    = 1*2*1 + 1*2*1 + 1*1*1 = 2+2+1 = 5
    
    Total: 7+7+7 + 12+4+10+10+10+5 = 21 + 51 = 72
    
    Getting closer! One more Strassen block?
    
    What about A[0:2, 2:4] * B[2:4, 0:2]? That's for C[0:2, 0:2] part.
    But I already have A[0:2, 2:5] * B[2:5, 0:2] as schoolbook. If I split that:
    A[0:2, 2:4] * B[2:4, 0:2] (2x2*2x2, Strassen rank 7) + A[0:2, 4] * B[4, 0:2] (2x1*1x2 = 4)
    
    Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> rank 7
    
    New remaining for C[0:2, 0:2]: A[0:2, 4] * B[4, 0:2] = 2*1*2 = 4 (was 12, now 7+4=11, saved 1)
    
    Total: 7+7+7+7 + 4+4+10+10+10+5 = 28 + 43 = 71!
    
    Yes! Rank 71!
    
    Let me verify the full accounting:
    
    Strassen blocks (each rank 7):
    1. A[0:2,0:2] * B[0:2,0:2] -> contributes to C[0:2,0:2]
    2. A[0:2,2:4] * B[2:4,2:4] -> contributes to C[0:2,2:4]
    3. A[0:2,0:2] * B[0:2,2:4] -> contributes to C[0:2,2:4]
    4. A[0:2,2:4] * B[2:4,0:2] -> contributes to C[0:2,0:2]
    
    Schoolbook terms:
    5. A[0:2,4] * B[4,0:2] -> C[0:2,0:2]: rank 2*1*2=4
    6. A[0:2,4] * B[4,2:4] -> C[0:2,2:4]: rank 2*1*2=4
    7. C[0:2,4]: A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4] + A[0:2,4]*B[4,4]: rank 4+4+2=10
    8. C[2,0:2]: A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2] + A[2,4]*B[4,0:2]: rank 4+4+2=10
    9. C[2,2:4]: A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4] + A[2,4]*B[4,2:4]: rank 4+4+2=10
    10. C[2,4]: A[2,0:2]*B[0:2,4] + A[2,2:4]*B[2:4,4] + A[2,4]*B[4,4]: rank 2+2+1=5
    
    Total: 4*7 + 4+4+10+10+10+5 = 28 + 43 = 71 ✓
    
    Now I need to implement this. Each Strassen block and each schoolbook block contributes rank-1 terms to U, V, W.
    
    For a Strassen block with A rows {i1,i2}, A cols {j1,j2}, B rows {j1,j2}, B cols {k1,k2}:
    The 7 Strassen terms each contribute one rank-1 term.
    U[i1*m+j1, c] += coeff, etc.
    
    For a schoolbook block with A rows I, A cols J, B rows J, B cols K:
    For each (i in I, j in J, k in K): one rank-1 term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    
    Let me implement this.
    """
    n, m, p = 3, 5, 5
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    sU, sV, sW = _strassen_222()
    
    # Strassen block 1: A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {0,1}
    # -> C[0:2, 0:2]
    a_rows = [0, 1]
    a_cols = [0, 1]
    b_rows = [0, 1]
    b_cols = [0, 1]
    for c in range(7):
        for idx_i, i in enumerate(a_rows):
            for idx_j, j in enumerate(a_cols):
                if sU[idx_i * 2 + idx_j][c] != 0:
                    U[i * m + j][col] += sU[idx_i * 2 + idx_j][c]
        for idx_j, j in enumerate(b_rows):
            for idx_k, k in enumerate(b_cols):
                if sV[idx_j * 2 + idx_k][c] != 0:
                    V[j * p + k][col] += sV[idx_j * 2 + idx_k][c]
        for idx_i, i in enumerate(a_rows):
            for idx_k, k in enumerate(b_cols):
                if sW[idx_i * 2 + idx_k][c] != 0:
                    W[k * n + i][col] += sW[idx_i * 2 + idx_k][c]
        col += 1
    
    # Strassen block 2: A rows {0,1}, A cols {2,3}, B rows {2,3}, B cols {2,3}
    # -> C[0:2, 2:4]
    a_rows = [0, 1]
    a_cols = [2, 3]
    b_rows = [2, 3]
    b_cols = [2, 3]
    for c in range(7):
        for idx_i, i in enumerate(a_rows):
            for idx_j, j in enumerate(a_cols):
                if sU[idx_i * 2 + idx_j][c] != 0:
                    U[i * m + j][col] += sU[idx_i * 2 + idx_j][c]
        for idx_j, j in enumerate(b_rows):
            for idx_k, k in enumerate(b_cols):
                if sV[idx_j * 2 + idx_k][c] != 0:
                    V[j * p + k][col] += sV[idx_j * 2 + idx_k][c]
        for idx_i, i in enumerate(a_rows):
            for idx_k, k in enumerate(b_cols):
                if sW[idx_i * 2 + idx_k][c] != 0:
                    W[k * n + i][col] += sW[idx_i * 2 + idx_k][c]
        col += 1
    
    # Strassen block 3: A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {2,3}
    # -> C[0:2, 2:4]
    a_rows = [0, 1]
    a_cols = [0, 1]
    b_rows = [0, 1]
    b_cols = [2, 3]
    for c in range(7):
        for idx_i, i in enumerate(a_rows):
            for idx_j, j in enumerate(a_cols):
                if sU[idx_i * 2 + idx_j][c] != 0:
                    U[i * m + j][col] += sU[idx_i * 2 + idx_j][c]
        for idx_j, j in enumerate(b_rows):
            for idx_k, k in enumerate(b_cols):
                if sV[idx_j * 2 + idx_k][c] != 0:
                    V[j * p + k][col] += sV[idx_j * 2 + idx_k][c]
        for idx_i, i in enumerate(a_rows):
            for idx_k, k in enumerate(b_cols):
                if sW[idx_i * 2 + idx_k][c] != 0:
                    W[k * n + i][col] += sW[idx_i * 2 + idx_k][c]
        col += 1
    
    # Strassen block 4: A rows {0,1}, A cols {2,3}, B rows {2,3}, B cols {0,1}
    # -> C[0:2, 0:2]
    a_rows = [0, 1]
    a_cols = [2, 3]
    b_rows = [2, 3]
    b_cols = [0, 1]
    for c in range(7):
        for idx_i, i in enumerate(a_rows):
            for idx_j, j in enumerate(a_cols):
                if sU[idx_i * 2 + idx_j][c] != 0:
                    U[i * m + j][col] += sU[idx_i * 2 + idx_j][c]
        for idx_j, j in enumerate(b_rows):
            for idx_k, k in enumerate(b_cols):
                if sV[idx_j * 2 + idx_k][c] != 0:
                    V[j * p + k][col] += sV[idx_j * 2 + idx_k][c]
        for idx_i, i in enumerate(a_rows):
            for idx_k, k in enumerate(b_cols):
                if sW[idx_i * 2 + idx_k][c] != 0:
                    W[k * n + i][col] += sW[idx_i * 2 + idx_k][c]
        col += 1
    
    # Schoolbook blocks:
    # 5. A[0:2,4] * B[4,0:2] -> C[0:2,0:2]
    for i in [0, 1]:
        for j in [4]:
            for k in [0, 1]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 6. A[0:2,4] * B[4,2:4] -> C[0:2,2:4]
    for i in [0, 1]:
        for j in [4]:
            for k in [2, 3]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 7. C[0:2,4]: A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4] + A[0:2,4]*B[4,4]
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [0, 1]:
        for j in [4]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 8. C[2,0:2]: A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2] + A[2,4]*B[4,0:2]
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [2, 3]:
            for k in [0, 1]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [4]:
            for k in [0, 1]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 9. C[2,2:4]: A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4] + A[2,4]*B[4,2:4]
    for i in [2]:
        for j in [0, 1]:
            for k in [2, 3]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [2, 3]:
            for k in [2, 3]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [4]:
            for k in [2, 3]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 10. C[2,4]: A[2,0:2]*B[0:2,4] + A[2,2:4]*B[2:4,4] + A[2,4]*B[4,4]
    for i in [2]:
        for j in [0, 1]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [2, 3]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for i in [2]:
        for j in [4]:
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 100  # 100s safety margin
    
    # Build the Strassen-based block decomposition (rank 71)
    U, V, W = _build_block_decomposition(n, m, p)
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
    if time.time() > deadline - 10:
        U_f, V_f, W_f = _schoolbook_factors(n, m, p)
        return {"U": U_f, "V": V_f, "W": W_f}
    
    U_f, V_f, W_f = _schoolbook_factors(n, m, p)
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
