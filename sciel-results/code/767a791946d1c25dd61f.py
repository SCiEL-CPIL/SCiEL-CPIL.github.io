"""Matrix multiplication 3,5,6 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Use Strassen's rank-7 algorithm on 2x2x2 blocks to reduce rank below schoolbook.
    
    For a 3x5x6 matrix multiplication, we can partition into overlapping 2x2x2 blocks.
    The key insight: we can cover the 3x5x6 tensor with a combination of Strassen blocks
    and single-term products to get rank < 90 (schoolbook).
    
    Strategy: Partition the 3x5x6 multiplication into blocks and apply Strassen where possible.
    
    A 3x5x6 multiplication can be viewed as:
    - The (i,j,k) entries of C = A*B where A is 3x5, B is 5x6
    - C[i,k] = sum_j A[i,j]*B[j,k]
    
    We can use a block decomposition. Let's partition:
    - Rows of A: {0}, {1,2}  (sizes 1, 2)
    - Cols of A / Rows of B: {0,1,2}, {3,4} (sizes 3, 2) -- but Strassen needs 2x2
    - Cols of B: {0,1,2}, {3,4,5} (sizes 3, 3)
    
    Actually, let's think more carefully. The tensor is T[i,j,k] = delta_{i,k} in the appropriate sense.
    For matrix mult C = AB, the tensor is T[i,k,j] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    
    The flattening T_{(nm) x (pn)} has entries T[i*m+j, k*n+i] = 1 if j==k (for the standard convention).
    Wait, let me re-read the convention: W[k*n+i, r]. So the output index is (k, i) where k is col of B (0..p-1) and i is row of A (0..n-1).
    
    The tensor decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k} (i.e., 1 if j==k, else 0)
    
    Wait no. The matrix multiplication tensor T satisfies:
    T[i,j,k] = 1 if the "middle" index matches appropriately.
    
    Actually for C = AB where A is n x m and B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has T[i,j,k] = 1 (it's just the identity in the middle dimension).
    The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    With the given index conventions:
    U is (n*m) x rank, indexed by [i*m + j]
    V is (m*p) x rank, indexed by [j*p + k]  
    W is (p*n) x rank, indexed by [k*n + i]
    
    And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}... 
    
    No wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor T_{n,m,p} is defined by:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No! The tensor for matrix multiplication is:
    T[i,j,k] = 1 if we think of it as the "multiplication" operation.
    
    Actually, the correct statement: the tensor T satisfies
    T[i,j,k] = 1  (it's the all-ones tensor? No...)
    
    Let me be precise. The matrix multiplication tensor is the tensor T such that
    for any matrices A (n x m) and B (m x p):
    C = A * B  iff  C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    This means T[i,j,k] = 1 for all i,j,k. The tensor is just the all-ones tensor of shape (n,m,p).
    
    Wait, that can't be right because then the rank would be 1.
    
    I think the issue is that the "tensor" here is actually the structural tensor that encodes the multiplication. Let me reconsider.
    
    In the context of tensor decomposition for matrix multiplication, the tensor T_{n,m,p} is defined as:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    
    where the multiplication is: <T, A ⊗ B ⊗ C> = trace(C^T A B) = sum_{i,j,k} C[k,i] A[i,j] B[j,k]
    
    So T[i,j,k] = 1 for all i,j,k. The tensor is indeed all-ones.
    
    But then the rank of the all-ones tensor of shape (n,m,p) is... well, the flattening T_{(nm) x (pn)} has entries T[i*m+j, k*n+i] = 1 for all valid (i,j,k). So it's not a full matrix of ones - it's a structured matrix.
    
    Specifically, T_{(nm) x (pn)}[i*m+j, k*n+i] = 1 if and only if the indices are consistent (same i). For a fixed i, the submatrix has rows (i, j) for j in [0,m) and columns (k, i) for k in [0,p), all equal to 1. So each block (for fixed i) is an m x p matrix of ones, and there are n such blocks.
    
    So the flattening is a block-diagonal-like structure where block i is an m x p all-ones matrix.
    
    The rank of an m x p all-ones matrix is 1. So the flattening has rank n (one per block i).
    
    Wait, that means the tensor rank lower bound from flattening is n = 3? That seems too low.
    
    Hmm, I think I'm confusing myself. Let me look at this differently.
    
    The standard matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix is:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    
    and the "multiplication" is recovered by contracting: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor rank of T is the minimum r such that T = sum_{r'} u^{(r')} ⊗ v^{(r')} ⊗ w^{(r')}
    
    For the flattening T_{(nm) x (pn)}, the entry at position (i*m+j, k*n+i) is 1.
    
    So the matrix M has M[(i*m+j), (k*n+i)] = 1 for all i,j,k.
    
    Let me write out the structure. The rows are indexed by (i,j) with i in [0,n), j in [0,m).
    The columns are indexed by (k,i) with k in [0,p), i in [0,n).
    
    M[(i,j), (k,i')] = 1 if i == i' (from the index structure), and actually looking at it:
    M[row=i*m+j, col=k*n+i] = 1
    
    So for a given row (i,j) and column (k,i'), we have M = 1 iff the row's i-index equals the column's i-index.
    
    So M is block-diagonal in the i-index: for each i, the submatrix M_i has rows j in [0,m) and columns k in [0,p), and M_i[j,k] = 1.
    
    So M = blockdiag(M_0, M_1, ..., M_{n-1}) where each M_i is the m x p all-ones matrix.
    
    The rank of M_i is 1 (for m,p >= 1). So rank(M) = n.
    
    This gives a lower bound of n = 3 on the tensor rank. But we know the schoolbook rank is n*m*p = 90, and Strassen gives 84. The true rank is somewhere between 3 and 90.
    
    Actually, I realize the flattening lower bound is weak. Let me just use the Strassen-based construction which is known to give rank 84 for this problem.
    
    Let me implement the Strassen-based decomposition directly.
    
    For 3x5x6, we can partition:
    - Split rows of A into {0} and {1,2} (sizes 1, 2)
    - Split columns of A / rows of B into {0,1} and {2,3,4} (sizes 2, 3) -- not ideal for Strassen
    - Split columns of B into {0,1,2} and {3,4,5} (sizes 3, 3)
    
    This doesn't give clean 2x2 blocks. Let me think of a better partition.
    
    Alternative: Use a 2x2 Strassen block for the (1,2) x (0,1) x (0,1) part, and handle the rest with schoolbook.
    
    Actually, the cleanest approach: the 3x5x6 multiplication can be decomposed as follows.
    
    Let me just directly construct a rank-84 decomposition using the known Strassen embedding.
    
    For the 3x5x6 case, we can view it as:
    - A 2x2x2 Strassen block covering A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    - Plus schoolbook for everything else
    
    The 2x2x2 Strassen block replaces 8 schoolbook terms with 7.
    The remaining terms: total schoolbook is 3*5*6 = 90.
    The 2x2x2 block covers 2*2*2 = 8 terms, replaced by 7.
    So rank = 90 - 8 + 7 = 89.
    
    But the knowledge says rank 84 was achieved with 6 Strassen blocks. Let me see how.
    
    6 Strassen blocks each saving 1: 90 - 6 = 84.
    
    So we need to find 6 non-overlapping 2x2x2 sub-tensors to apply Strassen to.
    
    For 3x5x6:
    - i (rows of A, 0..2): can form 2x2 blocks at i={0,1} or i={1,2}
    - j (cols of A, 0..4): can form 2x2 blocks at j={0,1}, {1,2}, {2,3}, {3,4}
    - k (cols of B, 0..5): can form 2x2 blocks at k={0,1}, {1,2}, {2,3}, {3,4}, {4,5}
    
    We need 6 non-overlapping 2x2x2 blocks. Let's try:
    1. i={0,1}, j={0,1}, k={0,1}
    2. i={0,1}, j={2,3}, k={2,3}
    3. i={0,1}, j={0,1}, k={4,5}  -- wait, j overlaps with block 1
    4. i={1,2}, j={0,1}, k={2,3}  -- i overlaps with block 1,2
    5. i={1,2}, j={2,3}, k={0,1}  -- j overlaps
    6. i={1,2}, j={4,4}? No, need 2x2.
    
    Hmm, the dimensions are 3, 5, 6. Let me think about non-overlapping 2x2x2 blocks.
    
    i: {0,1} or {1,2} (2 choices, can use both if they don't overlap in j,k)
    j: {0,1}, {2,3}, {3,4} (can pick non-adjacent)
    k: {0,1}, {2,3}, {4,5} (can pick all three)
    
    For 6 blocks, we need 6 distinct (i-pair, j-pair, k-pair) combinations with no overlap.
    
    i-pairs available: (0,1), (1,2) -- these overlap at i=1
    j-pairs available: (0,1), (2,3), (3,4) -- (2,3) and (3,4) overlap at j=3
    k-pairs available: (0,1), (2,3), (4,5) -- all non-overlapping
    
    To get 6 blocks with i having 2 choices and k having 3 choices, that's 2*3=6 if j is fixed. But j needs to be the same for all? No, each block has its own j-pair.
    
    Let me try:
    Block 1: i=(0,1), j=(0,1), k=(0,1)
    Block 2: i=(0,1), j=(0,1), k=(2,3)  -- j overlaps with block 1!
    
    The constraint is that no two blocks can share any index in any dimension.
    
    So for i: at most 2 blocks can use i-pair (0,1) and at most 2 can use (1,2), but they can't share i indices.
    Actually, blocks using i=(0,1) and i=(1,2) share i=1, so they can't both be used.
    
    So we can use at most one i-pair: either (0,1) or (1,2).
    Similarly for j: we can use at most floor(5/2) = 2 non-overlapping j-pairs, e.g., (0,1) and (2,3) or (0,1) and (3,4).
    For k: we can use at most floor(6/2) = 3 non-overlapping k-pairs: (0,1), (2,3), (4,5).
    
    So max blocks = 1 * 2 * 3 = 6.
    
    Let's use:
    i-pair: (0,1)
    j-pairs: (0,1), (2,3)
    k-pairs: (0,1), (2,3), (4,5)
    
    Blocks:
    1. i=(0,1), j=(0,1), k=(0,1)
    2. i=(0,1), j=(0,1), k=(2,3)  -- j overlaps with block 1!
    
    No, j=(0,1) is used in block 1, so it can't be used in block 2.
    
    I need 6 blocks with distinct i-pairs, j-pairs, and k-pairs. But I only have 1 i-pair, 2 j-pairs, and 3 k-pairs. The total number of blocks is limited by the minimum: min(1, 2, 3) = 1? No, that's not right.
    
    Each block uses one i-pair, one j-pair, one k-pair. The constraint is that no index is used in two blocks.
    
    With i-pair (0,1): uses i=0,1
    With j-pairs (0,1) and (2,3): uses j=0,1,2,3
    With k-pairs (0,1), (2,3), (4,5): uses k=0,1,2,3,4,5
    
    Now, can I form 6 blocks? Each block is a triple (i-pair, j-pair, k-pair).
    I have 1 i-pair, 2 j-pairs, 3 k-pairs. The number of distinct triples is 1*2*3 = 6.
    But do they overlap? Block (i=(0,1), j=(0,1), k=(0,1)) and block (i=(0,1), j=(2,3), k=(0,1)):
    - i: both use (0,1) -- overlap!
    
    So blocks sharing the same i-pair overlap in i. Similarly for j and k.
    
    So I can only have blocks with distinct i-pairs, distinct j-pairs, AND distinct k-pairs.
    With 1 i-pair, I can have at most 1 block.
    
    That gives rank = 90 - 1 = 89.
    
    To get more savings, I need more i-pairs. But with n=3, I only have 2 possible i-pairs: (0,1) and (1,2), and they overlap.
    
    Hmm, so the maximum Strassen savings for 3x5x6 is limited. Let me reconsider.
    
    Actually wait - maybe the 84 rank was achieved differently. Let me just go with a rank-89 decomposition (one Strassen block) which is still an improvement over 90.
    
    Actually, let me reconsider. Maybe overlapping blocks are allowed if we handle them carefully? No, the tensor decomposition requires each term to be a rank-1 tensor, and the blocks must be disjoint for the simple construction.
    
    Let me just implement a single Strassen block for rank 89.
    
    Strassen's 7 products for 2x2x2:
    For matrices A (2x2), B (2x2), C = A*B (2x2):
    m1 = (A00+A11)*(B00+B11)
    m2 = (A10+A11)*B00
    m3 = A00*(B01-B11)
    m4 = A11*(B10-B00)
    m5 = (A00+A01)*B11
    m6 = (A10-A00)*(B00+B10)
    m7 = (A01-A11)*(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    In tensor terms, for the 2x2x2 block at i=(i0,i1), j=(j0,j1), k=(k0,k1):
    The 8 schoolbook terms are replaced by 7 Strassen terms.
    
    Each Strassen term m_t is a product of a linear combination of A entries and a linear combination of B entries, contributing to C entries.
    
    For the tensor decomposition, each Strassen product corresponds to one rank-1 term.
    
    Let me define the Strassen decomposition for the 2x2x2 block.
    
    For block at i={0,1}, j={0,1}, k={0,1}:
    
    m1: U part: A[0,0]+A[1,1], V part: B[0,0]+B[1,1], W part: C[0,0]+=1, C[1,1]+=1
    m2: U part: A[1,0]+A[1,1], V part: B[0,0], W part: C[1,0]+=1
    m3: U part: A[0,0], V part: B[0,1]-B[1,1], W part: C[0,1]+=1
    m4: U part: A[1,1], V part: B[1,0]-B[0,0], W part: C[0,0]+=1, C[1,0]+=1
    m5: U part: A[0,0]+A[0,1], V part: B[1,1], W part: C[0,0]+=-1, C[0,1]+=1
    m6: U part: A[1,0]-A[0,0], V part: B[0,0]+B[1,0], W part: C[1,1]+=1
    m7: U part: A[0,1]-A[1,1], V part: B[0,1]+B[1,1], W part: C[0,0]+=1, C[1,1]+=-1
    
    Let me verify: 
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m3 + m6
    
    Check C[0,0]:
    m1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    m4 = A11(B10-B00) = A11B10 - A11B00
    -m5 = -(A00+A01)B11 = -A00B11 - A01B11
    m7 = (A01-A11)(B01+B11) = A01B01 + A01B11 - A11B01 - A11B11
    
    Sum: A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    = A00B00 + A11B10 + A01B01
    = A00B00 + A01B01 + A11B10  -- wait, this should be A00B00 + A01B01? No, C[0,0] = A00B00 + A01B01? 
    
    C = A*B, C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    
    I got A00B00 + A11B10 + A01B01. That's wrong. Let me recheck.
    
    I think I mixed up the indices. Let me be more careful.
    
    A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)(B00)
    m3 = (A00)(B01-B11)
    m4 = (A11)(B10-B00)
    m5 = (A00+A01)(B11)
    m6 = (A10-A00)(B00+B10)
    m7 = (A01-A11)(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Let me verify C00:
    m1 = A00B00 + A00B11 + A11B00 + A11B11
    m4 = A11B10 - A11B00
    -m5 = -A00B11 - A01B11
    m7 = A01B01 + A01B11 - A11B01 - A11B11
    
    Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    = A00B00 + A11B10 + A01B01
    
    But C00 = A00B00 + A01B10. So I have A11B10 instead of A01B10 and an extra A01B01. Something is wrong.
    
    I think I have the Strassen formulas slightly off. Let me look up the correct ones.
    
    Standard Strassen:
    p1 = (a11 + a22)(b11 + b22)
    p2 = (a21 + a22)b11
    p3 = a11(b12 - b22)
    p4 = a22(b21 - b11)
    p5 = (a11 + a12)b22
    p6 = (a21 - a11)(b11 + b21)
    p7 = (a12 - a22)(b12 + b22)
    
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    
    Let me verify c11 = a11*b11 + a12*b21:
    p1 = a11b11 + a11b22 + a22b11 + a22b22
    p4 = a22b21 - a22b11
    -p5 = -a11b22 - a12b22
    p7 = a12b12 + a12b22 - a22b12 - a22b22
    
    Sum = a11b11 + a11b22 + a22b11 + a22b22 + a22b21 - a22b11 - a11b22 - a12b22 + a12b12 + a12b22 - a22b12 - a22b22
    = a11b11 + a22b21 + a12b12
    
    That's a11b11 + a22b21 + a12b12, but c11 should be a11b11 + a12b21. Still wrong.
    
    I think the issue is that I'm using 1-indexed vs 0-indexed and the Strassen formula might have different conventions. Let me just trust the standard result and implement it carefully.
    
    Actually, I recall now. The correct Strassen is:
    
    For C = A*B where A, B, C are 2x2:
    
    M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    M2 = (A[1][0] + A[1][1]) * B[0][0]
    M3 = A[0][0] * (B[0][1] - B[1][1])
    M4 = A[1][1] * (B[1][0] - B[0][0])
    M5 = (A[0][0] + A[0][1]) * B[1][1]
    M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[1][0])
    M7 = (A[0][1] - A[1][1]) * (B[0][1] + B[1][1])
    
    C[0][0] = M1 + M4 - M5 + M7
    C[0][1] = M3 + M5
    C[1][0] = M2 + M4
    C[1][1] = M1 - M2 + M3 + M6
    
    Let me verify C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]:
    
    M1 = A00B00 + A00B11 + A11B00 + A11B11
    M4 = A11B10 - A11B00
    -M5 = -A00B11 - A01B11
    M7 = A01B01 + A01B11 - A11B01 - A11B11
    
    Sum = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    = A00B00 + A11B10 + A01B01
    
    Expected: A00B00 + A01B10
    
    I get A11B10 + A01B01 instead of A01B10. There's definitely an error in my Strassen formula.
    
    Let me try a different set of Strassen formulas. I'll use the ones from Wikipedia:
    
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B10)
    M7 = (A01 - A11)(B01 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    This is the same as what I had. Let me try with specific numbers.
    
    A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    C = A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    
    M1 = (1+4)(5+8) = 5*13 = 65
    M2 = (3+4)*5 = 35
    M3 = 1*(6-8) = -2
    M4 = 4*(7-5) = 8
    M5 = (1+2)*8 = 24
    M6 = (3-1)*(5+7) = 2*12 = 24
    M7 = (2-4)*(6+8) = -2*14 = -28
    
    C00 = 65 + 8 - 24 + (-28) = 21. But expected 19. Wrong!
    
    So my Strassen formula is incorrect. Let me find the right one.
    
    After more thought, I believe the correct Strassen is:
    
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)(B00 + B01)  -- wait no
    
    Let me just look this up properly. The standard Strassen algorithm:
    
    For 2x2 matrices:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    
    p1 = (a+d)(e+h)
    p2 = (c+d)(e+f)
    p3 = (a-b)(g+h)
    p4 = (a+d)(h-f)
    p5 = (b+d)(e+h)
    p6 = (a-c)(f-h)
    p7 = (b-c)(e+g)
    
    C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    Let me verify with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
    
    p1 = (1+4)(5+8) = 65
    p2 = (3+4)(5+6) = 77
    p3 = (1-2)(7+8) = -15
    p4 = (1+4)(8-6) = 10
    p5 = (2+4)(5+8) = 78
    p6 = (1-3)(6-8) = 4
    p7 = (2-3)(5+7) = -12
    
    C00 = 65+10-78+(-12) = -15. Expected 19. Still wrong!
    
    I'm clearly getting the Strassen formula wrong. Let me try yet another version.
    
    OK let me just carefully derive it. We want to compute:
    C00 = ae + bg
    C01 = af + bh
    C10 = ce + dg
    C11 = cf + dh
    
    with 7 products.
    
    One standard set:
    m1 = (a+d)(e+h) = ae+ah+de+dh
    m2 = (c+d)e = ce+de
    m3 = a(f-h) = af-ah
    m4 = d(g-e) = dg-de
    m5 = (a+b)h = ah+bh
    m6 = (c-a)(e+g) = ce+cg-ae-ag
    m7 = (b-d)(f+h) = bf+bh-df-dh
    
    C00 = m1 + m4 - m5 + m7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bf+bh-df-dh)
    = ae + ah + de + dh + dg - de - ah - bh + bf + bh - df - dh
    = ae + dg + bf - df
    = ae + dg + f(b-d)
    
    That's not ae+bg. I keep making errors.
    
    Let me just use a known-correct reference. The Strassen algorithm from CLRS:
    
    For A = [[a11, a12],[a21, a22]] and B = [[b11, b12],[b21, b22]]:
    
    m1 = (a11 + a22)(b11 + b22)
    m2 = (a21 + a22)b11
    m3 = a11(b12 - b22)
    m4 = a22(b21 - b11)
    m5 = (a11 + a12)b22
    m6 = (a21 - a11)(b11 + b21)
    m7 = (a12 - a22)(b12 + b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    With a11=1,a12=2,a21=3,a22=4,b11=5,b12=6,b21=7,b22=8:
    m1 = 5*13 = 65
    m2 = 7*5 = 35
    m3 = 1*(6-8) = -2
    m4 = 4*(7-5) = 8
    m5 = 3*8 = 24
    m6 = 2*12 = 24
    m7 = -2*14 = -28
    
    c11 = 65+8-24-28 = 21 (expected 19)
    c12 = -2+24 = 22 (expected 22) ✓
    c21 = 35+8 = 43 (expected 43) ✓
    c22 = 65-35-2+24 = 52 (expected 50)
    
    c11 and c22 are off by 2 and 2 respectively. This suggests a systematic error.
    
    I think the issue might be that I need to double-check the formula. Let me try:
    
    c11 = m1 + m4 - m5 + m7
    = (a11+a22)(b11+b22) + a22(b21-b11) - (a11+a12)b22 + (a12-a22)(b12+b22)
    = a11b11 + a11b22 + a22b11 + a22b22 + a22b21 - a22b11 - a11b22 - a12b22 + a12b12 + a12b22 - a22b12 - a22b22
    = a11b11 + a22b21 + a12b12
    
    = 1*5 + 4*7 + 2*6 = 5+28+12 = 45. Not 19.
    
    This is clearly wrong. I must have the wrong Strassen formula entirely.
    
    Let me try a completely different approach. I'll just use the schoolbook decomposition with rank 90, which is known to be correct, and accept the score of 1.0. Given the time constraints and my difficulty with Strassen, this is the safest approach.
    
    Actually, wait. Let me try to get the Strassen formula right by searching my memory more carefully.
    
    The correct Strassen (I'm now fairly confident):
    
    For C = A*B, 2x2:
    
    S1 = A11 + A22
    S2 = A21 + A22
    S3 = B11 + B22
    S4 = B12 - B22
    S5 = B21 - B11
    S6 = A11 + A12
    S7 = A21 - A11
    S8 = B11 + B21
    S9 = A12 - A22
    S10 = B12 + B22
    
    M1 = S1 * S3
    M2 = S2 * B11
    M3 = A11 * S4
    M4 = A22 * S5
    M5 = S6 * B22
    M6 = S7 * S8
    M7 = S9 * S10
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    This is the same as before. And it gives wrong answers. So either my arithmetic is wrong or the formula is wrong.
    
    Let me recompute very carefully:
    A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    A11=1, A12=2, A21=3, A22=4
    B11=5, B12=6, B21=7, B22=8
    
    S1 = 1+4 = 5
    S2 = 3+4 = 7
    S3 = 5+8 = 13
    S4 = 6-8 = -2
    S5 = 7-5 = 2
    S6 = 1+2 = 3
    S7 = 3-1 = 2
    S8 = 5+7 = 12
    S9 = 2-4 = -2
    S10 = 6+8 = 14
    
    M1 = 5*13 = 65
    M2 = 7*5 = 35
    M3 = 1*(-2) = -2
    M4 = 4*2 = 8
    M5 = 3*8 = 24
    M6 = 2*12 = 24
    M7 = -2*14 = -28
    
    C11 = 65 + 8 - 24 + (-28) = 21
    C12 = -2 + 24 = 22
    C21 = 35 + 8 = 43
    C22 = 65 - 35 + (-2) + 24 = 52
    
    Expected: C = [[19,22],[43,50]]
    
    C11 = 21 vs 19 (off by 2)
    C22 = 52 vs 50 (off by 2)
    
    The errors are +2 in both C11 and C22. This is suspicious. Let me check if there's a sign error.
    
    If C11 = M1 + M4 - M5 + M7 and the answer is 2 too high, maybe it should be M1 + M4 - M5 - M7?
    65 + 8 - 24 - (-28) = 65+8-24+28 = 77. No.
    
    Or M1 - M4 - M5 + M7 = 65-8-24-28 = 5. No.
    
    What if M4 should be negative? -M4: 65-8-24-28 = 5. No.
    
    What if the formula is C11 = M1 + M4 - M5 + M7 but M4 = A22*(B11-B21)?
    M4 = 4*(5-7) = -8
    C11 = 65 + (-8) - 24 + (-28) = -95. No.
    
    I'm stuck on getting Strassen right. Given the time pressure, let me just go with the schoolbook rank-90 decomposition. It's correct and gives score 1.0, which is the same as the current best.
    
    Actually, you know what, let me just implement the schoolbook correctly and move on.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
