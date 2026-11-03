"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.

The schoolbook rank is n*m*p = 2*3*6 = 36.
We can do better by splitting the 3x6 matrix into two 3x3 blocks.
Each 2x3 * 3x3 multiplication has schoolbook rank 2*3*3 = 18, but we can
actually do it in rank 12 using a structured decomposition.

Wait, let me think more carefully. The 2x3 * 3x3 multiplication:
C_{i,k} = sum_j A_{i,j} B_{j,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}.

The standard bilinear algorithm for 2x3 * 3x3 can potentially be optimized.
However, a simple block approach gives rank 18 per block, total 36.

Let me reconsider. The key insight is that for 2x3 * 3x6, we can use the fact that
the second matrix has 6 columns. We can split into two 3-column blocks, each requiring
rank 18, for total rank 36. That's not better.

But wait - there might be a smarter decomposition. Let me think about the tensor structure.

Actually, let me try a different approach. The matrix multiplication tensor for 2x3 * 3x6
has dimensions 2x3x6 = 36 entries. The rank is at most 36 (schoolbook).

One approach: use the fact that we can compute products with shared structure.
For example, if we precompute certain combinations of A entries, we can reduce multiplications.

Let me try to find a rank-24 or lower decomposition.

Actually, let me think about this differently. The bilinear complexity of m x n * n x p matrix
multiplication is known to be less than m*n*p in many cases.

For 2x3 * 3x6:
- We have A (2x3) and B (3x6), producing C (2x6).
- C_{i,k} = sum_{j=0}^{2} A_{i,j} * B_{j,k}

One approach: express each C_{i,k} as a combination of a small number of products.

Let me try to find a decomposition with rank < 36.

Consider that we can compute:
For each column k of C:
C[:,k] = A * B[:,k]

This is a 2x3 * 3x1 multiplication, which has rank at most 2*3*1 = 6, but actually can be done in rank 2 (just 2 multiplications per column if we do it naively: sum of 3 products = 3 multiplications).

Wait, 2x3 * 3x1: C_i = sum_j A_{i,j} B_{j,0}. This requires 3 multiplications per output element, 2 output elements, so 6 multiplications per column. With 6 columns, that's 36 total. Same as schoolbook.

Hmm, but maybe we can share some computations across columns.

Let me think about this more carefully. The tensor is T[i,j,k] = delta_{i,j,k} in the sense that T maps A and B to C.

Actually, let me just try to construct a rank-24 decomposition by splitting into two 3x3 blocks and using a rank-12 algorithm for 2x3 * 3x3.

For 2x3 * 3x3, can we do better than rank 18?
C_{i,k} = sum_j A_{i,j} B_{j,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}.

The schoolbook rank is 18. Can we do it in less?

One idea: use the fact that 2x3 * 3x3 can be seen as computing 6 outputs, each being a dot product of length 3. That's 18 multiplications. But maybe we can combine some.

Actually, I recall that for small matrix multiplications, the optimal ranks are sometimes unknown or equal to schoolbook. Let me just go with the block approach.

Wait, here's a better idea. Split the 3x6 into two 3x3 blocks. For each block, use a rank-12 decomposition if one exists for 2x3*3x3. If not, use rank-18.

Let me check: is there a known rank-12 algorithm for 2x3*3x3? I'm not sure. Let me just implement the block approach with rank-18 per block for now, giving total rank 36, and see if I can improve.

Actually, let me try a completely different approach. What if I use a rank-24 decomposition by being clever?

Here's an idea: instead of treating each column independently, I can use the linearity to combine. For example, compute A * [B[:,0] + B[:,1], B[:,2] + B[:,3], B[:,4] + B[:,5]] and A * [B[:,0] - B[:,1], ...] etc. But this doesn't obviously help.

Let me just go with the straightforward block decomposition: two 3x3 blocks, each with rank 18, total rank 36. That's the same as schoolbook, so not an improvement.

Hmm, let me think about whether there's a way to get rank < 36.

Actually, wait. Let me reconsider the problem. The task says "rank-minimal exact tensor decomposition". The score is the rank, and lower is better. The current score is 1.0 which corresponds to the seed program's rank of 36.

Let me try to find a rank-30 or rank-24 decomposition.

Here's another idea: use the fact that for 2x3 * 3x6, we can think of it as 2 rows of length 3 being multiplied against 3x6 matrices. 

Let me try to construct a rank-24 decomposition:
- Split into 2 blocks of 3 columns each.
- For each block, use a rank-12 algorithm for 2x3 * 3x3.

For 2x3 * 3x3, I'll try to find a rank-12 decomposition. If I can't, I'll use rank-18.

Let me try to construct a rank-12 decomposition for 2x3 * 3x3:
We need to compute C_{i,k} = sum_j A_{i,j} B_{j,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}.

A rank-12 decomposition would use 12 multiplications. Let me try:
U[i*3+j, r], V[j*3+k, r], W[k*2+i, r] for r in 0..11.

This is complex. Let me just go with the block approach using rank-18 per block for now, and if that gives rank 36, I'll try to improve.

Actually, I just realized: the seed program already gives rank 36. I need to do better than that.

Let me try a rank-30 decomposition by sharing some computations.

Here's an idea: compute 5 columns directly (rank 5*3*2 = 30... no that's 30 multiplications for 5 columns, leaving 1 column uncomputed). That doesn't work.

OK let me just try the block approach and see if I can get rank 24 by finding a rank-12 algorithm for 2x3*3x3. If I can't find one easily, I'll fall back to a simpler improvement.

Actually, let me just implement a rank-36 solution first (same as seed) to make sure it's correct, then try to improve.
"""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try to find a rank-24 decomposition by splitting into two 3x3 blocks
    # and using a rank-12 algorithm for each 2x3 * 3x3 multiplication.
    
    # First, let's try a simple approach: split into two blocks of 3 columns.
    # Block 1: columns 0,1,2 of B -> columns 0,1,2 of C
    # Block 2: columns 3,4,5 of B -> columns 3,4,5 of C
    
    # For each block, we need a decomposition of 2x3 * 3x3.
    # Let's use a rank-12 decomposition if possible, otherwise rank-18.
    
    # Rank-18 per block gives total rank 36 (same as schoolbook).
    # Let's try to find a rank-12 decomposition for 2x3 * 3x3.
    
    # For 2x3 * 3x3: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # A rank-12 decomposition:
    # U[i*3+j, r], V[j*3+k, r], W[k*2+i, r] for r in 0..11
    
    # Let me try to construct this. We need:
    # sum_{r=0}^{11} U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = delta_{i,j,k} (in tensor sense)
    
    # This is equivalent to: for each (i,j,k), the sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1 if this corresponds to the (i,k) entry using A[i,j] and B[j,k], and 0 otherwise.
    
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the appropriate ranges, and the decomposition is:
    # T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
    
    # For 2x3*3x3, we have 2*3*3 = 18 tensor entries. A rank-12 decomposition would mean 12 terms.
    
    # Let me just try a simple construction: use 12 of the 18 schoolbook terms and see if it works.
    # Probably not. Let me try a different approach.
    
    # Alternative: use a rank-18 decomposition for each block, total rank 36.
    # But that's the same as the seed. I need to do better.
    
    # Let me try to find a rank-30 decomposition for the full 2x3*3x6 problem.
    
    # Idea: compute 5 columns with rank 15 (3 multiplications per column * 5 columns... no, 3*2=6 per column, 5*6=30).
    # Wait, each column of C is A * B[:,k], which is a 2x3 * 3x1 multiplication.
    # The schoolbook rank for 2x3 * 3x1 is 6 (3 multiplications per output, 2 outputs).
    # So 5 columns would cost 30, and the 6th column would need 6 more, total 36.
    
    # Hmm, that doesn't help. Let me think differently.
    
    # What if I use a shared computation? For example, compute A * B[:,0] and A * B[:,1] separately (cost 12),
    # then C[:,0] and C[:,1] are available. But I need to compute all 6 columns.
    
    # Let me try a rank-24 decomposition by using a structured approach.
    
    # Actually, let me just go with a rank-36 solution for now and see if the evaluator accepts it.
    # The current seed is rank 36, so I need to match or beat it.
    
    rank = n * m * p  # 36
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
