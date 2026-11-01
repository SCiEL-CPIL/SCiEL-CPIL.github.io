"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (6, 30, 20)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want rank-minimal decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # Strategy: Block-Kronecker structure
    # Decompose the 3x10 multiplication by splitting the 10 columns into blocks
    # Block 1: columns 0-1 (2 cols) -> treat as 3x2 multiplication
    # Blocks 2-5: columns 2-9 (8 cols) -> treat as 3x2 multiplication for each pair
    
    # For a 3x2 matrix multiplication, we can use:
    # Schoolbook: rank 3*2 = 6
    # Or we can try to find a better decomposition
    
    # Actually, let's think about this differently.
    # The full problem is 2x3 * 3x10 = 2x10
    # Schoolbook rank = 2*3*10 = 60
    
    # Block-Kronecker approach:
    # Split the 10 columns into 5 groups of 2
    # For each group of 2 columns, we compute 2x3 * 3x2 = 2x2
    # For 2x2 multiplication, Strassen gives rank 7 (instead of 4 schoolbook)
    # But we need to be careful: the 3x2 middle part is shared across rows
    
    # Actually, the tensor structure is:
    # For each (i, j, k): A[i,j] * B[j,k] contributes to C[i,k]
    # i in 0..1, j in 0..2, k in 0..9
    
    # Let's split k into 5 blocks of 2: k_block in 0..4, k_local in 0..1
    # For each block, we have a 2x3 * 3x2 sub-problem
    # The A matrix is shared (2x3), B is block-diagonal in k
    
    # For 2x3 * 3x2 -> 2x2, schoolbook rank = 2*3*2 = 12
    # Can we do better? 
    # 
    # Actually, let's use a different approach. Let's split the j dimension (3 rows of B / 3 cols of A).
    # 
    # Alternative: Split j into blocks. j in 0..2.
    # Block j=0: A[:,0] * B[0,:] -> rank 2*1*10 = 20 (schoolbook)
    # Block j=1: A[:,1] * B[1,:] -> rank 2*1*10 = 20
    # Block j=2: A[:,2] * B[2,:] -> rank 2*1*10 = 20
    # Total: 60 (same as schoolbook)
    
    # Let's try the block-Kronecker on k dimension with 2x2 Strassen:
    # For each block of 2 k-values, we compute 2x3 * 3x2 = 2x2
    # The 2x3 * 3x2 problem: can we decompose it with rank < 12?
    # 
    # 2x3 * 3x2: This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    # We can write this as: C = A * B where A is 2x3, B is 3x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # This is a 2x3x2x2 tensor. Schoolbook rank = 2*3*2 = 12.
    # Can we do better? The rank of matrix multiplication <2,3,2> is at most 12.
    # Known results: rank of <2,3,2> is 12 (I believe it's not known to be less).
    
    # Let me try a different block structure. What if we use Strassen on the 2x2 output?
    # For each pair of k values, we have a 2x2 output. Strassen for 2x2 uses 7 multiplications.
    # But Strassen requires the inputs to be 2x2. Our inputs are 2x3 and 3x2.
    
    # Let's try: for each block of 2 k-values, compute the 2x2 product using a rank-7 scheme
    # adapted from Strassen. The key idea:
    # M1 = (A[0,:] + A[1,:]) * (B[:,0] + B[:,1])  -- but these are vectors, not matrices
    # This doesn't directly work because A[0,:] is 1x3 and B[:,0] is 3x1.
    
    # Actually, let's think of it as: for a fixed k-block, we have:
    # C_block = A (2x3) * B_block (3x2)
    # We can use the identity:
    # A*B = (A + I*B_perm) ... this gets complicated.
    
    # Let me try a direct approach: use a known low-rank decomposition for <2,3,2>.
    # Actually, I'll try to construct one manually or use a search.
    
    # For now, let's use the schoolbook for each block: 5 * 12 = 60. Same as before.
    
    # Let's try splitting differently. What about using the fact that n=2?
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # For each k, this is a 2x3 * 3x1 product. Rank per k = 3 (schoolbook).
    # Total = 10*3 = 30? No wait, that's not right either because the decomposition
    # needs to be a sum of rank-1 tensors over the full (i,j,k) space.
    
    # Let me reconsider. The tensor T has entries T[(i,j), (j,k), (k,i)] = 1.
    # A rank-1 decomposition term is u[(i,j)] * v[(j,k)] * w[(k,i)].
    # 
    # Schoolbook: for each (i,j,k), term = e_{i,j} * e_{j,k} * e_{k,i}. Rank = 60.
    
    # Can we combine terms? For fixed i, k: sum over j of A[i,j]*B[j,k].
    # The contribution to C[i,k] is sum_j A[i,j]*B[j,k].
    # In the tensor, this means T has 1's at positions where the indices match.
    
    # Let's try: for each pair (i,k), the sum over j can be done with rank 3 (schoolbook for 1x3*3x1).
    # But we need the same decomposition to work for all (i,j,k) simultaneously.
    
    # Actually, the key insight: the tensor is a direct sum over k of 2x3x1 tensors (one for each k).
    # No wait, it's not a direct sum because j is shared.
    
    # Let me just try to find a good decomposition by search.
    
    # First, let's establish the schoolbook baseline of rank 60.
    rank_schoolbook = n * m * p  # 60
    
    # Now let's try to find a better decomposition.
    # Strategy: Try to use a block structure where we split k into groups.
    
    # For each group of k values, we have a sub-tensor. If we use rank r_g for group g,
    # total rank = sum_g r_g.
    
    # Let's try: split k into 5 groups of 2. For each group, we need to decompose
    # the 2x3*3x2 sub-problem.
    
    # For 2x3*3x2, let's try to find a decomposition of rank < 12.
    # I'll do a small search.
    
    # Actually, let me try a specific construction. For 2x2x2x2 (i.e., 2x2*2x2), Strassen gives rank 7.
    # For 2x3*3x2, we can embed this.
    
    # Here's an idea: 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let's define:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,1]*B[1,0]
    # M3 = A[0,2]*B[2,0]
    # M4 = A[0,0]*B[0,1]
    # M5 = A[0,1]*B[1,1]
    # M6 = A[0,2]*B[2,1]
    # M7 = A[1,0]*B[0,0]
    # M8 = A[1,1]*B[1,0]
    # M9 = A[1,2]*B[2,0]
    # M10 = A[1,0]*B[0,1]
    # M11 = A[1,1]*B[1,1]
    # M12 = A[1,2]*B[2,1]
    
    # That's 12 terms (schoolbook). Can we combine?
    # C[0,0] = M1 + M2 + M3
    # C[0,1] = M4 + M5 + M6
    # C[1,0] = M7 + M8 + M9
    # C[1,1] = M10 + M11 + M12
    
    # What if we try:
    # N1 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1]) = A[0,0]B[0,0] + A[0,0]B[0,1] + A[1,0]B[0,0] + A[1,0]B[0,1]
    # N2 = (A[0,1]+A[1,1])*(B[1,0]+B[1,1]) = A[0,1]B[1,0] + A[0,1]B[1,1] + A[1,1]B[1,0] + A[1,1]B[1,1]
    # N3 = (A[0,2]+A[1,2])*(B[2,0]+B[2,1]) = A[0,2]B[2,0] + A[0,2]B[2,1] + A[1,2]B[2,0] + A[1,2]B[2,1]
    
    # N1 + N2 + N3 = sum of all 12 terms
    # C[0,0] + C[0,1] + C[1,0] + C[1,1] = N1 + N2 + N3
    
    # We need to extract individual C entries. Let's define more combinations:
    # N4 = (A[0,0]-A[1,0])*(B[0,0]+B[0,1]) = A[0,0]B[0,0] + A[0,0]B[0,1] - A[1,0]B[0,0] - A[1,0]B[0,1]
    # N5 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) = A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1]
    # N6 = (A[0,2]-A[1,2])*(B[2,0]+B[2,1]) = A[0,2]B[2,0] + A[0,2]B[2,1] - A[1,2]B[2,0] - A[1,2]B[2,1]
    
    # N7 = (A[0,0]+A[1,0])*(B[0,0]-B[0,1]) = A[0,0]B[0,0] - A[0,0]B[0,1] + A[1,0]B[0,0] - A[1,0]B[0,1]
    # N8 = (A[0,1]+A[1,1])*(B[1,0]-B[1,1]) = A[0,1]B[1,0] - A[0,1]B[1,1] + A[1,1]B[1,0] - A[1,1]B[1,1]
    # N9 = (A[0,2]+A[1,2])*(B[2,0]-B[2,1]) = A[0,2]B[2,0] - A[0,2]B[2,1] + A[1,2]B[2,0] - A[1,2]B[2,1]
    
    # N10 = (A[0,0]-A[1,0])*(B[0,0]-B[0,1]) = A[0,0]B[0,0] - A[0,0]B[0,1] - A[1,0]B[0,0] + A[1,0]B[0,1]
    # N11 = (A[0,1]-A[1,1])*(B[1,0]-B[1,1]) = A[0,1]B[1,0] - A[0,1]B[1,1] - A[1,1]B[1,0] + A[1,1]B[1,1]
    # N12 = (A[0,2]-A[1,2])*(B[2,0]-B[2,1]) = A[0,2]B[2,0] - A[0,2]B[2,1] - A[1,2]B[2,0] + A[1,2]B[2,1]
    
    # This is getting complex. Let me just use 12 terms per block (schoolbook) = 60 total.
    
    # Let me try a different approach: use a randomized search to find a low-rank decomposition.
    
    # Actually, let me try the approach from the knowledge: 5 blocks of rank 11 each = 55.
    # This suggests a rank-11 decomposition for 2x3*3x2.
    
    # Let me try to construct a rank-11 decomposition for 2x3*3x2.
    # 
    # For 2x2*2x2, Strassen uses 7 multiplications. For 2x3*3x2, maybe we can use
    # a similar idea.
    
    # Here's a rank-11 construction for 2x3*3x2:
    # We have 4 output entries: C00, C01, C10, C11
    # Each is a sum of 3 products.
    
    # Let's try:
    # T1 = A[0,0]*B[0,0]
    # T2 = A[0,1]*B[1,0]  
    # T3 = A[0,2]*B[2,0]
    # T4 = A[0,0]*B[0,1]
    # T5 = A[0,1]*B[1,1]
    # T6 = A[0,2]*B[2,1]
    # T7 = A[1,0]*B[0,0]
    # T8 = A[1,1]*B[1,0]
    # T9 = A[1,2]*B[2,0]
    # T10 = (A[0,0]+A[1,0]+A[0,1]+A[1,1]+A[0,2]+A[1,2])*(B[0,0]+B[1,0]+B[2,0]+B[0,1]+B[1,1]+B[2,1])
    #     = sum of all 12 products
    # T11 = T10 - T1 - T2 - T3 - T4 - T5 - T6 - T7 - T8 - T9 = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # So:
    # C00 = T1 + T2 + T3
    # C01 = T4 + T5 + T6
    # C10 = T7 + T8 + T9
    # C11 = T11
    
    # That's 11 terms! Let me verify:
    # T11 = (sum A)*(sum B) - sum of the 9 other products
    # = (A00+A01+A02+A10+A11+A12)*(B00+B10+B20+B01+B11+B21) - (A00B00+A01B10+A02B20+A00B01+A01B11+A02B21+A10B00+A11B10+A12B20)
    # = A10B01 + A11B11 + A12B21 + (other cross terms that should cancel)
    
    # Wait, let me be more careful:
    # (sum A)*(sum B) = A00B00 + A00B10 + A00B20 + A00B01 + A00B11 + A00B21
    #                  + A01B00 + A01B10 + A01B20 + A01B01 + A01B11 + A01B21
    #                  + A02B00 + A02B10 + A02B20 + A02B01 + A02B11 + A02B21
    #                  + A10B00 + A10B10 + A10B20 + A10B01 + A10B11 + A10B21
    #                  + A11B00 + A11B10 + A11B20 + A11B01 + A11B11 + A11B21
    #                  + A12B00 + A12B10 + A12B20 + A12B01 + A12B11 + A12B21
    
    # The 9 terms we subtract are:
    # A00B00, A01B10, A02B20, A00B01, A01B11, A02B21, A10B00, A11B10, A12B20
    
    # What remains:
    # A00B10 + A00B20 + A00B11 + A00B21
    # A01B00 + A01B20 + A01B01 + A01B21
    # A02B00 + A02B10 + A02B01 + A02B11
    # A10B10 + A10B20 + A10B01 + A10B11 + A10B21
    # A11B00 + A11B20 + A11B01 + A11B11 + A11B21
    # A12B00 + A12B10 + A12B01 + A12B11 + A12B21
    
    # That's not just A10B01 + A11B11 + A12B21. So my construction is wrong.
    
    # Let me try a different approach. For rank 11, I need 11 bilinear forms.
    
    # Actually, let me just use the schoolbook for each block (rank 12 per block, 60 total)
    # and then try to optimize.
    
    # Let me try a simple randomized search for a better decomposition.
    
    # Build the tensor T as a 6x30x20 array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition using random initialization and gradient-free optimization
    # This is complex, so let me just use the known result: 5 blocks of rank 11 = 55.
    
    # Let me construct a rank-55 decomposition using the block structure.
    # For each block of 2 k-values, use rank 11.
    
    # For 2x3*3x2 with rank 11, here's a valid construction:
    # We can use the fact that <2,3,2> has rank at most 11 (I'll verify this).
    
    # Actually, let me just use schoolbook for now (rank 60) and see if I can improve.
    
    # Let me try: for each k, use rank 3 (the 3 j-values). Total = 10*3 = 30.
    # But this doesn't work because the decomposition must be consistent across all (i,j,k).
    
    # The correct schoolbook is rank 60. Let me just return that.
    
    # Actually, wait. Let me re-read the problem. The tensor is:
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    # A rank-1 term is u[a]*v[b]*w[c].
    # For the schoolbook, each term corresponds to a specific (i,j,k):
    # u[a] = 1 if a = i*m+j else 0
    # v[b] = 1 if b = j*p+k else 0
    # w[c] = 1 if c = k*n+i else 0
    
    # Total rank = n*m*p = 60.
    
    # Can we do better? The rank of <2,3,10> is at most 60.
    # Lower bounds: rank >= n*m*p / max(n*m, m*p, p*n) ... not a standard bound.
    # rank >= max(n*m, m*p, p*n) = max(6, 30, 20) = 30 (very weak).
    
    # Let me try a block decomposition. Split k into 5 groups of 2.
    # For each group, the sub-tensor involves indices where k is in that group.
    # The sub-tensor for group g (k in {2g, 2g+1}) has entries:
    # T[i*m+j, j*p+2g, (2g)*n+i] = 1
    # T[i*m+j, j*p+2g+1, (2g+1)*n+i] = 1
    # for i in 0..1, j in 0..2
    
    # This sub-tensor has 2*3*2 = 12 nonzero entries.
    # Its rank is at most 12 (schoolbook).
    
    # If I can decompose each sub-tensor with rank 11, total rank = 55.
    
    # Let me try to find a rank-11 decomposition for the 2x3*3x2 sub-problem.
    
    # For the sub-problem with k in {0,1} (for simplicity):
    # Output: C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    
    # Rank-11 construction:
    # Use 9 schoolbook terms for C00, C01, C10, and 2 additional terms for C11.
    # T1 = A[0,0]*B[0,0]
    # T2 = A[0,1]*B[1,0]
    # T3 = A[0,2]*B[2,0]
    # T4 = A[0,0]*B[0,1]
    # T5 = A[0,1]*B[1,1]
    # T6 = A[0,2]*B[2,1]
    # T7 = A[1,0]*B[0,0]
    # T8 = A[1,1]*B[1,0]
    # T9 = A[1,2]*B[2,0]
    # T10 = A[1,0]*B[0,1]
    # T11 = A[1,1]*B[1,1] + A[1,2]*B[2,1]  -- but this is not rank 1!
    
    # T11 is a sum of two products, so it's not a valid rank-1 term.
    
    # I need each term to be a single product u[a]*v[b]*w[c].
    
    # So T10 = A[1,0]*B[0,1] (rank 1)
    # T11 = A[1,1]*B[1,1] (rank 1)
    # Then C11 = T10 + T11 + A[1,2]*B[2,1] -- but I've used up my 11 terms.
    
    # I need 12 terms for schoolbook. To get 11, I need to combine two terms into one.
    
    # What if I use:
    # T11 = (A[1,1]+A[1,2])*(B[1,1]+B[2,1])
    #     = A[1,1]B[1,1] + A[1,1]B[2,1] + A[1,2]B[1,1] + A[1,2]B[2,1]
    
    # Then C11 = T10 + T11 - A[1,1]B[2,1] - A[1,2]B[1,1]
    # But I don't have separate terms for A[1,1]B[2,1] and A[1,2]B[1,1].
    
    # This approach is getting too complex. Let me just use schoolbook (rank 60).
    
    # Actually, let me try a completely different approach: use a randomized search.
    
    # For a tensor of size 6x30x20, finding a low-rank decomposition is hard.
    # Let me just return the schoolbook decomposition.
    
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
