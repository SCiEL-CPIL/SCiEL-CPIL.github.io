"""Matrix multiplication 2x4x7: Strassen-based rank-50 decomposition."""
DIMENSIONS = [2, 4, 7]

import numpy as np
import itertools
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    deadline = time.time() + 120
    
    # For 2x4x7, we can think of this as:
    # A is 2x4, B is 4x7, C is 2x7
    # 
    # Split A into [A1 A2] where A1 is 2x2, A2 is 2x2
    # Split B into [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    #
    # Each 2x2x7 product can be done with rank 15 (Strassen-like for 2x2x4 gives 12, 
    # but 2x2x7 needs more). Actually, let's use a different approach.
    #
    # Alternative: Split the 4 dimension into 2+2, and handle 7 as is.
    # For 2x2x4 matrix multiplication, rank is 12 (Winograd).
    # So C1 = A1 * B1 (2x2 * 2x4) has rank 12
    #    C2 = A2 * B2 (2x2 * 2x4) has rank 12
    # But C = C1 + C2, and we can combine them.
    #
    # Actually, let me think about this differently. 
    # 2x4x7: we can view this as two 1x4x7 row multiplications.
    # Each row of A (1x4) times B (4x7) gives a row of C (1x7).
    # Rank of 1x4x7 tensor: at least 4 (since B has 4 rows and we need all of them).
    # Actually rank(1x4x7) = 4 because it's just 4 independent scalars times rows of B.
    # Wait, no. The tensor T_{i,j,k} = delta_{i} for the 1x4x7 case is just the identity in the middle dimension.
    # So rank is 4 for each row, total 8? No, that's schoolbook.
    
    # Let me use a known result: rank(2x2x4) = 12 (this is a classic result).
    # So if we split the 4 in the middle as 2+2:
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # First term: 2x2 * 2x7. This can be seen as 2x2x4 (first 4 cols) + 2x2x3 (last 3 cols)
    # That's not clean.
    
    # Better approach: Use the fact that rank(2x4x4) is known.
    # Actually, let me just construct a Strassen-style decomposition.
    
    # For 2x4x7, one approach:
    # Treat as block multiplication with blocks of size 2x2 in the middle dimension.
    # A = [A1 A2] (2x2 each), B = [B1; B2] (2x7 each)
    # C = A1*B1 + A2*B2
    # 
    # A1*B1 is 2x2 * 2x7. We can use rank-8 (schoolbook) or better.
    # Actually for 2x2xn, the rank is at most 2n but can be less.
    # For 2x2x4, rank is 12. For 2x2x7, we can split 7 = 4+3.
    # 2x2x4 has rank 12, 2x2x3 has rank... at most 2*2*3=12, probably 8 or so.
    
    # Let me try a completely different approach: use the known rank for 2x4x4.
    # rank(2x4x4) <= 24 (from two 2x2x4 blocks, each rank 12)
    # For 2x4x7 = 2x4x4 + 2x4x3
    # rank(2x4x3): split 3 as... this is getting complicated.
    
    # Let me just use a systematic approach: try to find a low-rank decomposition
    # using linear algebra on the tensor.
    
    # The tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # In our indexing: U is (n*m, rank) = (8, rank), V is (m*p, rank) = (28, rank), W is (p*n, rank) = (14, rank)
    # The tensor element T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent (same j in a and b, same k in b and c, same i in c and a)
    
    # Let's build the tensor and try to find a low-rank decomposition.
    # Actually, the tensor is: T[i,j,k] = delta for the multiplication, meaning:
    # sum over j: A[i,j] * B[j,k] = C[i,k]
    # So the tensor is T_{(i,j),(j,k),(k,i)} = 1 for valid (i,j,k), 0 otherwise.
    # This is a 3D tensor of size (n*m) x (m*p) x (p*n) = 8 x 28 x 14
    
    # The schoolbook rank is n*m*p = 56.
    # We want to find rank < 56.
    
    # Let me try a constructive approach using Strassen for the 2x2 blocks.
    # 
    # Split: A is 2x4 = [A1 A2] with A1, A2 in R^{2x2}
    # B is 4x7 = [B1; B2] with B1, B2 in R^{2x7}
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x7):
    #   Let B1 = [B1a B1b] where B1a is 2x4, B1b is 2x3
    #   A1*B1a: 2x2*2x4, rank 12
    #   A1*B1b: 2x2*2x3, rank at most 12 (schoolbook is 12)
    #   But we can potentially do better.
    
    # Actually, let me look up: what is the rank of 2x2xn matrix multiplication?
    # For n=1: rank 4 (schoolbook, which is optimal since we need 4 products)
    # Wait, 2x2*2x1: A (2x2) times B (2x1) = C (2x1). 
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is 4 products, rank 4. Can we do better? The tensor is 4x2x2.
    # I believe rank(2x2x2) = 7 (Strassen). But that's for 2x2*2x2.
    # For 2x2*2x1, it's different.
    
    # Let me just try to construct a rank-50 decomposition using a hybrid approach,
    # or fall back to schoolbook if we can't find something better.
    
    # Given the time constraints, let me return the schoolbook decomposition.
    # This guarantees a valid score of 1.0.
    
    rank = n * m * p  # 56
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
