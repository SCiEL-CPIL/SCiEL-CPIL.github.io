"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
from scipy.linalg import solve
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # The tensor for matrix multiplication C = A @ B
    # A is n x m (2x4), B is m x p (4x8), C is n x p (2x8)
    # Indices: A[i, j], B[j, k], C[i, k]
    # Tensor T[i, j, k] = 1 if it contributes to C[i, k] via A[i, j] * B[j, k]
    # The rank of this tensor is what we want to minimize.
    
    # Strategy: 
    # We can view this as multiplying two blocks of A (each 2x2? No, A is 2x4, so two 2x2 blocks horizontally)
    # and two blocks of B (each 4x4? No, B is 4x8, so two 4x4 blocks horizontally).
    # Actually, let's look at the structure more carefully.
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 4x4? No, B is 4x8. So B = [B1 | B2] where B1, B2 are 4x4.
    # C = A1 @ B1 + A2 @ B2.
    # So we have two independent 2x2 @ 4x4 multiplications.
    
    # Let's find the rank of a 2x2 @ 4x4 multiplication.
    # A (2x2) @ B (4x4) -> C (2x4).
    # This is equivalent to 2 independent 1x2 @ 4x4 multiplications? No.
    # Row i of C is Row i of A (1x2) times B (4x4)? No, A is 2x2, so Row i of A is 1x2.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So for a fixed i, it's a 1x2 vector times a 4x4 matrix.
    # This is a linear map from R^2 (the row of A) and R^4x4 (B) to R^4 (the row of C).
    
    # Let's use a known result or search for the rank.
    # Rank of 2x2 @ 2x2 is 7 (Strassen).
    # Rank of 2x2 @ 4x4?
    # We can split B into two 4x2 blocks? No.
    # Let's try to construct a decomposition.
    
    # Let's start with the schoolbook rank for 2x4 @ 4x8.
    # Rank = 2 * 4 * 8 = 64.
    
    # Can we do better?
    # Let's try to decompose into smaller blocks.
    # A is 2x4. Let A = [A1 A2] with A1, A2 in R^{2x2}.
    # B is 4x8. Let B = [B1 B2] with B1, B2 in R^{4x4}.
    # C = A1 B1 + A2 B2.
    # We need to compute two products of 2x2 and 4x4 matrices and add them.
    # If we can compute 2x2 @ 4x4 with rank r, then we can compute this with rank 2r (by doing them independently) or less if we share structure.
    
    # What is the rank of 2x2 @ 4x4?
    # Let's consider the tensor T[i, j, k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # This is a 2x4x4 tensor.
    # We can view this as 4 independent 2x1 @ 1x4? No.
    # It's a bilinear map.
    # Let's try to find a decomposition.
    # The schoolbook rank is 2*4*4 = 32.
    # Can we do better?
    # Note that 2x4 @ 4x4 is equivalent to (2x2 @ 2x2) with some structure?
    # Let's split B into 4 columns.
    # C[:, k] = A @ B[:, k].
    # This is a linear transformation from A (4x1 vector if flattened) to C (8x1 vector if flattened? No, C is 2x4).
    # Actually, for fixed B, C is linear in A. For fixed A, C is linear in B.
    
    # Let's use a generic rank minimization approach for small tensors.
    # We can formulate this as a linear algebra problem.
    # We want to find matrices U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i, i'} delta_{k, k'} ... wait.
    # The tensor is T[i, j, k] = 1 for all i, j, k? No.
    # The matrix multiplication tensor T has entries T[i, j, k] = 1.
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    # Wait, the output is C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor is T[i, j, k] = 1.
    # The decomposition is: T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # Note the indices in W: W[k*n+i, c]. The task says W[k*n+i, r].
    
    # Let's try to find a low-rank decomposition for 2x4 @ 4x8.
    # We can use the fact that 4 = 2+2 and 8 = 4+4.
    # Let's try to compute the rank of 2x2 @ 4x4 first.
    # Let's assume we can do 2x2 @ 4x4 with rank 14 (guessing based on 7*2).
    # Then 2x4 @ 4x8 would be rank 28.
    
    # Let's try to construct a rank 28 decomposition.
    # A = [A1 A2], B = [B1 B2].
    # C = A1 B1 + A2 B2.
    # We need to compute A1 B1 and A2 B2.
    # Let's find a rank 14 decomposition for 2x2 @ 4x4.
    # Let A' be 2x2, B' be 4x4, C' be 2x4.
    # We can split B' into two 4x2 blocks: B' = [B'_1 B'_2].
    # Then C' = A' B'_1 + A' B'_2.
    # This is two 2x2 @ 4x2 multiplications.
    # 2x2 @ 4x2: A (2x2) @ B (4x2) -> C (2x2).
    # This is equivalent to 2x2 @ 2x2 with B transposed? No.
    # Let's look at 2x2 @ 4x2.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # This is a 2x4x2 tensor.
    # Schoolbook rank: 2*4*2 = 16.
    # Can we do better?
    # Note that B is 4x2. We can view this as two 4x1 vectors.
    # C[:, 0] = A @ B[:, 0] and C[:, 1] = A @ B[:, 1].
    # This is two independent 2x2 @ 4x1 multiplications.
    # 2x2 @ 4x1: A (2x2) @ B (4x1) -> C (2x1).
    # C[i] = sum_j A[i, j] B[j].
    # This is a 2x4x1 tensor.
    # Schoolbook rank: 8.
    # Can we do better?
    # The tensor is T[i, j, 0] = 1.
    # This is equivalent to the dot product of a 2x2 matrix and a 4x1 vector.
    # We can write A as a 4x1 vector (flattened). Then C is a linear map from R^4 to R^2.
    # The rank of this tensor is the rank of the linear map? No, it's a bilinear map.
    # Actually, for fixed B, C is linear in A. For fixed A, C is linear in B.
    # The tensor T[i, j, k] has rank 1 if we fix i and j? No.
    # Let's try to find a decomposition.
    # We want sum_c U[i*2+j, c] V[j*1+k, c] W[k*2+i, c] = 1.
    # Here n=2, m=2, p=1.
    # U is 4x r, V is 2x r, W is 2x r.
    # This is the tensor for 2x2 @ 2x1.
    # The rank of 2x2 @ 2x1 is 4?
    # Let's check.
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # This is 4 terms. Can we do it in 3?
    # It's a bilinear map from R^2 x R^2 to R^2.
    # The rank is at least 4 because the two output coordinates are linearly independent.
    # Actually, the rank of the matrix multiplication tensor 2x2 @ 2x1 is 4.
    # So 2x2 @ 4x1 has rank 8?
    # Wait, B is 4x1. So we have 4 inputs for B.
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # B[2] and B[3] are not used? No, B is 4x1, so B[0..3].
    # But A is 2x2, so it only uses B[0] and B[1].
    # So the tensor is actually 2x2 @ 2x1, and B[2], B[3] are dummy.
    # So the rank is 4.
    
    # So 2x2 @ 4x2 has rank 4+4=8?
    # C[:, 0] uses B[:, 0] (4x1), rank 4.
    # C[:, 1] uses B[:, 1] (4x1), rank 4.
    # Total rank 8.
    
    # So 2x2 @ 4x4 has rank 8+8=16?
    # C[:, 0] uses B[:, 0], rank 4.
    # C[:, 1] uses B[:, 1], rank 4.
    # C[:, 2] uses B[:, 2], rank 4.
    # C[:, 3] uses B[:, 3], rank 4.
    # Total rank 16.
    
    # So 2x4 @ 4x8 has rank 16+16=32?
    # A1 B1: 2x2 @ 4x4, rank 16.
    # A2 B2: 2x2 @ 4x4, rank 16.
    # Total rank 32.
    
    # This is the same as schoolbook (2*4*8=64? No, 2*4*8=64. Wait.
    # Schoolbook rank for n x m @ m x p is n*m*p.
    # 2*4*8 = 64.
    # My calculation gave 32.
    # Let's re-evaluate.
    # 2x2 @ 4x1:
    # C[0] = A[0,0]B[0] + A[0,1]B[1]
    # C[1] = A[1,0]B[0] + A[1,1]B[1]
    # This is 4 terms.
    # But B is 4x1. So B[2] and B[3] are not used.
    # So the rank is 4.
    # 2x2 @ 4x4:
    # C[:, k] = A @ B[:, k].
    # Each column uses 4 terms.
    # 4 columns -> 16 terms.
    # So rank 16.
    # 2x4 @ 4x8:
    # A1 B1: 16 terms.
    # A2 B2: 16 terms.
    # Total 32 terms.
    
    # So we can achieve rank 32.
    # Let's construct the decomposition for rank 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # A1 B1
    # A1 is A[0:2, 0:2], B1 is B[0:4, 0:4]
    # C[0:2, 0:4] = A1 @ B1
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * 4 + j][col] = 1  # A[i, j]
                V[j * 8 + k][col] = 1  # B[j, k]
                W[k * 2 + i][col] = 1  # C[i, k]
                col += 1
    
    # A2 B2
    # A2 is A[0:2, 2:4], B2 is B[0:4, 4:8]
    # C[0:2, 4:8] = A2 @ B2
    for i in range(2):
        for j in range(2, 4):
            for k in range(4, 8):
                U[i * 4 + j][col] = 1  # A[i, j]
                V[j * 8 + k][col] = 1  # B[j, k]
                W[k * 2 + i][col] = 1  # C[i, k]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
