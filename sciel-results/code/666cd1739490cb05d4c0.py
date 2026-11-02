"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition using a structured approach.
    # The multiplication tensor T[i,j,k] = 1 if output[i,k] depends on A[i,j]*B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Strategy: Use a combination of:
    # 1. Strassen-like decomposition for 4x4x4 (rank 7) for first 4 columns
    # 2. Additional terms for last 2 columns
    
    # Let's use a direct bilinear form approach.
    # For 4x4x6, we can split B into B1 (4x4) and B2 (4x2).
    # C = A*B1 + A*B2
    
    # For A*B1 (4x4x4): use rank-7 Strassen decomposition
    # For A*B2 (4x4x2): use additional rank
    
    # Strassen 4x4x4 decomposition (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's index A as 2x2 blocks: A[i1*2+i2, j1*2+j2]
    # B as 2x2 blocks: B[j1*2+j2, k1*2+k2]
    # C as 2x2 blocks: C[i1*2+i2, k1*2+k2]
    
    # For the 4x4x4 part (k=0,1,2,3), use Strassen with 7 terms.
    # Each Strassen term M_s = (linear combo of A blocks) * (linear combo of B blocks)
    # This gives 7 * (2x2 * 2x2) = 7 * 4 = 28 rank-1 tensors for the 4x4x4 part.
    
    # Wait, that's not right. Each M_s is a 2x2 matrix product, which itself has rank 4.
    # So total rank for 4x4x4 via Strassen = 7 * 4 = 28.
    
    # But we can do better by using the full 4x4x4 Strassen directly.
    # Actually, the standard Strassen for 4x4x4 has rank 7 if we view it as 
    # 7 scalar products of 4x4 linear forms? No, that's not standard.
    
    # Let me reconsider. The tensor rank of 4x4x4 matrix multiplication is 7 
    # (Strassen's result). This means we can write:
    # C = sum_{r=1}^{7} (U_r A V_r^T) * W_r
    # where U_r, V_r, W_r are 4x4 matrices.
    
    # Actually no, the rank-7 decomposition means:
    # C[i,k] = sum_{r=1}^{7} (sum_{i',j'} A[i',j'] * u_r[i',i]) * (sum_{j',k'} B[j',k'] * v_r[j',k]) * w_r[k]
    
    # Let me just use a known rank-7 decomposition for 4x4x4.
    
    # For now, let's use a simpler approach: 
    # Use the fact that 4x4x6 can be decomposed by splitting into smaller problems.
    
    # Split: C = A * B where B = [B_left (4x4), B_right (4x2)]
    # C_left = A * B_left (4x4x4)
    # C_right = A * B_right (4x4x2)
    
    # For 4x4x4: use rank-7 (Strassen)
    # For 4x4x2: use rank-8 (schoolbook) or better
    
    # Total rank: 7 + 8 = 15? That seems too low. Let me verify.
    
    # Actually, the rank of the 4x4x4 multiplication tensor is 7, meaning we need 
    # 7 rank-1 tensors. Each rank-1 tensor is a product of three vectors.
    
    # For 4x4x2: the tensor has dimensions 16 x 8 x 8 (since n*m=16, m*p=8, p*n=8)
    # Wait, no. For (n,m,p) = (4,4,2): U is (n*m, rank) = (16, rank), V is (m*p, rank) = (8, rank), W is (p*n, rank) = (8, rank)
    
    # Let me just build it directly using known decompositions.
    
    # I'll use a constructive approach with explicit bilinear forms.
    
    # Define helper: create rank-1 tensor contribution
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
    
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] * B[j,k]
    
    # Let me use a grid of bilinear forms.
    
    # For a systematic approach, let's use the following:
    # For each pair of "row index" i (0..3) and "col index" k (0..5),
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can group these into bilinear forms.
    
    # Let's try rank 24 (the lower bound) and see if we can achieve it.
    
    # Actually, let me just implement a known good decomposition.
    
    # For 4x4x6, a known upper bound is around 30-35.
    
    # Let me use a block decomposition:
    # Split A into 2x2 blocks, B into 2x3 blocks (since 4x6 = 2x3 blocks of 2x2)
    
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (each 2x2)
    
    # C = [[C11, C12, C13], [C21, C22, C23]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Each C block is a sum of two 2x2 products.
    # A 2x2 product has rank 4 (schoolbook) or 7 (Strassen).
    # A sum of two 2x2 products: X*P + Y*Q
    
    # For X*P + Y*Q where X,Y,P,Q are 2x2:
    # We can use 8 rank-1 tensors (schoolbook for each, 4+4=8)
    # Or we can use Strassen-like: compute M1=(X+Y)(P+Q), M2=X*(P-Q), M3=Y*(P+Q)
    # Then X*P + Y*Q = (M1 + M2 + M3 - M2)/2 ... let me work this out.
    
    # M1 = (X+Y)(P+Q) = X*P + X*Q + Y*P + Y*Q
    # M2 = X*(P-Q) = X*P - X*Q
    # M3 = Y*(P+Q) = Y*P + Y*Q
    # M1 - M2 + M3 = (X*P + X*Q + Y*P + Y*Q) - (X*P - X*Q) + (Y*P + Y*Q)
    #                 = 2*X*Q + 2*Y*P + 2*Y*Q ... not quite.
    
    # Let me use: M1 = (X+Y)(P+Q), M2 = (X-Y)(P-Q)
    # M1 - M2 = (X*P + X*Q + Y*P + Y*Q) - (X*P - X*Q - Y*P + Y*Q)
    #           = 2*X*Q + 2*Y*P
    # That gives X*Q + Y*P, not X*P + Y*Q.
    
    # Let me just use rank 8 for each pair (schoolbook).
    # 6 blocks * 8 = 48 rank. Plus we might be able to share.
    
    # Actually, let me just implement the schoolbook and then try to reduce.
    
    # For now, let's use a rank-32 decomposition based on the block structure.
    
    # Each 2x2 product X*P can be written as sum of 4 rank-1 tensors:
    # (X[0,0]*P[0,0] + X[0,1]*P[1,0], etc.)
    
    # For C11 = A11*B11 + A12*B21:
    # This is 8 rank-1 tensors (4 for each product).
    
    # Total: 6 blocks * 8 = 48.
    
    # But we can do better by using Strassen for the 2x2 products.
    # Each 2x2 product has rank 7 (Strassen) instead of 4? No, Strassen is for 
    # computing the full product, so it replaces 8 scalar multiplications with 7.
    # In tensor rank terms, the 2x2x2 multiplication tensor has rank 7.
    
    # So each 2x2 product = 7 rank-1 tensors.
    # 6 blocks * 2 products each * 7 = 84? That's worse.
    
    # Wait, I'm confusing myself. Let me be more careful.
    
    # The tensor for 2x2x2 matrix multiplication has rank 7 (Strassen).
    # This means: for computing C = A*B where A,B,C are 2x2, we need 7 rank-1 tensors.
    
    # In our problem, each C block (2x2) is computed as sum of two 2x2 products.
    # C_block = A_block1 * B_block1 + A_block2 * B_block2
    
    # This is a (2,2,2,2,2,2) tensor, which can be seen as the sum of two (2,2,2) tensors.
    # The rank of the sum is at most the sum of the ranks = 7 + 7 = 14.
    # But it could be less due to shared structure.
    
    # For now, let's use rank 14 per C block, giving 6 * 14 = 84.
    
    # That's still better than 96 (schoolbook).
    
    # Let me implement this.
    
    # Actually, I realize I should just implement a working decomposition and verify it.
    
    # Let me use a simpler approach: direct construction with verification.
    
    # I'll use the following decomposition:
    # For each (i, j, k) where T[i,j,k] = 1, we need to cover it.
    # T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..5.
    
    # The schoolbook uses 96 rank-1 tensors, one for each (i,j,k).
    
    # To reduce rank, we use bilinear forms that cover multiple (i,j,k) at once.
    
    # Let me use a structured set of bilinear forms.
    
    # Define 32 bilinear forms (rank 32) and verify they work.
    
    # For now, let me just return the schoolbook and add a deadline guard.
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 4, 4, 6
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
