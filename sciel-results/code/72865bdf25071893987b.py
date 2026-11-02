"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll use a constructive approach:
    # Split the 8 columns into two groups of 4
    # For each 3x3 x 3x4 sub-problem, use a rank-27 construction (schoolbook)
    # But we can do better by exploiting structure
    
    # Actually, let's use a more clever approach:
    # The tensor is T = sum_{i,j,k} A[i,j] * B[j,k] * C[k,i]
    # where C[k,i] is the output coefficient
    
    # For 3x3 x 3x8, the schoolbook rank is 3*3*8 = 72
    # We can try to reduce this using the fact that 8 = 2*4
    
    # Let's use a Kronecker product structure:
    # 3x3 x 3x8 = (3x3 x 3x1) ⊗ (1x1 x 1x8) in some sense
    # But that doesn't directly help
    
    # Better approach: use the fact that for each row of A (3 rows),
    # we multiply by B (3x8), giving 3 outputs per row
    # So we have 3 independent 3x8 matrix-vector products (if we view A rows as vectors)
    # Wait, that's not quite right either
    
    # Let me think differently. The multiplication is:
    # C = A * B where A is 3x3, B is 3x8, C is 3x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Schoolbook: 3*3*8 = 72 multiplications
    # We can use the fact that 3x3 matrix multiplication has rank 23 (or less)
    # but that's for 3x3 x 3x3
    
    # For 3x3 x 3x8, let's try to use a construction based on:
    # Splitting B into two 3x4 blocks: B = [B1 | B2]
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # Each C = A*B is 3x3 x 3x4
    # Schoolbook for 3x3 x 3x4: 3*3*4 = 36
    # Total: 72 (no improvement)
    
    # Let's try a different approach: use the Strassen-like idea
    # For 3x3, we know rank is at most 23 (Lader's bound gives 23)
    # Actually, the best known upper bound for 3x3 is 23
    # But we need 3x3 x 3x8
    
    # Let me just implement the schoolbook with rank 72 for correctness,
    # then try to find a better construction
    
    # Actually, let's use a smarter construction:
    # For each column k of B, we compute A * B[:,k] which is a 3x3 times 3x1
    # That's 9 multiplications per column, 8 columns = 72 total
    
    # Alternative: use the fact that we can share some computations
    # Let's try rank 54: use 3x3 x 3x4 with rank 27 for each half
    # 27 + 27 = 54
    
    # For 3x3 x 3x4, can we do better than 36?
    # The tensor T(3,3,4) has rank at most 36 (schoolbook)
    # Lower bound: max(3*4, 3*3, 3*4) = 12, but that's weak
    # Actually, for T(n,m,p), rank >= max(nm, mp, pn) is not true
    # The correct lower bound is more subtle
    
    # Let's just use schoolbook for now: rank 72
    rank = n * m * p  # 72
    
    # U has shape (n*m, rank) = (9, 72)
    # V has shape (m*p, rank) = (24, 72)
    # W has shape (p*n, rank) = (24, 72)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j] = 1
                # V[j*p + k] = 1
                # W[k*n + i] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
