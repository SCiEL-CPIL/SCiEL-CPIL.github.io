"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # For 4x4 * 4x7 matrix multiplication:
    # We can decompose as (4x4 * 4x4) + (4x4 * 4x3)
    # 
    # For the 4x4 * 4x4 part, use Strassen's algorithm (rank 7 for 2x2, so rank 7*4=28 for 4x4)
    # For the 4x4 * 4x3 part, use schoolbook (rank 4*3=12)
    # Total rank = 28 + 12 = 40
    #
    # Actually, let's think more carefully.
    # Strassen for 2x2*2x2 uses rank 7.
    # For 4x4*4x4, we can split into four 2x2 blocks, and use Strassen, giving rank 7*4 = 28.
    # For 4x4*4x3, we can split the 4x3 into a 4x2 and 4x1, or just use schoolbook: 4*3 = 12 multiplications.
    # But we can do better: split 4x4 into two 2x4, and 4x3 into... this is getting complicated.
    #
    # Let's use a simpler approach: 
    # Split A (4x4) into A1 (2x4) and A2 (2x4)
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # Then C1 = A1*B1 (2x4) and C2 = A2*B2 (2x3)
    # For C1: use Strassen on 2x2 blocks. A1 is 2x4, B1 is 4x4.
    # Split A1 into A11 (2x2) and A12 (2x2), B1 into B11 (2x2) and B12 (2x2)
    # C1_left = A11*B11 (2x2), C1_right = A12*B12 (2x2)
    # Each 2x2*2x2 uses rank 7, so C1 uses rank 14.
    # For C2: A2 is 2x4, B2 is 4x3.
    # Split A2 into A21 (2x2) and A22 (2x2), B2 into B21 (2x2) and B22 (2x1)
    # C2_left = A21*B21 (2x2) uses rank 7
    # C2_right = A22*B22 (2x1) uses rank 2 (schoolbook: 2 multiplications)
    # Total rank = 14 + 7 + 2 = 23
    #
    # Wait, I need to be more careful. Let me just use schoolbook for simplicity and correctness first.
    # Schoolbook for 4x4 * 4x7: 4*4*7 = 112 multiplications.
    #
    # Let me try a mixed approach:
    # Split A into two 2x4 matrices: A_top (2x4), A_bottom (2x4)
    # Split B into B_left (4x4) and B_right (4x3)
    # C_top_left = A_top * B_left (2x4) - use Strassen: rank 7
    # C_top_right = A_top * B_right (2x3) - use schoolbook: rank 6
    # C_bottom_left = A_bottom * B_left (2x4) - use Strassen: rank 7
    # C_bottom_right = A_bottom * B_right (2x3) - use schoolbook: rank 6
    # Total rank = 7 + 6 + 7 + 6 = 26
    #
    # Let me implement Strassen for 2x2*2x2 and combine with schoolbook for 2x3.
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For the U, V, W matrices, I need to express each multiplication as a tensor decomposition.
    # This is complex. Let me just return the schoolbook decomposition for now and verify it works.
    #
    # Schoolbook: for each (i,j,k), T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    #
    # The rank-1 decomposition: for each (i,j,k), we have a term A[i,j]*B[j,k]*C[i,k]
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    #
    # This gives rank = n*m*p = 4*4*7 = 112
    #
    # Let me just return this and make sure it's valid.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

# EVOLVE-BLOCK-END
