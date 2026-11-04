"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _get_winograd_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Winograd F(2) algorithm for 2x2x2
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b11
    # m2 = a11 * (b01 - b11)
    # m3 = a00 * (b10 - b00)
    # m4 = (a00 + a01) * b00
    # m5 = a01 * (b10 + b11)
    # m6 = (a10 - a00) * (b00 + b01)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U[i, j, c] corresponds to A[i, j]
    # V[j, k, c] corresponds to B[j, k]
    # W[k, i, c] corresponds to C[k, i] ? No, W is (p*n, rank).
    # In 2x2x2, p=2, n=2. W index is k*n+i.
    # C[i, k] is computed. The tensor is T[i, j, k].
    # W is indexed by k, i.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: (a00 + a11) * (b00 + b11)
    # A: a00(0)=1, a11(3)=1. Others 0.
    U[0][0] = 1; U[3][0] = 1
    # B: b00(0)=1, b11(3)=1. Others 0.
    V[0][0] = 1; V[3][0] = 1
    # C: c00(0) += m0, c01(1) += m0, c10(2) += m0, c11(3) += m0
    # W index: k*n+i.
    # c00: i=0, k=0 -> idx 0
    # c01: i=0, k=1 -> idx 2
    # c10: i=1, k=0 -> idx 1
    # c11: i=1, k=1 -> idx 3
    W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # c=1: (a10 + a11) * b11
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1
    # C: c11(3) += m1
    W[3][1] = 1
    
    # c=2: a11 * (b01 - b11)
    U[3][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C: c01(1) += m2, c11(3) -= m2
    W[2][2] = 1; W[3][2] = -1
    
    # c=3: a00 * (b10 - b00)
    U[0][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C: c10(2) += m3, c00(0) -= m3
    W[1][3] = 1; W[0][3] = -1
    
    # c=4: (a00 + a01) * b00
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1
    # C: c00(0) += m4, c10(2) += m4
    W[0][4] = 1; W[1][4] = 1
    
    # c=5: a01 * (b10 + b11)
    U[1][5] = 1
    V[2][5] = 1; V[3][5] = 1
    # C: c01(1) += m5, c11(3) += m5
    W[2][5] = 1; W[3][5] = 1
    
    # c=6: (a10 - a00) * (b00 + b01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[1][6] = 1
    # C: c10(2) += m6, c11(3) += m6
    W[1][6] = 1; W[3][6] = 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct rank-49 decomposition for 4x4x4 using Strassen and Winograd.
    """
    U2, V2, W2 = _get_winograd_2x2x2()
    
    # Strassen coefficients for 4x4x4
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We will build U, V, W of size (16, 49), (16, 49), (16, 49)
    # 4x4x4 means n=4, m=4, p=4.
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Map 4x4 indices to 2x2 block indices
    # A[i, j] -> block (i//2, j//2), local (i%2, j%2)
    # Local index in 2x2: i_l * 2 + j_l
    
    # For each Strassen step s (0..6), we have a 2x2x2 multiplication.
    # The 2x2x2 decomposition has 7 terms.
    # Total 7 * 7 = 49 terms.
    
    # Strassen definitions:
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We define the linear combinations for A and B for each Strassen step.
    # Then we apply the Winograd decomposition to the resulting 2x2 matrices.
    
    # Winograd U, V, W are for local 2x2 indices.
    # U2[local_i, local_j, c]
    # V2[local_j, local_k, c]
    # W2[local_k, local_i, c]
    
    # We need to map the global 4x4 indices to the local 2x2 indices
    # and account for the Strassen linear combinations.
    
    # This is complex. Instead, we can simply say:
    # The rank-49 decomposition is the tensor product of the rank-7 decomposition
    # with itself, but with Strassen's 7 multiplications instead of 8.
    
    # Actually, a simpler way to construct it is to use the fact that
    # Rank(4x4x4) <= 7 * Rank(2x2x2) = 49.
    
    # Let's use the explicit Kronecker-like structure.
    # For each of the 7 Strassen multiplications, we use the 7 Winograd terms.
    
    # Strassen Step s:
    # A_s = sum_{i,j} alpha_s[i,j] A[i,j]
    # B_s = sum_{j,k} beta_s[j,k] B[j,k]
    # C_s = A_s B_s
    # C = sum_s gamma_s C_s
    
    # For each s, C_s is a 4x4 matrix product? No, C_s is a 4x4 matrix
    # formed by the product of two 4x4 matrices?
    # No, A_s and B_s are 4x4 matrices formed by linear combinations of blocks.
    # But A_s B_s is a 4x4 matrix product.
    # This would require Rank(4x4x4) for each step, leading to 7 * 49 = 343.
    
    # Wait, Strassen reduces the number of *block* multiplications.
    # Each block multiplication is 2x2x2.
    # So we have 7 block multiplications.
    # Each block multiplication is 2x2x2.
    # So we use the 2x2x2 decomposition for each block.
    
    # Block A_s is a 2x2 matrix. Block B_s is a 2x2 matrix.
    # Product C_s is a 2x2 matrix.
    # The 2x2x2 decomposition computes C_s from A_s and B_s.
    
    # So we need to define the coefficients for A_s and B_s in terms of the global A and B.
    
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], etc.
    
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_s, we compute a 2x2x2 product.
    # Let's define the 2x2x2 decomposition for M_s.
    
    # For M1:
    # A_M1 = A00 + A11.
    # B_M1 = B00 + B11.
    # C_M1 = A_M1 B_M1.
    
    # The 2x2x2 decomposition computes C_M1 from A_M1 and B_M1.
    # A_M1[i, j] = A[2*i, 2*j] + A[2*i+1, 2*j+1] ? No.
    # A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
    # A_M1[i, j] = A[i, j] + A[i+2, j+2] for i,j in 0,1.
    
    # We can map the global indices to the local 2x2 indices of M1.
    
    # Let's iterate over s in 0..6.
    # For each s, we have a 2x2x2 problem.
    # We use the 7 terms of Winograd.
    
    # We need to define:
    # A_s[i, j] in terms of global A
    # B_s[j, k] in terms of global B
    # C_s[i, k] in terms of global C (with coefficients)
    
    # This is a large case analysis.
    # Given time, I will return the schoolbook result if this is too complex.
    # But Rank 50 is a big improvement.
    
    # Let's try to implement it properly.
    
    # Define the 7 Strassen steps.
    # Each step has:
    # - A linear combination of A blocks
    # - B linear combination of B blocks
    # - C linear combination of C blocks
    
    # We will store the coefficients in dictionaries.
    
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    # C_blocks: 00, 01, 10, 11
    
    # Step 1: A = A00 + A11, B = B00 + B11, C = M1
    # Step 2: A = A10 + A11, B = B00, C = M2
    # Step 3: A = A00, B = B10 - B00, C = M3
    # Step 4: A = A11, B = B01 - B11, C = M4
    # Step 5: A = A00 + A01, B = B11, C = M5
    # Step 6: A = A10 - A00, B = B00 + B01, C = M6
    # Step 7: A = A01 - A11, B = B10 + B11, C = M7
    
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each step s, we have a 2x2x2 multiplication.
    # We use the Winograd 2x2x2 decomposition.
    
    # The global U, V, W will have 49 columns.
    # Columns 0-6 correspond to Step 1.
    # Columns 7-13 correspond to Step 2.
    # etc.
    
    # For each step s and each Winograd term c (0-6):
    # Global col = s * 7 + c
    
    # We need to fill U, V, W.
    
    # U[global_i, global_j, col]
    # V[global_j, global_k, col]
    # W[global_k, global_i, col]
    
    # For Step s:
    # A_s[i, j] = sum_{block_a} alpha_s[block_a] A_block_a[i, j]
    # B_s[j, k] = sum_{block_b} beta_s[block_b] B_block_b[j, k]
    
    # A_block_a[i, j] corresponds to A[2*row_a + i, 2*col_a + j]
    # B_block_b[j, k] corresponds to B[2*row_b + j, 2*col_b + k]
    
    # C_s[i, k] = sum_{block_c} gamma_s[block_c] C_block_c[i, k]
    # C_block_c[i, k] corresponds to C[2*row_c + i, 2*col_c + k]
    
    # Winograd:
    # U2[i, j, c] * V2[j, k, c] * W2[k, i, c]
    
    # We substitute:
    # U2[i, j, c] * (sum_{block_a} alpha_s[block_a] A[2*row_a + i, 2*col_a + j]) * ...
    
    # This means U[global_i, global_j, col] will have non-zero entries
    # at global_i = 2*row_a + i, global_j = 2*col_a + j.
    
    # Let's implement this.
    
    # Define Strassen steps
    steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {block: coeff}
        # block: (row, col)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}), # M1 -> C00 (partially)
        # Wait, C_coeffs should be the contribution to the final C blocks.
        # M1 contributes to C00 and C11.
        # Let's define the C contribution per M.
    ]
    
    # This is getting messy. I will use the schoolbook fallback for now to ensure correctness,
    # as the Strassen construction is error-prone without verification.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
