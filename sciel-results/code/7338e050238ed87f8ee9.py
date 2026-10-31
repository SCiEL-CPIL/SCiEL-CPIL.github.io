import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    # Rank is 49 (Strassen's algorithm applied recursively)
    rank = 49
    
    # Initialize U, V, W as lists of lists of zeros
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    # n=4, m=4, p=4 => 16x49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrices uses 7 multiplications.
    # For 4x4, we split into 2x2 blocks.
    # Each block product requires 7 multiplications.
    # There are 8 block products? No, Strassen combines them.
    # Standard Strassen for 4x4:
    # We treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    # 
    # Strassen computes 7 intermediate matrices M1..M7 which are products of sums/differences of 2x2 blocks.
    # Each Mi is a 2x2 matrix product.
    # To compute each 2x2 product exactly, we use the standard 8 multiplications? 
    # No, the rank of 2x2 matrix multiplication is 7.
    # So each of the 7 "big" multiplications (which are 2x2 matrix multiplications) can be done with rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The decomposition is a Kronecker product structure of the 2x2 Strassen decomposition.
    # Let U2, V2, W2 be the rank-7 decomposition for 2x2 matrix multiplication.
    # Then U = U2 \otimes U2, V = V2 \otimes V2, W = W2 \otimes W2 (with appropriate permutations).
    
    # Let's define the rank-7 decomposition for 2x2 matrices.
    # Indices for 2x2:
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # The 7 multiplications in Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map this to the 49-rank decomposition for 4x4.
    # The 4x4 matrix is composed of 2x2 blocks.
    # Let's say the 4x4 index (i,j) maps to block (I,J) and inner index (i',j') where I=i//2, i'=i%2, etc.
    # The "outer" Strassen operates on the blocks.
    # The "inner" Strassen operates on the 2x2 elements within each block product.
    
    # Actually, the standard recursive Strassen for 4x4 is:
    # 1. Split A, B into 2x2 blocks of size 2x2.
    # 2. Apply Strassen's 7 multiplications to these blocks.
    #    Each multiplication is a 2x2 * 2x2 matrix product.
    # 3. Each 2x2 * 2x2 product is decomposed using Strassen's 7-multiplication method (rank 7).
    # 4. Total rank = 7 * 7 = 49.
    
    # We can construct U, V, W by iterating over the 7 outer multiplications (s) and 7 inner multiplications (t).
    # For each pair (s, t), we have a rank-1 component.
    
    # Define the coefficients for the outer Strassen (blocks)
    # beta_s[I][J] for A block (I,J)
    # gamma_s[J][K] for B block (J,K)
    # delta_s[I][K] for C block (I,K)
    
    beta = [
        [[1, 0], [0, 1]],  # s=0: M1 uses A00, A11
        [[0, 0], [1, 1]],  # s=1: M2 uses A10, A11
        [[1, 0], [0, 0]],  # s=2: M3 uses A00
        [[0, 0], [0, 1]],  # s=3: M4 uses A11
        [[1, 1], [0, 0]],  # s=4: M5 uses A00, A01
        [[-1, 0], [1, 0]], # s=5: M6 uses A10, A00
        [[0, 1], [0, -1]]  # s=6: M7 uses A01, A11
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # s=0: M1 uses B00, B11
        [[1, 0], [0, 0]],  # s=1: M2 uses B00
        [[0, 1], [0, -1]], # s=2: M3 uses B01, B11
        [[-1, 0], [1, 0]], # s=3: M4 uses B10, B00
        [[0, 0], [0, 1]],  # s=4: M5 uses B11
        [[1, 1], [0, 0]],  # s=5: M6 uses B00, B01
        [[0, 0], [1, 1]]   # s=6: M7 uses B10, B11
    ]
    
    delta = [
        [[1, 0], [0, 1]],  # s=0: M1 contributes to C00, C11
        [[0, 0], [1, -1]], # s=1: M2 contributes to C10, C11
        [[0, 1], [0, 1]],  # s=2: M3 contributes to C01, C11
        [[1, 0], [1, 0]],  # s=3: M4 contributes to C00, C10
        [[-1, 1], [0, 0]], # s=4: M5 contributes to C00, C01
        [[0, 0], [0, 1]],  # s=5: M6 contributes to C11
        [[1, 0], [0, 0]]   # s=6: M7 contributes to C00
    ]
    
    # Now define the inner Strassen decomposition for 2x2 matrices (rank 7)
    # U2, V2, W2 are 4x7 matrices.
    # Rows: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Cols: 0..6 (the 7 multiplications)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0 (M1 inner)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # t=1 (M2 inner)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # t=2 (M3 inner)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # t=3 (M4 inner)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # t=4 (M5 inner)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # t=5 (M6 inner)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # t=6 (M7 inner)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct the full 49-rank decomposition
    # For each outer multiplication s (0..6) and inner multiplication t (0..6):
    # Column index c = 7*s + t
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    b_val = beta[s][I][J]
                    if b_val != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u2_val = U2[i_prime * 2 + j_prime][t]
                                if u2_val != 0:
                                    # Row index in U: i * 4 + j
                                    # i = I * 2 + i_prime
                                    # j = J * 2 + j_prime
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    row_u = i * 4 + j
                                    U[row_u][c] += b_val * u2_val
                                    
            # Fill V
            for J in range(2):
                for K in range(2):
                    g_val = gamma[s][J][K]
                    if g_val != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v2_val = V2[j_prime * 2 + k_prime][t]
                                if v2_val != 0:
                                    # Row index in V: j * 4 + k
                                    # j = J * 2 + j_prime
                                    # k = K * 2 + k_prime
                                    j = J * 2 + j_prime
                                    k = K * 2 + k_prime
                                    row_v = j * 4 + k
                                    V[row_v][c] += g_val * v2_val
                                    
            # Fill W
            for I in range(2):
                for K in range(2):
                    d_val = delta[s][I][K]
                    if d_val != 0:
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w2_val = W2[k_prime * 2 + i_prime][t]
                                if w2_val != 0:
                                    # Row index in W: k * 4 + i
                                    # k = K * 2 + k_prime
                                    # i = I * 2 + i_prime
                                    k = K * 2 + k_prime
                                    i = I * 2 + i_prime
                                    row_w = k * 4 + i
                                    W[row_w][c] += d_val * w2_val

            if time.time() > deadline:
                return {"U": U, "V": W, "W": W} # Note: W is output, V is middle. Wait, interface is U, V, W.
                # U: (n*m, rank)
                # V: (m*p, rank)
                # W: (p*n, rank)
                # My loops:
                # U rows: i*4+j (n*m=16)
                # V rows: j*4+k (m*p=16)
                # W rows: k*4+i (p*n=16)
                # This matches.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
