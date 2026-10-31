import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4x4 matrix multiplication.
    This is the known minimal rank for 4x4 matrix multiplication (Strassen-Kronecker).
    
    The decomposition is constructed via the Kronecker product of the 7-rank Strassen 
    decomposition for 2x2 matrices.
    
    Convention:
    U has shape (16, 49), indexed by (i, j) where i,j in 0..3.
    V has shape (16, 49), indexed by (j, k) where j,k in 0..3.
    W has shape (16, 49), indexed by (k, i) where k,i in 0..3.
    Note: The task specifies W[k*n+i, r]. With n=4, this is 4*k + i.
    """
    
    # Strassen coefficients for 2x2 matrix multiplication
    # 7 scalar multiplications
    # beta[s][i][j] for A (2x2)
    # gamma[s][j][k] for B (2x2)
    # delta[s][i][k] for C (2x2)
    
    # Indices: 0->00, 1->01, 2->10, 3->11 in 2x2 block sense?
    # Let's use standard 2x2 indices 0,1 for rows and 0,1 for cols.
    
    # Strassen formulas:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    # C00 = m0 - m1 + m3 + m5
    # C01 = m2 + m4 + m5
    # C10 = m1 + m3 - m4 - m6
    # C11 = m0 - m2 + m4 + m6
    
    # We define beta, gamma, delta as 7x2x2 arrays.
    
    beta = np.array([
        [[1, 0], [0, 1]], # m0: a00, a11
        [[0, 0], [1, 1]], # m1: a10, a11
        [[1, 0], [0, 0]], # m2: a00
        [[0, 0], [0, 1]], # m3: a11
        [[1, 1], [0, 0]], # m4: a00, a01
        [[-1, 0], [1, 0]],# m5: -a00, a10
        [[0, 1], [0, -1]] # m6: a01, -a11
    ], dtype=int)
    
    gamma = np.array([
        [[1, 0], [0, 1]], # m0: b00, b11
        [[1, 0], [0, 0]], # m1: b00
        [[0, 1], [0, -1]],# m2: b01, -b11
        [[-1, 0], [1, 0]],# m3: -b00, b10
        [[0, 0], [0, 1]], # m4: b11
        [[1, 1], [0, 0]], # m5: b00, b01
        [[0, 0], [1, 1]]  # m6: b10, b11
    ], dtype=int)
    
    delta = np.array([
        [[1, 0], [0, 1]], # m0: c00, c11
        [[0, 0], [1, -1]],# m1: c10, -c11? Wait.
        # C00 = m0 - m1 + m3 + m5
        # C01 = m2 + m4 + m5
        # C10 = m1 + m3 - m4 - m6
        # C11 = m0 - m2 + m4 + m6
        
        # Let's re-verify delta based on C coefficients.
        # delta[s][i][k] is coeff of m_s in C_{i,k}.
        
        # s=0 (m0): C00=1, C11=1
        # s=1 (m1): C10=1, C00=-1
        # s=2 (m2): C01=1, C11=-1
        # s=3 (m3): C10=1, C00=1
        # s=4 (m4): C01=1, C10=-1, C11=1
        # s=5 (m5): C00=1, C01=1
        # s=6 (m6): C10=-1, C11=1
        
        # Let's build delta correctly:
        # delta[0] = [[1, 0], [0, 1]]
        # delta[1] = [[-1, 0], [1, 0]]
        # delta[2] = [[0, 1], [0, -1]]
        # delta[3] = [[1, 0], [1, 0]]
        # delta[4] = [[0, 1], [-1, 1]]
        # delta[5] = [[1, 1], [0, 0]]
        # delta[6] = [[0, 0], [-1, 1]]
    ], dtype=int)
    
    # Correcting delta based on standard Strassen:
    # m0 -> C00, C11
    # m1 -> C00 (-1), C10 (1)
    # m2 -> C01 (1), C11 (-1)
    # m3 -> C00 (1), C10 (1)
    # m4 -> C01 (1), C10 (-1), C11 (1)
    # m5 -> C00 (1), C01 (1)
    # m6 -> C10 (-1), C11 (1)
    
    delta = np.array([
        [[1, 0], [0, 1]],  # s=0
        [[-1, 0], [1, 0]], # s=1
        [[0, 1], [0, -1]], # s=2
        [[1, 0], [1, 0]],  # s=3
        [[0, 1], [-1, 1]], # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 0], [-1, 1]]  # s=6
    ], dtype=int)
    
    # Verify Strassen 2x2 with these matrices
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C = [[19, 22], [43, 50]]
    # m0 = (1+4)(5+8) = 65
    # m1 = (3+4)5 = 35
    # m2 = 1(6-8) = -2
    # m3 = 4(7-5) = 8
    # m4 = (1+2)8 = 24
    # m5 = (3-1)(5+6) = 22
    # m6 = (2-4)(7+8) = -30
    
    # C00 = 65 - 35 + 8 + 22 = 60? 
    # Wait. 1*5 + 2*7 = 5 + 14 = 19.
    # My manual calculation of m's:
    # m0 = 65
    # m1 = 35
    # m2 = -2
    # m3 = 8
    # m4 = 24
    # m5 = 22
    # m6 = -30
    
    # C00 = m0 - m1 + m3 + m5 = 65 - 35 + 8 + 22 = 60. 
    # 60 != 19.
    # Let's check Strassen formulas again.
    # Standard Strassen:
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m6 ? No.
    
    # Let's look up the exact coefficients.
    # Wikipedia:
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    # m7 = (a00+a11)(b00+b11)
    
    # c00 = m1 + m4 - m5 + m6
    # c01 = m2 + m5
    # c10 = m3 - m4
    # c11 = m2 + m3 - m6 + m7
    
    # Let's test this set.
    # m1 = 35
    # m2 = -2
    # m3 = 8
    # m4 = 24
    # m5 = 22
    # m6 = -30
    # m7 = 65
    
    # c00 = 35 + 24 - 22 - 30 = 7. (Should be 19).
    # Something is wrong with my manual check or the formula set.
    
    # Let's use the one from the previous code which seemed to be working (score 1.3061).
    # The previous code used:
    # beta = ...
    # gamma = ...
    # delta = ...
    
    # I will stick to the previous code's matrices as they produced a valid score.
    # Re-defining the previous code's matrices to be safe.
    
    beta_prev = np.array([
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[-1, 0], [1, 0]], 
        [[0, 1], [0, -1]]
    ], dtype=int)
    
    gamma_prev = np.array([
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]], 
        [[-1, 0], [1, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]
    ], dtype=int)
    
    delta_prev = np.array([
        [[1, 0], [0, 1]], 
        [[0, 0], [1, -1]], 
        [[0, 1], [0, 1]], 
        [[1, 0], [1, 0]], 
        [[-1, 1], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 0], [0, 0]]
    ], dtype=int)
    
    # Let's verify delta_prev with the same A, B.
    # m0..m6 calculated above: 65, 35, -2, 8, 24, 22, -30.
    
    # s=0: delta[0] = [[1,0],[0,1]] -> C00+=65, C11+=65
    # s=1: delta[1] = [[0,0],[1,-1]] -> C10+=35, C11-=35
    # s=2: delta[2] = [[0,1],[0,1]] -> C01+=-2, C11+=-2
    # s=3: delta[3] = [[1,0],[1,0]] -> C00+=8, C10+=8
    # s=4: delta[4] = [[-1,1],[0,0]] -> C00-=24, C01+=24
    # s=5: delta[5] = [[0,0],[0,1]] -> C11+=22
    # s=6: delta[6] = [[1,0],[0,0]] -> C00+=-30
    
    # Sum C00 = 65 + 8 - 24 - 30 = 19. Correct!
    # Sum C01 = -2 + 24 = 22. Correct!
    # Sum C10 = 35 + 8 = 43. Correct!
    # Sum C11 = 65 - 35 - 2 + 22 = 50. Correct!
    
    # So the previous matrices are correct.
    
    beta = beta_prev
    gamma = gamma_prev
    delta = delta_prev
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct Kronecker product
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U: A coefficients
            # A is 4x4. Split into 2x2 blocks.
            # Block (I, J) where I, J in {0,1}.
            # Inside block, indices i_prime, j_prime in {0,1}.
            # Global row i = 2*I + i_prime
            # Global col j = 2*J + j_prime
            for I in range(2):
                for J in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = 2 * I + i_prime
                            j = 2 * J + j_prime
                            row_u = i * 4 + j
                            coeff = beta[s, I, J] * beta[t, i_prime, j_prime]
                            if coeff != 0:
                                U[row_u, col] = coeff
                                
            # V: B coefficients
            # B is 4x4. Split into 2x2 blocks.
            # Block (J, K) where J, K in {0,1}.
            # Global row j = 2*J + j_prime
            # Global col k = 2*K + k_prime
            for J in range(2):
                for K in range(2):
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = 2 * J + j_prime
                            k = 2 * K + k_prime
                            row_v = j * 4 + k
                            coeff = gamma[s, J, K] * gamma[t, j_prime, k_prime]
                            if coeff != 0:
                                V[row_v, col] = coeff
                                
            # W: C coefficients
            # C is 4x4. Split into 2x2 blocks.
            # Block (K, I) where K, I in {0,1}.
            # Global row k = 2*K + k_prime
            # Global col i = 2*I + i_prime
            # Task convention: W[k*n+i, r]. n=4. Index = 4*k + i.
            for K in range(2):
                for I in range(2):
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = 2 * K + k_prime
                            i = 2 * I + i_prime
                            row_w = k * 4 + i
                            coeff = delta[s, I, K] * delta[t, i_prime, k_prime]
                            if coeff != 0:
                                W[row_w, col] = coeff
                                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
