"""
Optimized tensor decomposition for 2x4 by 4x6 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks.
Structure:
A (2x4) = [A1, A2] where A1, A2 are 2x2.
B (4x6) = [[B11, B12, B13], [B21, B22, B23]] where Bik are 2x2.
C (2x6) = [C1, C2, C3] where Ci are 2x2.
Ci = A1*B1i + A2*B2i.
This is a direct sum of 3 independent 2x4x2 multiplications.
Each 2x4x2 multiplication is a direct sum of two 2x2x2 multiplications.
Rank of 2x2x2 (Strassen) is 7.
Rank of 2x4x2 is 7 + 7 = 14.
Total rank is 3 * 14 = 42.
"""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # C = A * B
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # U_coeffs[t][u][v] for A[u][v]
    U_coeffs = [
        {}, # index 0 unused
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # V_coeffs[t][v][w] for B[v][w]
    V_coeffs = [
        {},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # W_coeffs[t][u][w] for C[u][w]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_coeffs = [
        {},
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]

    col = 0
    # 3 blocks in p (columns of B and C)
    for k_block in range(3):
        # 2 blocks in m (rows of B, cols of A)
        for m_block in range(2):
            # 7 Strassen steps
            for t in range(1, 8):
                # U: A indices
                # A is 2x4. A[u][2*m_block + v]
                for u in range(2):
                    for v in range(2):
                        coeff = U_coeffs[t].get((u, v), 0)
                        if coeff != 0:
                            i = u
                            j = 2 * m_block + v
                            U[i * m + j][col] += coeff
                            
                # V: B indices
                # B is 4x6. B[2*m_block + v][2*k_block + w]
                for v in range(2):
                    for w in range(2):
                        coeff = V_coeffs[t].get((v, w), 0)
                        if coeff != 0:
                            j = 2 * m_block + v
                            k = 2 * k_block + w
                            V[j * p + k][col] += coeff
                            
                # W: C indices
                # C is 2x6. C[u][2*k_block + w]
                # W[k*n + i]
                for u in range(2):
                    for w in range(2):
                        coeff = W_coeffs[t].get((u, w), 0)
                        if coeff != 0:
                            i = u
                            k = 2 * k_block + w
                            W[k * n + i][col] += coeff
                            
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
