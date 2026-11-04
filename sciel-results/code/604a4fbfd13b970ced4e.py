import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose 2x4x7 into two parts:
    # 1. 2x4x4 (B split into 4x4 and 4x3)
    # 2. 2x4x3
    # 
    # A = [A1, A2] where A1, A2 are 2x2
    # B1 = [B11; B21] where B11, B21 are 2x4
    # B2 = [B12; B22] where B12, B22 are 2x3
    #
    # AB1 = A1 B11 + A2 B21
    # AB2 = A1 B12 + A2 B22
    #
    # We use Strassen (rank 7) for 2x2x2 blocks.
    # For 2x2x4, we use two Strassen multiplications (rank 14).
    # For 2x2x3, we use one Strassen (rank 7) + one Schoolbook (rank 4) = rank 11.
    
    # Total rank = 2 * 14 + 2 * 11 = 28 + 22 = 50.
    
    rank_total = 50
    U = [[0] * rank_total for _ in range(n * m)] # 8 x 50
    V = [[0] * rank_total for _ in range(m * p)] # 28 x 50
    W = [[0] * rank_total for _ in range(p * n)] # 14 x 50
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # M = [[m00, m01], [m10, m11]]
    # N = [[n00, n01], [n10, n11]]
    # P = MN
    # Strassen:
    # M1 = (m00+m11)(n00+n11)
    # M2 = (m10+m11)n00
    # M3 = m00(n01-n11)
    # M4 = m11(n10-n00)
    # M5 = (m00+m01)n11
    # M6 = (m10-m00)(n00+n01)
    # M7 = (m01-m11)(n10+n11)
    # P00 = M1+M4-M5+M7
    # P01 = M3+M5
    # P10 = M2+M4
    # P11 = M1-M2+M3+M6
    
    def add_strassen(u_off, v_off, w_off, m_indices, n_indices, p_indices):
        nonlocal col
        # m_indices: 4 indices in U (row-major 2x2)
        # n_indices: 4 indices in V (row-major 2x2)
        # p_indices: 4 indices in W (row-major 2x2)
        
        # M1
        U[m_indices[0]][col] = 1; U[m_indices[3]][col] = 1
        V[n_indices[0]][col] = 1; V[n_indices[3]][col] = 1
        W[p_indices[0]][col] = 1; W[p_indices[3]][col] = 1
        col += 1
        
        # M2
        U[m_indices[2]][col] = 1; U[m_indices[3]][col] = 1
        V[n_indices[0]][col] = 1
        W[p_indices[2]][col] = 1; W[p_indices[3]][col] = 1
        col += 1
        
        # M3
        U[m_indices[0]][col] = 1
        V[n_indices[1]][col] = 1; V[n_indices[3]][col] = -1
        W[p_indices[0]][col] = 1; W[p_indices[1]][col] = 1
        col += 1
        
        # M4
        U[m_indices[3]][col] = 1
        V[n_indices[2]][col] = 1; V[n_indices[0]][col] = -1
        W[p_indices[0]][col] = -1; W[p_indices[2]][col] = 1
        col += 1
        
        # M5
        U[m_indices[0]][col] = 1; U[m_indices[1]][col] = 1
        V[n_indices[3]][col] = 1
        W[p_indices[0]][col] = -1; W[p_indices[1]][col] = 1
        col += 1
        
        # M6
        U[m_indices[2]][col] = 1; U[m_indices[0]][col] = -1
        V[n_indices[0]][col] = 1; V[n_indices[1]][col] = 1
        W[p_indices[3]][col] = 1
        col += 1
        
        # M7
        U[m_indices[1]][col] = 1; U[m_indices[3]][col] = -1
        V[n_indices[2]][col] = 1; V[n_indices[3]][col] = 1
        W[p_indices[0]][col] = 1; W[p_indices[3]][col] = 1
        col += 1

    def add_schoolbook_2x2x1(u_off, v_off, w_off, m_indices, n_indices, p_indices):
        nonlocal col
        # m_indices: 4 indices
        # n_indices: 2 indices (column)
        # p_indices: 2 indices (column)
        
        # P0 = m00 n0 + m01 n1
        U[m_indices[0]][col] = 1
        V[n_indices[0]][col] = 1
        W[p_indices[0]][col] = 1
        col += 1
        
        U[m_indices[1]][col] = 1
        V[n_indices[1]][col] = 1
        W[p_indices[0]][col] = 1
        col += 1
        
        # P1 = m10 n0 + m11 n1
        U[m_indices[2]][col] = 1
        V[n_indices[0]][col] = 1
        W[p_indices[1]][col] = 1
        col += 1
        
        U[m_indices[3]][col] = 1
        V[n_indices[1]][col] = 1
        W[p_indices[1]][col] = 1
        col += 1

    # Indices for A (2x4)
    # A1 (2x2): rows 0,1 of A; cols 0,1
    # A2 (2x2): rows 0,1 of A; cols 2,3
    # U shape: 8 x rank
    # A1 indices in U: 0,1,2,3 (row 0: 0,1; row 1: 2,3)
    # A2 indices in U: 4,5,6,7 (row 0: 4,5; row 1: 6,7)
    
    # Indices for B (4x7)
    # B1 (4x4): rows 0-3; cols 0-3
    # B11 (2x4): rows 0,1; cols 0-3
    # B21 (2x4): rows 2,3; cols 0-3
    # B2 (4x3): rows 0-3; cols 4-6
    # B12 (2x3): rows 0,1; cols 4-6
    # B22 (2x3): rows 2,3; cols 4-6
    
    # V shape: 28 x rank
    # B11 indices: 0,1,2,3 (row 0: 0,1,2,3; row 1: 4,5,6,7) -> Wait, V is (m*p, rank) = (4*7, rank) = 28x rank
    # V row index = j*p + k. j is row of B (0-3), k is col of B (0-6).
    # B11 (rows 0,1; cols 0-3):
    #   r0c0: 0*7+0=0
    #   r0c1: 0*7+1=1
    #   r0c2: 0*7+2=2
    #   r0c3: 0*7+3=3
    #   r1c0: 1*7+0=7
    #   r1c1: 1*7+1=8
    #   r1c2: 1*7+2=9
    #   r1c3: 1*7+3=10
    # B21 (rows 2,3; cols 0-3):
    #   r2c0: 2*7+0=14
    #   r2c1: 2*7+1=15
    #   r2c2: 2*7+2=16
    #   r2c3: 2*7+3=17
    #   r3c0: 3*7+0=21
    #   r3c1: 3*7+1=22
    #   r3c2: 3*7+2=23
    #   r3c3: 3*7+3=24
    # B12 (rows 0,1; cols 4-6):
    #   r0c4: 4, r0c5: 5, r0c6: 6
    #   r1c4: 11, r1c5: 12, r1c6: 13
    # B22 (rows 2,3; cols 4-6):
    #   r2c4: 18, r2c5: 19, r2c6: 20
    #   r3c4: 25, r3c5: 26, r3c6: 27

    # W shape: 14 x rank
    # W index = k*n + i. k is col of B (0-6), i is row of A (0-1).
    # Output C (2x7).
    # C11 (2x4): cols 0-3
    #   c00: 0*2+0=0, c01: 0*2+1=1
    #   c10: 1*2+0=2, c11: 1*2+1=3
    #   c20: 2*2+0=4, c21: 2*2+1=5
    #   c30: 3*2+0=6, c31: 3*2+1=7
    # C12 (2x3): cols 4-6
    #   c40: 8, c41: 9
    #   c50: 10, c51: 11
    #   c60: 12, c61: 13

    # 1. A1 B11 (2x2x4)
    # Split B11 into B11a (2x2, cols 0,1) and B11b (2x2, cols 2,3)
    # B11a indices: V[0], V[1], V[7], V[8]
    # B11b indices: V[2], V[3], V[9], V[10]
    # W indices for cols 0,1: W[0], W[1], W[2], W[3]
    # W indices for cols 2,3: W[4], W[5], W[6], W[7]
    
    # A1 indices: U[0], U[1], U[2], U[3]
    
    # Strassen for A1 B11a
    add_strassen(0, 0, 0, [0,1,2,3], [0,1,7,8], [0,1,2,3])
    # Strassen for A1 B11b
    add_strassen(0, 0, 0, [0,1,2,3], [2,3,9,10], [4,5,6,7])
    
    # 2. A2 B21 (2x2x4)
    # B21a (2x2, cols 0,1): V[14], V[15], V[21], V[22]
    # B21b (2x2, cols 2,3): V[16], V[17], V[23], V[24]
    # A2 indices: U[4], U[5], U[6], U[7]
    
    add_strassen(0, 0, 0, [4,5,6,7], [14,15,21,22], [0,1,2,3])
    add_strassen(0, 0, 0, [4,5,6,7], [16,17,23,24], [4,5,6,7])
    
    # 3. A1 B12 (2x2x3)
    # Split B12 into B12a (2x2, cols 4,5) and B12b (2x1, col 6)
    # B12a: V[4], V[5], V[11], V[12]
    # B12b: V[6], V[13]
    # W indices for cols 4,5: W[8], W[9], W[10], W[11]
    # W index for col 6: W[12], W[13]
    
    # Strassen for A1 B12a
    add_strassen(0, 0, 0, [0,1,2,3], [4,5,11,12], [8,9,10,11])
    # Schoolbook for A1 B12b
    add_schoolbook_2x2x1(0, 0, 0, [0,1,2,3], [6,13], [12,13])
    
    # 4. A2 B22 (2x2x3)
    # B22a (2x2, cols 4,5): V[18], V[19], V[25], V[26]
    # B22b (2x1, col 6): V[20], V[27]
    # A2 indices: U[4], U[5], U[6], U[7]
    
    add_strassen(0, 0, 0, [4,5,6,7], [18,19,25,26], [8,9,10,11])
    add_schoolbook_2x2x1(0, 0, 0, [4,5,6,7], [20,27], [12,13])
    
    return {"U": U, "V": V, "W": W}
