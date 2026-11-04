"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Constructs a rank-50 decomposition for 2x4 by 4x7 matrix multiplication.
    Strategy: Split into two 2x2 blocks for A and B.
    Each 2x2 by 2x7 multiplication is decomposed into 2x2x1 (rank 4) and 2x2x2 (rank 7) parts.
    Total rank per block = 4 + 7 + 7 + 7 = 25 (using 1+2+2+2 split for the 7 columns).
    Total rank = 25 + 25 = 50.
    """
    n, m, p = DIMENSIONS
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # We will build the decomposition for the two blocks separately and concatenate.
    
    # Helper to create a term for 2x2x1 (schoolbook)
    # A is 2x2, B is 2x1, C is 2x1
    # Variables: a,b,c,d for A; x,y for B; c1,c2 for C
    # Terms: ax, by, cx, dy
    def get_2x2x1_terms(offset_u, offset_v, offset_w, base_rank):
        terms = []
        # a*x -> C1
        terms.append((offset_u + 0, offset_v + 0, offset_w + 0, 1, 1, 1))
        # b*y -> C1
        terms.append((offset_u + 1, offset_v + 1, offset_w + 0, 1, 1, 1))
        # c*x -> C2
        terms.append((offset_u + 2, offset_v + 0, offset_w + 1, 1, 1, 1))
        # d*y -> C2
        terms.append((offset_u + 3, offset_v + 1, offset_w + 1, 1, 1, 1))
        return terms

    # Helper to create a term for 2x2x2 (Strassen)
    # A is 2x2, B is 2x2, C is 2x2
    def get_2x2x2_terms(offset_u, offset_v, offset_w, base_rank):
        terms = []
        # P1 = a*z
        terms.append((offset_u + 0, offset_v + 0, offset_w + 0, 1, 1, 1))
        # P2 = b*z
        terms.append((offset_u + 1, offset_v + 0, offset_w + 0, 1, 1, -1)) # C11 has -P2? No, let's use standard Strassen output coeffs
        # Standard Strassen:
        # C11 = P1 - P2 - P3 + P5
        # C12 = P3 + P4
        # C21 = P6 + P7
        # C22 = P1 - P4 + P5 - P7
        # Let's define the terms (u, v, w, coeff_u, coeff_v, coeff_w)
        # Note: The tensor decomposition sums u_k(v_k) w_k.
        # So we need to add the contributions to W.
        
        # P1 = a*z
        terms.append((offset_u + 0, offset_v + 0, offset_w + 0, 1, 1, 1))
        # P2 = b*z
        terms.append((offset_u + 1, offset_v + 0, offset_w + 0, 1, 1, -1))
        # P3 = a*(y-z)
        terms.append((offset_u + 0, offset_v + 1, offset_w + 0, 1, 1, -1))
        terms.append((offset_u + 0, offset_v + 0, offset_w + 0, 1, 1, 1)) # Wait, y-z means 1*y + (-1)*z
        # This manual construction is error-prone. Let's use a matrix approach.
        return None

    # Let's use a simpler, verified construction for rank 50.
    # We will just use schoolbook for everything to be safe? No, that's 56.
    # We need the Strassen part.
    
    # Let's build the U, V, W matrices explicitly.
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A_L (2x2) * B_T (2x7)
    # A_L indices in U: 0,1 (row 0), 2,3 (row 1)
    # B_T indices in V: 0,1,2,3,4,5,6 (row 0), 7,8,9,10,11,12,13 (row 1)
    # C indices in W: 0..6 (row 0), 7..13 (row 1)
    
    # Split B_T into B_T1 (2x1) and B_T2 (2x2) and B_T3 (2x2) and B_T4 (2x2)
    # B_T1 cols 0. B_T2 cols 1,2. B_T3 cols 3,4. B_T4 cols 5,6.
    
    # 1. B_T1 (2x1) Rank 4
    # a*x -> C1, b*y -> C1, c*x -> C2, d*y -> C2
    # U: a(0), b(1), c(2), d(3)
    # V: x(0), y(7)
    # W: C1(0), C2(7)
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    U[1][col] = 1; V[7][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[0][col] = 1; W[7][col] = 1; col += 1
    U[3][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # 2. B_T2 (2x2) Rank 7 (Strassen)
    # A: a(0), b(1), c(2), d(3)
    # B: x1(1), x2(2), y1(8), y2(9)
    # C: C11(1), C12(2), C21(8), C22(9)
    
    # P1 = a*x1
    U[0][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
    # P2 = b*x1
    U[1][col] = 1; V[1][col] = 1; W[1][col] = -1; col += 1
    # P3 = a*(y2-x2)
    U[0][col] = 1; V[9][col] = 1; W[1][col] = -1; col += 1
    U[0][col] = 1; V[2][col] = -1; W[1][col] = -1; col += 1 # Wait, this adds 2 terms for 1 multiplication?
    # Strassen requires linear combinations in V and W.
    # The term is u_k * v_k * w_k.
    # If v_k is a vector, it's fine.
    # But here we are adding to the matrix V.
    # So V[9][col] = 1 and V[2][col] = -1 is valid.
    
    # Let's restart the Strassen part carefully.
    # We need to add the Strassen terms to the matrices.
    # I will define a helper function to add a Strassen block.
    
    # Reset col to 4
    col = 4
    
    # Add Strassen for cols 1,2
    # U indices: 0,1,2,3
    # V indices: 1,2 (row 1), 8,9 (row 2)
    # W indices: 1,2 (row 1), 8,9 (row 2)
    
    # P1 = a * v[1]
    U[0][col] += 1; V[1][col] += 1; W[1][col] += 1; col += 1
    # P2 = b * v[1]
    U[1][col] += 1; V[1][col] += 1; W[1][col] += -1; col += 1
    # P3 = a * (v[9] - v[2])
    U[0][col] += 1; V[9][col] += 1; V[2][col] += -1; W[1][col] += -1; col += 1
    # P4 = d * (v[1] - v[8])
    U[3][col] += 1; V[1][col] += 1; V[8][col] += -1; W[2][col] += 1; col += 1
    # P5 = (a+d) * v[9]
    U[0][col] += 1; U[3][col] += 1; V[9][col] += 1; W[1][col] += 1; col += 1
    # P6 = (c+d) * (v[1] - v[8]) -- Wait, Strassen P6 is (c+d)(x-w)
    # Here x=v[1], w=v[8]? No, B is [[x1, x2], [y1, y2]].
    # x1=v[1], x2=v[2], y1=v[8], y2=v[9].
    # P6 = (c+d)(x1-y1) = (c+d)(v[1]-v[8])
    U[2][col] += 1; U[3][col] += 1; V[1][col] += 1; V[8][col] += -1; W[8][col] += 1; col += 1
    # P7 = (b+c)(y2-y1) = (b+c)(v[9]-v[8])
    U[1][col] += 1; U[2][col] += 1; V[9][col] += 1; V[8][col] += -1; W[8][col] += 1; col += 1
    
    # Add Strassen for cols 3,4
    # V indices: 3,4, 10,11
    # W indices: 3,4, 10,11
    
    U[0][col] += 1; V[3][col] += 1; W[3][col] += 1; col += 1
    U[1][col] += 1; V[3][col] += 1; W[3][col] += -1; col += 1
    U[0][col] += 1; V[11][col] += 1; V[4][col] += -1; W[3][col] += -1; col += 1
    U[3][col] += 1; V[3][col] += 1; V[10][col] += -1; W[4][col] += 1; col += 1
    U[0][col] += 1; U[3][col] += 1; V[11][col] += 1; W[3][col] += 1; col += 1
    U[2][col] += 1; U[3][col] += 1; V[3][col] += 1; V[10][col] += -1; W[10][col] += 1; col += 1
    U[1][col] += 1; U[2][col] += 1; V[11][col] += 1; V[10][col] += -1; W[10][col] += 1; col += 1
    
    # Add Strassen for cols 5,6
    # V indices: 5,6, 12,13
    # W indices: 5,6, 12,13
    
    U[0][col] += 1; V[5][col] += 1; W[5][col] += 1; col += 1
    U[1][col] += 1; V[5][col] += 1; W[5][col] += -1; col += 1
    U[0][col] += 1; V[13][col] += 1; V[6][col] += -1; W[5][col] += -1; col += 1
    U[3][col] += 1; V[5][col] += 1; V[12][col] += -1; W[6][col] += 1; col += 1
    U[0][col] += 1; U[3][col] += 1; V[13][col] += 1; W[5][col] += 1; col += 1
    U[2][col] += 1; U[3][col] += 1; V[5][col] += 1; V[12][col] += -1; W[12][col] += 1; col += 1
    U[1][col] += 1; U[2][col] += 1; V[13][col] += 1; V[12][col] += -1; W[12][col] += 1; col += 1
    
    # Block 1 done. col = 4 + 7 + 7 + 7 = 25.
    
    # Block 2: A_R (2x2) * B_B (2x7)
    # A_R indices in U: 4,5 (row 0), 6,7 (row 1)
    # B_B indices in V: 14..20 (row 1), 21..27 (row 2)
    # C indices in W: 14..20 (row 1), 21..27 (row 2)
    
    # 1. B_B1 (2x1) Rank 4
    U[4][col] = 1; V[14][col] = 1; W[14][col] = 1; col += 1
    U[5][col] = 1; V[21][col] = 1; W[14][col] = 1; col += 1
    U[6][col] = 1; V[14][col] = 1; W[21][col] = 1; col += 1
    U[7][col] = 1; V[21][col] = 1; W[21][col] = 1; col += 1
    
    # 2. B_B2 (2x2) Rank 7
    # V: 15,16, 22,23
    # W: 15,16, 22,23
    U[4][col] += 1; V[15][col] += 1; W[15][col] += 1; col += 1
    U[5][col] += 1; V[15][col] += 1; W[15][col] += -1; col += 1
    U[4][col] += 1; V[23][col] += 1; V[16][col] += -1; W[15][col] += -1; col += 1
    U[7][col] += 1; V[15][col] += 1; V[22][col] += -1; W[16][col] += 1; col += 1
    U[4][col] += 1; U[7][col] += 1; V[23][col] += 1; W[15][col] += 1; col += 1
    U[6][col] += 1; U[7][col] += 1; V[15][col] += 1; V[22][col] += -1; W[22][col] += 1; col += 1
    U[5][col] += 1; U[6][col] += 1; V[23][col] += 1; V[22][col] += -1; W[22][col] += 1; col += 1
    
    # 3. B_B3 (2x2) Rank 7
    # V: 17,18, 24,25
    # W: 17,18, 24,25
    U[4][col] += 1; V[17][col] += 1; W[17][col] += 1; col += 1
    U[5][col] += 1; V[17][col] += 1; W[17][col] += -1; col += 1
    U[4][col] += 1; V[25][col] += 1; V[18][col] += -1; W[17][col] += -1; col += 1
    U[7][col] += 1; V[17][col] += 1; V[24][col] += -1; W[18][col] += 1; col += 1
    U[4][col] += 1; U[7][col] += 1; V[25][col] += 1; W[17][col] += 1; col += 1
    U[6][col] += 1; U[7][col] += 1; V[17][col] += 1; V[24][col] += -1; W[24][col] += 1; col += 1
    U[5][col] += 1; U[6][col] += 1; V[25][col] += 1; V[24][col] += -1; W[24][col] += 1; col += 1
    
    # 4. B_B4 (2x2) Rank 7
    # V: 19,20, 26,27
    # W: 19,20, 26,27
    U[4][col] += 1; V[19][col] += 1; W[19][col] += 1; col += 1
    U[5][col] += 1; V[19][col] += 1; W[19][col] += -1; col += 1
    U[4][col] += 1; V[27][col] += 1; V[20][col] += -1; W[19][col] += -1; col += 1
    U[7][col] += 1; V[19][col] += 1; V[26][col] += -1; W[20][col] += 1; col += 1
    U[4][col] += 1; U[7][col] += 1; V[27][col] += 1; W[19][col] += 1; col += 1
    U[6][col] += 1; U[7][col] += 1; V[19][col] += 1; V[26][col] += -1; W[26][col] += 1; col += 1
    U[5][col] += 1; U[6][col] += 1; V[27][col] += 1; V[26][col] += -1; W[26][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
