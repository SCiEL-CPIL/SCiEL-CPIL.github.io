"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with convention W[k*n+i, r]
    
    The tensor entry T[i, j, k] (where C[i,k] = sum_j A[i,j]B[j,k]) is 1 if (i,j,k) is a valid index.
    The decomposition says:
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    We check this for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target value is 1.0 for valid indices (which are all i in 0..n-1, j in 0..m-1, k in 0..p-1)
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _to_fraction_list(lst):
    """Convert a list of floats to a list of Fractions if possible."""
    result = []
    for x in lst:
        # Check if it's close to a simple fraction
        f = Fraction(x).limit_denominator(100)
        if abs(float(f) - x) < 1e-9:
            result.append(f)
        else:
            result.append(x)
    return result

def _rank_of_matrix(matrix):
    """Compute the rank of a matrix (list of lists)."""
    try:
        return int(np.linalg.matrix_rank(np.array(matrix)))
    except:
        return 0

def construct():
    n, m, p = DIMENSIONS
    target_rank = 19
    # U: (n*m, rank) -> (4, rank)
    # V: (m*p, rank) -> (10, rank)
    # W: (p*n, rank) -> (10, rank)
    
    # Strategy:
    # We know the schoolbook rank is 20.
    # We want to find a rank 19 decomposition.
    # One way is to use Strassen-like tricks.
    # Strassen works for 2x2 by 2x2.
    # Here we have 2x2 by 2x5.
    # We can split the 5 columns into 2+2+1? Or 2+3?
    # If we do Strassen for the first 2 columns (2x2 by 2x2), that takes 7 multiplications.
    # The remaining 3 columns are 2x2 by 2x3, which takes 2*2*3 = 12 multiplications in schoolbook.
    # Total 7 + 12 = 19.
    
    # Let's try to construct this explicitly.
    # Let A be 2x2, B be 2x5.
    # Split B into B_left (2x2, cols 0,1) and B_right (2x3, cols 2,3,4).
    # C_left (2x2) = A * B_left. Use Strassen.
    # C_right (2x3) = A * B_right. Use Schoolbook.
    
    # Strassen for C_left:
    # A = [[a00, a01], [a10, a11]]
    # B_left = [[b00, b01], [b10, b11]] (indices 0,1 in B)
    
    # m1 = a00 * (b00 + b11)  -> U: [a00], V: [b00, b11] (scaled? No, V is linear)
    # Wait, the tensor decomposition is T = sum u_v_w.
    # U[i,j] corresponds to coefficient of A[i,j].
    # V[j,k] corresponds to coefficient of B[j,k].
    # W[k,i] corresponds to coefficient of C[k,i] (in the sum).
    
    # Let's define the 7 Strassen terms for the 2x2x2 part (cols 0,1).
    # Indices for A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Indices for B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (relative to B_left)
    #   Global B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Indices for C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   Global C indices (k*n+i):
    #   k=0, i=0 -> 0
    #   k=0, i=1 -> 1
    #   k=1, i=0 -> 2
    #   k=1, i=1 -> 3
    
    # Strassen products:
    # P1 = A00 * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = (A00 + A01) * B11
    # P4 = A11 * (B01 - B00)
    # P5 = A00 * (B10 - B11)
    # P6 = (A10 - A00) * B00
    # P7 = (A01 - A11) * B11
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P7
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P6
    
    # Let's map these to U, V, W columns.
    # We have 7 columns from Strassen.
    
    # Column 0 (P1):
    # U[0,0]=1 (A00), U[1,0]=0, U[2,0]=0, U[3,0]=0
    # V[0,0]=1 (B00), V[1,0]=0, V[2,0]=0, V[3,0]=1 (B11)  (relative to B_left)
    #   Global V: V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=1
    # W[0,0]=1 (C00), W[1,0]=0, W[2,0]=0, W[3,0]=1 (C11)
    
    # Column 1 (P2):
    # U[2,1]=1 (A10), U[3,1]=1 (A11)
    # V[0,1]=1 (B00)
    # W[2,1]=1 (C10)
    
    # Column 2 (P3):
    # U[0,2]=1 (A00), U[1,2]=1 (A01)
    # V[3,2]=1 (B11)
    # W[1,2]=1 (C01), W[3,2]=1 (C11)
    
    # Column 3 (P4):
    # U[3,3]=1 (A11)
    # V[1,3]=1 (B01), V[0,3]=-1 (B00)
    # W[0,3]=1 (C00)
    
    # Column 4 (P5):
    # U[0,4]=1 (A00)
    # V[2,4]=1 (B10), V[3,4]=-1 (B11)
    # W[0,4]=-1 (C00), W[2,4]=1 (C10)
    
    # Column 5 (P6):
    # U[2,5]=1 (A10), U[0,5]=-1 (A00)
    # V[0,5]=1 (B00)
    # W[3,5]=1 (C11)
    
    # Column 6 (P7):
    # U[1,6]=1 (A01), U[3,6]=-1 (A11)
    # V[3,6]=1 (B11)
    # W[0,6]=-1 (C00), W[1,6]=1 (C01)
    
    # Now Schoolbook for C_right (cols 2,3,4 of B).
    # B_right indices:
    # (0,2)->4, (0,3)->5, (0,4)->6
    # (1,2)->7, (1,3)->8, (1,4)->9
    # C_right indices (k*n+i):
    # k=2: 4, 5
    # k=3: 6, 7
    # k=4: 8, 9
    
    # C[0,2] = A00*B02 + A01*B12  -> P8
    # C[1,2] = A10*B02 + A11*B12  -> P9
    # C[0,3] = A00*B03 + A01*B13  -> P10
    # C[1,3] = A10*B03 + A11*B13  -> P11
    # C[0,4] = A00*B04 + A01*B14  -> P12
    # C[1,4] = A10*B04 + A11*B14  -> P13
    
    # Wait, this is 6 more products. Total 7+6=13?
    # No, schoolbook for 2x2 by 2x3 is 2*2*3 = 12 products.
    # Let's list them:
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    # C03 = A00 B03 + A01 B13
    # C13 = A10 B03 + A11 B13
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # These are 12 terms if we don't factor?
    # A00 B02, A01 B12, A10 B02, A11 B12 (4 terms)
    # A00 B03, A01 B13, A10 B03, A11 B13 (4 terms)
    # A00 B04, A01 B14, A10 B04, A11 B14 (4 terms)
    # Total 12.
    # So 7 + 12 = 19. Correct.
    
    # Let's construct the matrices.
    # U: 4x19, V: 10x19, W: 10x19
    
    U = [[0]*19 for _ in range(4)]
    V = [[0]*19 for _ in range(10)]
    W = [[0]*19 for _ in range(10)]
    
    # Fill Strassen columns (0-6)
    # Col 0: P1
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # Col 1: P2
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Col 2: P3
    U[0][2] = 1
    U[1][2] = 1
    V[3][2] = 1
    W[1][2] = 1
    W[3][2] = 1
    
    # Col 3: P4
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    
    # Col 4: P5
    U[0][4] = 1
    V[2][4] = 1
    V[3][4] = -1
    W[0][4] = -1
    W[2][4] = 1
    
    # Col 5: P6
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    W[3][5] = 1
    
    # Col 6: P7
    U[1][6] = 1
    U[3][6] = -1
    V[3][6] = 1
    W[0][6] = -1
    W[1][6] = 1
    
    # Fill Schoolbook columns (7-18)
    # Col 7: A00 B02 -> C02
    U[0][7] = 1
    V[4][7] = 1  # B02
    W[4][7] = 1  # C02
    
    # Col 8: A01 B12 -> C02
    U[1][8] = 1
    V[7][8] = 1  # B12
    W[4][8] = 1  # C02
    
    # Col 9: A10 B02 -> C12
    U[2][9] = 1
    V[4][9] = 1  # B02
    W[5][9] = 1  # C12
    
    # Col 10: A11 B12 -> C12
    U[3][10] = 1
    V[7][10] = 1 # B12
    W[5][10] = 1 # C12
    
    # Col 11: A00 B03 -> C03
    U[0][11] = 1
    V[5][11] = 1 # B03
    W[6][11] = 1 # C03
    
    # Col 12: A01 B13 -> C03
    U[1][12] = 1
    V[8][12] = 1 # B13
    W[6][12] = 1 # C03
    
    # Col 13: A10 B03 -> C13
    U[2][13] = 1
    V[5][13] = 1 # B03
    W[7][13] = 1 # C13
    
    # Col 14: A11 B13 -> C13
    U[3][14] = 1
    V[8][14] = 1 # B13
    W[7][14] = 1 # C13
    
    # Col 15: A00 B04 -> C04
    U[0][15] = 1
    V[6][15] = 1 # B04
    W[8][15] = 1 # C04
    
    # Col 16: A01 B14 -> C04
    U[1][16] = 1
    V[9][16] = 1 # B14
    W[8][16] = 1 # C04
    
    # Col 17: A10 B04 -> C14
    U[2][17] = 1
    V[6][17] = 1 # B04
    W[9][17] = 1 # C14
    
    # Col 18: A11 B14 -> C14
    U[3][18] = 1
    V[9][18] = 1 # B14
    W[9][18] = 1 # C14
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook
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
