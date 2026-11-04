"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We want to decompose the matrix multiplication tensor for C = A @ B
    # where A is (n, m) = (3, 4) and B is (m, p) = (4, 8).
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i, j, k] = delta(i, j, k) means the (i,k) entry of C gets contribution from A[i,j]*B[j,k].
    # 
    # The output factor W follows the convention W[k*n+i, r].
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    #
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i).
    #
    # Approach: Use the known low-rank decomposition for 4x4 matrix multiplication (Laderman, rank 21).
    # We can embed our 3x4 @ 4x8 problem into a larger problem and use Kronecker products.
    #
    # Idea: 4x8 = (4x4) ⊗ (1x2) or we can view it as two 4x4 blocks.
    # Actually, let's think differently. We can use the fact that 8 = 2*4.
    # 
    # A cleaner approach: Use the known rank-21 Laderman algorithm for 4x4 @ 4x4.
    # We have A (3x4) and B (4x8). We can pad A to 4x4 (add a zero row) and split B into two 4x4 blocks.
    # Then C = A_padded @ B = [A_padded @ B1 | A_padded @ B2] where B1, B2 are 4x4.
    # The top 3 rows of each 4x4 result give us our desired output.
    #
    # But we need a single tensor decomposition. Let's use Kronecker product structure.
    #
    # Let's use the rank-21 Laderman decomposition for 4x4 @ 4x4:
    # For 4x4 @ 4x4, we have 21 terms. Each term r has:
    #   u_r: vector of length 16 (indices (i,j) for i,j in 0..3)
    #   v_r: vector of length 16 (indices (j,k) for j,k in 0..3)
    #   w_r: vector of length 16 (indices (k,i) for k,i in 0..3)
    #
    # For our problem: A is 3x4, B is 4x8.
    # Let's embed into 4x4 @ 4x8. Pad A to 4x4 with a zero row at the bottom.
    # Split B = [B1 | B2] where B1, B2 are 4x4.
    # C_padded = A_pad @ B = [A_pad @ B1 | A_pad @ B2]
    # The top 3 rows of each give us C.
    #
    # So C[i,k] for i in 0..2, k in 0..7.
    # C[i, k] for k in 0..3 comes from A_pad @ B1, row i.
    # C[i, k] for k in 4..7 comes from A_pad @ B2, row i.
    #
    # Using Laderman: (A_pad @ B1)[i, k] = sum_r u_r[i,j-ish] * v_r[j-ish,k] * w_r[k,i-ish]
    # This is getting complex. Let me just directly construct the tensors using known Laderman coefficients.
    
    # Laderman's algorithm for 4x4 matrix multiplication, rank 21.
    # Reference: standard Laderman algorithm.
    # Indices: A and B are 4x4, indexed 0..3.
    
    # I'll use a known set of Laderman coefficients.
    # Each term: (a_coeffs, b_coeffs, c_coeffs) where a_coeffs is a dict from (i,j) to value, etc.
    
    laderman_4x4 = [
        # P1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A00-A11)(B00-B11)  
        ({(0,0):1, (1,1):-1}, {(0,0):1, (1,1):-1}, {(0,0):1, (1,1):1}),
        # P3 = (A00+A22)(B00+B22)
        ({(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}, {(0,0):1, (2,2):1}),
        # P4 = (A00-A22)(B00-B22)
        ({(0,0):1, (2,2):-1}, {(0,0):1, (2,2):-1}, {(0,0):1, (2,2):1}),
        # P5 = (A10+A11+A21+A22)(B10+B20+B11+B21)
        ({(1,0):1, (1,1):1, (2,1):1, (2,2):1}, {(1,0):1, (2,0):1, (1,1):1, (2,1):1}, {(1,1):1, (2,2):1}),
        # P6 = (A11-A22+A00+A10)(B11-B22+B00+B10)
        ({(1,1):1, (2,2):-1, (0,0):1, (1,0):1}, {(1,1):1, (2,2):-1, (0,0):1, (1,0):1}, {(1,0):1, (0,1):1}),
        # P7 = (A02+A12+A01+A11)(B01+B11+B02+B12)
        ({(0,2):1, (1,2):1, (0,1):1, (1,1):1}, {(0,1):1, (1,1):1, (0,2):1, (1,2):1}, {(0,2):1, (1,1):1}),
        # P8 = (A02-A12+A00+A01)(B02-A12+B00+A01)
        ({(0,2):1, (1,2):-1, (0,0):1, (0,1):1}, {(0,2):1, (1,2):-1, (0,0):1, (0,1):1}, {(0,1):1, (1,0):1}),
        # P9 = (A20+A30+A21+A31)(B20+B30+B21+B31)
        ({(2,0):1, (3,0):1, (2,1):1, (3,1):1}, {(2,0):1, (3,0):1, (2,1):1, (3,1):1}, {(2,0):1, (3,1):1}),
        # P10 = (A31-A20+A22+A30)(B31-A20+B22+B30)
        ({(3,1):1, (2,0):-1, (2,2):1, (3,0):1}, {(3,1):1, (2,0):-1, (2,2):1, (3,0):1}, {(2,1):1, (3,0):1}),
        # P11 = (A23+A33+A22+A32)(B22+B32+B23+B33)
        ({(2,3):1, (3,3):1, (2,2):1, (3,2):1}, {(2,2):1, (3,2):1, (2,3):1, (3,3):1}, {(2,3):1, (3,2):1}),
        # P12 = (A23-A33+A20+A21)(B23-A33+B20+B21)
        ({(2,3):1, (3,3):-1, (2,0):1, (2,1):1}, {(2,3):1, (3,3):-1, (2,0):1, (2,1):1}, {(2,2):1, (3,1):1}),
        # P13 = (A01+A21)(B01+B21)
        ({(0,1):1, (2,1):1}, {(0,1):1, (2,1):1}, {(1,0):1, (2,1):1}),
        # P14 = (A01-A21)(B01-B21)
        ({(0,1):1, (2,1):-1}, {(0,1):1, (2,1):-1}, {(1,0):1, (2,1):1}),
        # P15 = (A10+A30)(B10+B30)
        ({(1,0):1, (3,0):1}, {(1,0):1, (3,0):1}, {(0,1):1, (3,0):1}),
        # P16 = (A10-A30)(B10-B30)
        ({(1,0):1, (3,0):-1}, {(1,0):1, (3,0):-1}, {(0,1):1, (3,0):1}),
        # P17 = (A13+A33+A12+A32)(B12+B32+B13+B33)
        ({(1,3):1, (3,3):1, (1,2):1, (3,2):1}, {(1,2):1, (3,2):1, (1,3):1, (3,3):1}, {(1,2):1, (3,3):1}),
        # P18 = (A13-A33+A10+A11)(B13-A33+B10+B11)
        ({(1,3):1, (3,3):-1, (1,0):1, (1,1):1}, {(1,3):1, (3,3):-1, (1,0):1, (1,1):1}, {(1,3):1, (3,2):1}),
        # P19 = (A20+A00)(B20+B00)
        ({(2,0):1, (0,0):1}, {(2,0):1, (0,0):1}, {(0,2):1, (2,0):1}),
        # P20 = (A20-A00)(B20-B00)
        ({(2,0):1, (0,0):-1}, {(2,0):1, (0,0):-1}, {(0,2):1, (2,0):1}),
        # P21 = (A31+A11)(B31+B11)
        ({(3,1):1, (1,1):1}, {(3,1):1, (1,1):1}, {(1,3):1, (3,1):1}),
    ]
    
    # This is getting very complex and error-prone to verify by hand.
    # Let me take a simpler but still improved approach: use the fact that we can
    # decompose the 4x8 part using the structure 8 = 2*4.
    #
    # Actually, let me just use the schoolbook decomposition but with a more efficient
    # indexing, and then try to improve using known low-rank algorithms for sub-blocks.
    #
    # For now, let me return a valid schoolbook decomposition. The rank is 3*4*8 = 96.
    # This is the baseline. Any improvement needs to be verified.
    
    rank = n * m * p  # 96
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
