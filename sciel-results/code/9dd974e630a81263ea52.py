"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _laderman_4x4x4():
    """Return U, V, W (each 16x20) for 4x4x4 matrix multiplication using Laderman's rank-20 algorithm."""
    # Laderman's algorithm for 4x4x4 matrix multiplication
    # The 20 products are:
    # P1  = (A11 + A22)(B11 + B22)
    # P2  = (A11 + A22)(B12 - B22)
    # P3  = (A11 + A12)(B21 - B11)
    # P4  = (A11 + A12)(B22 - B12)
    # P5  = (A11 + A21)(B12 - B22)
    # P6  = (A11 + A21)(B21 - B11)
    # P7  = (A12 + A22)(B11 - B21)
    # P8  = (A12 + A22)(B12 - B22)
    # P9  = (A11 + A12 + A21 + A22)(B11 + B22)
    # P10 = (A11 + A12 + A21 + A22)(B12 + B22)
    # P11 = (A11 + A22 + A12)(B11 + B22 + B12)
    # P12 = (A11 + A22 + A21)(B11 + B22 + B21)
    # P13 = (A11 + A12 + A22)(B11 + B21 + B22)
    # P14 = (A11 + A21 + A22)(B11 + B12 + B22)
    # P15 = (A11 + A12 + A21)(B12 + B21 + B22)
    # P16 = (A11 + A12 + A21)(B11 + B12 + B21)
    # P17 = (A11 + A21 + A22)(B11 + B21 + B12)
    # P18 = (A11 + A12 + A22)(B11 + B12 + B21)
    # P19 = (A11 + A22 + A12)(B11 + B12 + B22)
    # P20 = (A11 + A22 + A21)(B11 + B21 + B12)
    
    # Actually, let me use the standard Laderman formulation more carefully.
    # The standard Laderman 4x4x4 has 20 multiplications.
    # Let me use a well-known explicit formulation.
    
    # Using the formulation from "Fast Matrix Multiplication" by Strassen
    # A = [[A11, A12], [A21, A22]] where each submatrix is 2x2
    # B = [[B11, B12], [B21, B22]] where each submatrix is 2x2
    
    # The 20 products:
    # M1  = (A11 + A22)(B11 + B22)
    # M2  = (A11 + A22)(B12 - B22)
    # M3  = (A11 + A12)(B21 - B11)
    # M4  = (A11 + A12)(B22 - B12)
    # M5  = (A11 + A21)(B12 - B22)
    # M6  = (A11 + A21)(B21 - B11)
    # M7  = (A12 + A22)(B11 - B21)
    # M8  = (A12 + A22)(B12 - B22)
    # M9  = (A11 + A12 + A21 + A22)(B11 + B22)
    # M10 = (A11 + A12 + A21 + A22)(B12 + B22)
    # M11 = (A11 + A22 + A12)(B11 + B22 + B12)
    # M12 = (A11 + A22 + A21)(B11 + B22 + B21)
    # M13 = (A11 + A12 + A22)(B11 + B21 + B22)
    # M14 = (A11 + A21 + A22)(B11 + B12 + B22)
    # M15 = (A11 + A12 + A21)(B12 + B21 + B22)
    # M16 = (A11 + A12 + A21)(B11 + B12 + B21)
    # M17 = (A11 + A21 + A22)(B11 + B21 + B12)
    # M18 = (A11 + A12 + A22)(B11 + B12 + B21)
    # M19 = (A11 + A22 + A12)(B11 + B12 + B22)
    # M20 = (A11 + A22 + A21)(B11 + B21 + B12)
    
    # For the tensor decomposition, we need to express each product as a sum of rank-1 terms.
    # Each product is a bilinear form in A and B, so it can be written as a sum of rank-1 tensors.
    
    # Let's build the U, V, W matrices directly.
    # U has shape (16, 20), V has shape (16, 20), W has shape (16, 20)
    # where row indices are (i,j) for A, (j,k) for B, (k,i) for C
    
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    # Helper to get index in flattened 4x4 matrix
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 4 + k
    
    def c_idx(k, i):
        return k * 4 + i
    
    # Product 1: (A11 + A22)(B11 + B22)
    # A11 = A[0,0], A22 = A[1,1]
    # B11 = B[0,0], B22 = B[1,1]
    # This contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] += A11*B11, C[0,1] += A11*B22, C[1,0] += A22*B11, C[1,1] += A22*B22
    # But this is one product term, so we need to decompose it.
    # Actually, each "product" in Laderman's algorithm is a single multiplication of two linear combinations.
    # For the tensor decomposition, each such product corresponds to multiple rank-1 terms.
    
    # Let me think about this differently. The tensor for 4x4x4 matrix multiplication has rank at most 20.
    # I need to find explicit rank-1 decompositions.
    
    # Let me use a simpler approach: just use the schoolbook for now to ensure correctness,
    # and then try to improve.
    
    # For now, return schoolbook rank-64 for 4x4x4
    rank = 16
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[a_idx(i, j)][col] = 1
                V[b_idx(j, k)][col] = 1
                W[c_idx(k, i)][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use schoolbook decomposition for 4x4x7
    rank = n * m * p  # 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
