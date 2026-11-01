import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (6, 22)
    # V shape: (m*p, rank) = (12, 22)
    # W shape: (p*n, rank) = (8, 22)
    
    rank = 22
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to index U: A[i, k] -> i*m + k
    # Helper to index V: B[k, j] -> k*p + j
    # Helper to index W: C[i, j] -> j*n + i (based on W[k*n+i] where k is col j)
    
    def add_mult(r, a_coeffs, b_coeffs, c_coeffs):
        """
        r: rank index (0-based)
        a_coeffs: list of (i, k, coeff)
        b_coeffs: list of (k, j, coeff)
        c_coeffs: list of (i, j, coeff)
        """
        for i, k, c in a_coeffs:
            U[i * m + k, r] += c
        for k, j, c in b_coeffs:
            V[k * p + j, r] += c
        for i, j, c in c_coeffs:
            W[j * n + i, r] += c

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # M1 = (A11+A22)(B11+B22)
    add_mult(0, 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1)], 
             [(0,0,1), (1,1,1)])
             
    # M2 = (A21+A22)B11
    add_mult(1, 
             [(1,0,1), (1,1,1)], 
             [(0,0,1)], 
             [(1,0,1)]) # C21 += M2
    
    # M3 = A11(B12-B22)
    add_mult(2, 
             [(0,0,1)], 
             [(0,1,1), (1,1,-1)], 
             [(0,1,1)]) # C12 += M3

    # M4 = A22(B21-B11)
    add_mult(3, 
             [(1,1,1)], 
             [(1,0,1), (0,0,-1)], 
             [(0,0,1), (1,0,1)]) # C11 += M4, C21 += M4

    # M5 = (A11+A12)B22
    add_mult(4, 
             [(0,0,1), (0,1,1)], 
             [(1,1,1)], 
             [(0,1,1), (1,1,-1)]) # C12 += M5, C22 -= M5

    # M6 = (A21-A11)(B11+B12)
    add_mult(5, 
             [(1,0,1), (0,0,-1)], 
             [(0,0,1), (0,1,1)], 
             [(1,1,1)]) # C22 += M6

    # M7 = (A12-A22)(B21+B22)
    add_mult(6, 
             [(0,1,1), (1,1,-1)], 
             [(1,0,1), (1,1,1)], 
             [(0,0,1), (1,1,-1)]) # C11 += M7, C22 -= M7

    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # B11=B[0,2], B12=B[0,3], B21=B[1,2], B22=B[1,3]
    
    # M8 = (A11+A22)(B11+B22)
    add_mult(7, 
             [(0,0,1), (1,1,1)], 
             [(0,2,1), (1,3,1)], 
             [(0,2,1), (1,3,1)])
             
    # M9 = (A21+A22)B11
    add_mult(8, 
             [(1,0,1), (1,1,1)], 
             [(0,2,1)], 
             [(1,2,1)])
    
    # M10 = A11(B12-B22)
    add_mult(9, 
             [(0,0,1)], 
             [(0,3,1), (1,3,-1)], 
             [(0,3,1)])

    # M11 = A22(B21-B11)
    add_mult(10, 
             [(1,1,1)], 
             [(1,2,1), (0,2,-1)], 
             [(0,2,1), (1,2,1)])

    # M12 = (A11+A12)B22
    add_mult(11, 
             [(0,0,1), (0,1,1)], 
             [(1,3,1)], 
             [(0,3,1), (1,3,-1)])

    # M13 = (A21-A11)(B11+B12)
    add_mult(12, 
             [(1,0,1), (0,0,-1)], 
             [(0,2,1), (0,3,1)], 
             [(1,3,1)])

    # M14 = (A12-A22)(B21+B22)
    add_mult(13, 
             [(0,1,1), (1,1,-1)], 
             [(1,2,1), (1,3,1)], 
             [(0,2,1), (1,3,-1)])

    # Part 3: Schoolbook for A[0:2, 2:3] * B[2:3, 0:4]
    # A13=A[0,2], A23=A[1,2]
    # B31=B[2,0], B32=B[2,1], B33=B[2,2], B34=B[2,3]
    
    # M15 = A13 * B31
    add_mult(14, 
             [(0,2,1)], 
             [(2,0,1)], 
             [(0,0,1)])
             
    # M16 = A13 * B32
    add_mult(15, 
             [(0,2,1)], 
             [(2,1,1)], 
             [(0,1,1)])

    # M17 = A13 * B33
    add_mult(16, 
             [(0,2,1)], 
             [(2,2,1)], 
             [(0,2,1)])

    # M18 = A13 * B34
    add_mult(17, 
             [(0,2,1)], 
             [(2,3,1)], 
             [(0,3,1)])

    # M19 = A23 * B31
    add_mult(18, 
             [(1,2,1)], 
             [(2,0,1)], 
             [(1,0,1)])

    # M20 = A23 * B32
    add_mult(19, 
             [(1,2,1)], 
             [(2,1,1)], 
             [(1,1,1)])

    # M21 = A23 * B33
    add_mult(20, 
             [(1,2,1)], 
             [(2,2,1)], 
             [(1,2,1)])

    # M22 = A23 * B34
    add_mult(21, 
             [(1,2,1)], 
             [(2,3,1)], 
             [(1,3,1)])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
