"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target rank: 60
    # Structure:
    # Part 1: j=2. M_{3,1,7}. Rank 21.
    # Part 2: j in {0,1}. M_{3,2,7}.
    #   Sub-part 2a: i=2. M_{1,2,7}. Rank 14.
    #   Sub-part 2b: i in {0,1}. M_{2,2,7}. Rank 25 (using Strassen for 2x2x2 blocks).
    # Total Rank = 21 + 14 + 25 = 60.
    
    rank = 60
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Part 1: j = 2
    # Terms: A[i,2] * B[2,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            U[i*m + 2, col] = 1.0
            V[2*p + k, col] = 1.0
            W[k*n + i, col] = 1.0
            col += 1
    
    # Part 2a: j in {0,1}, i = 2
    # Terms: A[2,j] * B[j,k] -> C[2,k]
    for j in range(2):
        for k in range(p):
            U[2*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + 2, col] = 1.0
            col += 1
            
    # Part 2b: j in {0,1}, i in {0,1}
    # M_{2,2,7}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    
    def add_strassen_2x2x2(i0, k0, j0):
        """
        Adds a 7-term Strassen decomposition for M_{2,2,2}
        A is 2x2 (rows i0..i0+1, cols j0..j0+1)
        B is 2x2 (rows j0..j0+1, cols k0..k0+1)
        C is 2x2 (rows i0..i0+1, cols k0..k0+1)
        """
        nonlocal col
        # Strassen's 7 products for 2x2x2
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A10 - A00) * B00
        # M4 = A11 * (B00 - B10)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A00 + A01) * (B01 - B10)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # We need to map the local 2x2 indices to global indices.
        # i_local in {0,1} -> i_global = i0 + i_local
        # j_local in {0,1} -> j_global = j0 + j_local
        # k_local in {0,1} -> k_global = k0 + k_local
        
        # Term 1: M1 = A[i0, j0] * (B[j0, k0+1] - B[j0+1, k0+1])
        # U: A[i0, j0] -> 1
        # V: B[j0, k0+1] -> 1, B[j0+1, k0+1] -> -1
        # W: C[i0, k0+1] -> 1, C[i0+1, k0+1] -> 1
        # Wait, Strassen reconstruction:
        # C00 = M5 + M4 - M2 - M7
        # C01 = M1 + M7
        # C10 = M3 + M5
        # C11 = M2 - M1 + M3 + M6
        
        # Let's add the 7 terms.
        
        # Term 1: M1
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 2: M2
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + j0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[(k0+1)*n + i0, col] = -1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 3: M3
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 4: M4
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 5: M5
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 6: M6
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # Term 7: M7
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[k0*n + i0, col] = -1.0
        col += 1

    # Apply Strassen for k blocks {0,1}, {2,3}, {4,5}
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 4, 0)
    
    # Apply Schoolbook for k block {6}
    # i in {0,1}, j in {0,1}, k=6
    for i in range(2):
        for j in range(2):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
            col += 1
            
    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if something went wrong
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        rank = c

    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
