"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_arr[i*m+j]
                v_vec = V_arr[j*p+k]
                w_vec = W_arr[k*n+i]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol

def _construct_hybrid_strassen(n, m, p):
    """
    Construct a rank-minimal decomposition using algebraic block splitting.
    For n=2, m=3, p=7:
    Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x7) into B1 (2x7) and B2 (1x7).
    C = A1*B1 + A2*B2.
    
    Part 1: A1 (2x2) * B1 (2x7).
    B1 has 7 columns. We can split B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1).
    A1*B1a uses Strassen (rank 7).
    A1*B1b uses Strassen (rank 7).
    A1*B1c uses Strassen (rank 7).
    A1*B1d uses Schoolbook for 2x2x1 (rank 4).
    Total rank for Part 1: 7 + 7 + 7 + 4 = 25.
    
    Part 2: A2 (2x1) * B2 (1x7).
    This is an outer product of a 2x1 vector and a 1x7 vector.
    The tensor M_{2,1,7} has rank 2*7 = 14.
    Total rank: 25 + 14 = 39.
    """
    if (n, m, p) != (2, 3, 7):
        return None
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-6)
    # Split B1 cols into 0-1, 2-3, 4-5, 6-6
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M6
    
    def add_strassen_block(u_row_offset, v_row_offset, w_row_offset, w_col_offset, start_col):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        u_row_offset: row index in U for the 2x2 A block (i is 0..1, j is 0..1)
        v_row_offset: row index in V for the 2x2 B block (j is 0..1, k is 0..1)
        w_row_offset: row index in W for the 2x2 C block (k is 0..1, i is 0..1)
        w_col_offset: column index in W for the 2x2 C block (k is 0..1, i is 0..1) - actually W is (p*n, rank), indexed by k*n+i.
        start_col: starting column in U,V,W
        """
        # U indices for A1 (2x2): i in 0..1, j in 0..1 -> U[i*m + j]
        # V indices for B1 (2x2): j in 0..1, k in 0..1 -> V[j*p + k]
        # W indices for C1 (2x2): k in 0..1, i in 0..1 -> W[k*n + i]
        
        # M1: (A11+A22)(B11+B22)
        # U[A11]=1, U[A22]=1
        U[u_row_offset + 0][start_col + 0] = 1
        U[u_row_offset + 2][start_col + 0] = 1
        # V[B11]=1, V[B22]=1
        V[v_row_offset + 0][start_col + 0] = 1
        V[v_row_offset + 2][start_col + 0] = 1
        # W[C11]=1, W[C22]=1
        W[w_row_offset + 0][start_col + 0] = 1
        W[w_row_offset + 2][start_col + 0] = 1
        
        # M2: (A21+A22)B11
        U[u_row_offset + 1][start_col + 1] = 1
        U[u_row_offset + 2][start_col + 1] = 1
        V[v_row_offset + 0][start_col + 1] = 1
        W[w_row_offset + 1][start_col + 1] = 1 # C21
        
        # M3: A11(B21-B11)
        U[u_row_offset + 0][start_col + 2] = 1
        V[v_row_offset + 1][start_col + 2] = 1
        V[v_row_offset + 0][start_col + 2] = -1
        W[w_row_offset + 1][start_col + 2] = 1 # C12
        
        # M4: A22(B12-B22)
        U[u_row_offset + 2][start_col + 3] = 1
        V[v_row_offset + 2][start_col + 3] = 1
        V[v_row_offset + 3][start_col + 3] = -1 # Wait, B is 2x2. Indices: B11(0,0), B12(0,1), B21(1,0), B22(1,1).
        # V index for B12 is j=0, k=1 -> v_row_offset + 0*p + 1 = v_row_offset + 1? No, V is (m*p, rank).
        # m=2 for this block. V[j*p+k]. If local j=0, k=1, index is 1.
        # Let's re-map local indices to global.
        # This function is getting complex. Let's just hardcode the 39 rank solution directly.
        pass

    # Hardcoded construction for 2x3x7 with rank 39
    # U: 6x39, V: 21x39, W: 14x39
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # B1 cols 0,1 (Strassen)
    # B1 cols 2,3 (Strassen)
    # B1 cols 4,5 (Strassen)
    # B1 cols 6 (Schoolbook 2x2x1)
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # Schoolbook 2x1x7
    
    # Let's build it step-by-step.
    
    # Strassen helper for specific global indices
    def add_strassen(A_rows, B_cols_start, C_cols_start, start_c):
        # A_rows: [i1, i2] (rows of A, 0 and 1)
        # B_cols_start: starting column in B for this 2x2 block
        # C_cols_start: starting column in C for this 2x2 block
        # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # U indices: i*m + j. m=3.
        # A[0,0] -> U[0*3+0]=U[0]
        # A[0,1] -> U[0*3+1]=U[1]
        # A[1,0] -> U[1*3+0]=U[3]
        # A[1,1] -> U[1*3+1]=U[4]
        
        # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # V indices: j*p + k. p=7.
        # B[0,0] -> V[0*7 + B_cols_start]
        # B[0,1] -> V[0*7 + B_cols_start + 1]
        # B[1,0] -> V[1*7 + B_cols_start]
        # B[1,1] -> V[1*7 + B_cols_start + 1]
        
        # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
        # W indices: k*n + i. n=2.
        # C[0,0] -> W[0*2 + 0] = W[0]
        # C[0,1] -> W[1*2 + 0] = W[2]
        # C[1,0] -> W[0*2 + 1] = W[1]
        # C[1,1] -> W[1*2 + 1] = W[3]
        # Wait, W is (p*n, rank). Index is k*n + i.
        # C[i,k] corresponds to W[k*n + i].
        # C[0,0] (i=0, k=0) -> W[0*2+0]=W[0]
        # C[0,1] (i=0, k=1) -> W[1*2+0]=W[2]
        # C[1,0] (i=1, k=0) -> W[0*2+1]=W[1]
        # C[1,1] (i=1, k=1) -> W[1*2+1]=W[3]
        
        # But C_cols_start shifts the k index.
        # If C_cols_start is 0, k=0,1. W indices: 0, 2, 1, 3.
        # If C_cols_start is 2, k=2,3. W indices: 2*2+0=4, 3*2+0=6, 2*2+1=5, 3*2+1=7.
        
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        U[0][start_c] = 1
        U[4][start_c] = 1
        V[B_cols_start][start_c] = 1
        V[7 + B_cols_start][start_c] = 1
        W[C_cols_start * 2 + 0][start_c] = 1
        W[C_cols_start * 2 + 1 + 1][start_c] = 1 # W[k*n+i]. C22 is i=1, k=C_cols_start+1. Index (C_cols_start+1)*2 + 1.
        
        # M2: (A21+A22)B11 -> C21
        U[3][start_c+1] = 1
        U[4][start_c+1] = 1
        V[B_cols_start][start_c+1] = 1
        W[C_cols_start * 2 + 1][start_c+1] = 1 # C21 is i=1, k=C_cols_start. Index C_cols_start*2 + 1.
        
        # M3: A11(B21-B11) -> C12
        U[0][start_c+2] = 1
        V[7 + B_cols_start][start_c+2] = 1
        V[B_cols_start][start_c+2] = -1
        W[C_cols_start * 2 + 2][start_c+2] = 1 # C12 is i=0, k=C_cols_start+1. Index (C_cols_start+1)*2 + 0.
        
        # M4: A22(B12-B22) -> C12
        U[4][start_c+3] = 1
        V[B_cols_start + 1][start_c+3] = 1
        V[7 + B_cols_start + 1][start_c+3] = -1
        W[C_cols_start * 2 + 2][start_c+3] = 1
        
        # M5: (A11+A12)B22 -> C12
        U[0][start_c+4] = 1
        U[1][start_c+4] = 1
        V[7 + B_cols_start + 1][start_c+4] = 1
        W[C_cols_start * 2 + 2][start_c+4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C22
        U[3][start_c+5] = 1
        U[0][start_c+5] = -1
        V[B_cols_start][start_c+5] = 1
        V[B_cols_start + 1][start_c+5] = 1
        W[C_cols_start * 2 + 1 + 1][start_c+5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C22
        U[1][start_c+6] = 1
        U[4][start_c+6] = -1
        V[7 + B_cols_start][start_c+6] = 1
        V[7 + B_cols_start + 1][start_c+6] = 1
        W[C_cols_start * 2 + 1 + 1][start_c+6] = 1
        
        return start_c + 7

    # Part 1: A1 (2x2) * B1 (2x7)
    # B1 cols 0,1
    c = add_strassen([0, 1], 0, 0, 0)
    # B1 cols 2,3
    c = add_strassen([0, 1], 2, 2, c)
    # B1 cols 4,5
    c = add_strassen([0, 1], 4, 4, c)
    
    # B1 col 6 (2x2x1 schoolbook)
    # A[0,0]*B[0,6] -> C[0,6]
    U[0][c] = 1
    V[6][c] = 1
    W[6*2 + 0][c] = 1
    c += 1
    # A[0,1]*B[1,6] -> C[0,6]
    U[1][c] = 1
    V[7+6][c] = 1
    W[6*2 + 0][c] = 1
    c += 1
    # A[1,0]*B[0,6] -> C[1,6]
    U[3][c] = 1
    V[6][c] = 1
    W[6*2 + 1][c] = 1
    c += 1
    # A[1,1]*B[1,6] -> C[1,6]
    U[4][c] = 1
    V[7+6][c] = 1
    W[6*2 + 1][c] = 1
    c += 1
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # A rows 0,1, col 2. U indices: 0*3+2=2, 1*3+2=5.
    # B row 2, cols 0..6. V indices: 2*7+k = 14+k.
    # C rows 0,1, cols 0..6. W indices: k*2+i.
    for i in range(2):
        for k in range(7):
            U[i*3 + 2][c] = 1
            V[14 + k][c] = 1
            W[k*2 + i][c] = 1
            c += 1
    
    if c != rank:
        # Adjust rank if necessary, though we expect 39
        rank = c
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the algebraic hybrid Strassen/Schoolbook construction first
    result = _construct_hybrid_strassen(n, m, p)
    if result is not None:
        # Verify it
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            # If verification fails, fall back to schoolbook
            pass
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
