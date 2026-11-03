"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses a block decomposition:
    - Split the 4x4 matrix into 2x2 blocks of 2x2 matrices
    - Use Strassen's algorithm for the 2x2 block multiplication (rank 7 per block)
    - Handle the 6 columns appropriately
    
    The 4x4 * 4x6 multiplication can be viewed as:
    - 4x4 matrix A split into 2x2 blocks of 2x2
    - 4x6 matrix B split into 2x3 blocks of 2x2 (last block is 2x2 with zero padding)
    
    Actually, let me think more carefully.
    
    A is 4x4, B is 4x6, result C is 4x6.
    
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    Split B into 2x3 blocks: B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    
    Then C = A*B where:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    Each of these is a 2x2 * 2x2 multiplication, which has rank 7 using Strassen.
    There are 6 such multiplications, so total rank = 6 * 7 = 42.
    
    But we can do better by sharing some of the Strassen intermediate products.
    
    Actually, let's just use the simple block approach first:
    - 6 independent 2x2*2x2 multiplications, each rank 7
    - Total rank = 42
    
    This is much better than the schoolbook rank of 4*4*6 = 96.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition using Strassen for 2x2*2x2 blocks.
    # 
    # Strassen's algorithm for 2x2*2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # Strassen uses 7 products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # 
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # For each 2x2*2x2 multiplication, we have 7 rank-1 terms.
    # 
    # The 6 block multiplications are:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Each requires 7 rank-1 terms, total 42.
    
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen coefficients for 2x2*2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # For a 2x2 matrix [[x, y], [z, w]], the indices in the full matrix are:
    # (row_offset + 0)*4 + col_offset + 0 -> x
    # (row_offset + 0)*4 + col_offset + 1 -> y
    # (row_offset + 1)*4 + col_offset + 0 -> z
    # (row_offset + 1)*4 + col_offset + 1 -> w
    
    # Strassen term definitions:
    # Each term is (u_vec, v_vec, w_vec) where:
    # u_vec has entries in U (for A matrix, size 16)
    # v_vec has entries in V (for B matrix, size 24)
    # w_vec has entries in W (for C matrix, size 24)
    
    # Let's define the Strassen terms for a generic 2x2*2x2 multiplication
    # where A block is at rows [r0, r0+1], cols [c0, c0+1]
    # and B block is at rows [r0, r0+1], cols [c0, c0+1]
    # and C block is at rows [r0, r0+1], cols [c0, c0+1]
    
    def strassen_terms(r0, c0):
        """
        Return the 7 (u, v, w) triples for Strassen's 2x2*2x2 multiplication.
        r0, c0 are the top-left corner indices of the 2x2 blocks in the full matrices.
        
        U indices: (r0+dr)*4 + (c0+dc) for dr,dc in {0,1}
        V indices: (r0+dr)*6 + (c0+dc) for dr,dc in {0,1}
        W indices: (c0+dc)*4 + (r0+dr) for dr,dc in {0,1}
        """
        # A elements: A[r0, c0], A[r0, c0+1], A[r0+1, c0], A[r0+1, c0+1]
        # In U: row = r*4 + c
        a_idx = r0 * 4 + c0
        b_idx = r0 * 4 + c0 + 1
        c_idx = (r0 + 1) * 4 + c0
        d_idx = (r0 + 1) * 4 + c0 + 1
        
        # B elements: B[r0, c0], B[r0, c0+1], B[r0+1, c0], B[r0+1, c0+1]
        # In V: row = r*6 + c
        e_idx = r0 * 6 + c0
        f_idx = r0 * 6 + c0 + 1
        g_idx = (r0 + 1) * 6 + c0
        h_idx = (r0 + 1) * 6 + c0 + 1
        
        # C elements: C[r0, c0], C[r0, c0+1], C[r0+1, c0], C[r0+1, c0+1]
        # In W: index = c*4 + r
        c11_idx = c0 * 4 + r0
        c12_idx = c0 * 4 + r0 + 1  # Wait, this is wrong. Let me recheck.
        
        # W[k*n+i, r] where k is column of C, i is row of C
        # C[i, k] -> W[k*n + i]
        # So C[r0, c0] -> W[c0*4 + r0]
        # C[r0, c0+1] -> W[(c0+1)*4 + r0]
        # C[r0+1, c0] -> W[c0*4 + r0+1]
        # C[r0+1, c0+1] -> W[(c0+1)*4 + r0+1]
        
        c11_idx = c0 * 4 + r0
        c12_idx = (c0 + 1) * 4 + r0
        c21_idx = c0 * 4 + r0 + 1
        c22_idx = (c0 + 1) * 4 + r0 + 1
        
        terms = []
        
        # P1 = (a+d)(e+h)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[a_idx] = 1; u[d_idx] = 1
        v[e_idx] = 1; v[h_idx] = 1
        w[c11_idx] = 1; w[c22_idx] = 1
        w[c12_idx] = -1; w[c21_idx] = 1  # C11 += P1, C22 += P1, C12 -= P1, C21 += P1? No...
        
        # Wait, I need to be more careful. Let me re-derive.
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So for P1: contributes +1 to C11, +1 to C22
        # For P2: contributes +1 to C21, -1 to C22
        # For P3: contributes +1 to C12, +1 to C22
        # For P4: contributes -1 to C11, +1 to C21
        # For P5: contributes +1 to C11, +1 to C12
        # For P6: contributes +1 to C22
        # For P7: contributes -1 to C11
        
        # Let me redo this properly.
        
        terms = []
        
        # P1 = (a+d)(e+h)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[a_idx] = 1; u[d_idx] = 1
        v[e_idx] = 1; v[h_idx] = 1
        w[c11_idx] = 1; w[c22_idx] = 1
        terms.append((u, v, w))
        
        # P2 = (c+d)e
        u = [0]*16; v = [0]*24; w = [0]*24
        u[c_idx] = 1; u[d_idx] = 1
        v[e_idx] = 1
        w[c21_idx] = 1; w[c22_idx] = -1
        terms.append((u, v, w))
        
        # P3 = a(f-h)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[a_idx] = 1
        v[f_idx] = 1; v[h_idx] = -1
        w[c12_idx] = 1; w[c22_idx] = 1
        terms.append((u, v, w))
        
        # P4 = d(g-e)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[d_idx] = 1
        v[g_idx] = 1; v[e_idx] = -1
        w[c11_idx] = -1; w[c21_idx] = 1
        terms.append((u, v, w))
        
        # P5 = (a+b)h
        u = [0]*16; v = [0]*24; w = [0]*24
        u[a_idx] = 1; u[b_idx] = 1
        v[h_idx] = 1
        w[c11_idx] = 1; w[c12_idx] = 1
        terms.append((u, v, w))
        
        # P6 = (c-a)(e+f)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[c_idx] = 1; u[a_idx] = -1
        v[e_idx] = 1; v[f_idx] = 1
        w[c22_idx] = 1
        terms.append((u, v, w))
        
        # P7 = (b-d)(g+h)
        u = [0]*16; v = [0]*24; w = [0]*24
        u[b_idx] = 1; u[d_idx] = -1
        v[g_idx] = 1; v[h_idx] = 1
        w[c11_idx] = -1
        terms.append((u, v, w))
        
        return terms
    
    # Now, for each of the 6 block multiplications, we add 7 Strassen terms.
    # Block (i, j) where i in {0,1}, j in {0,1,2}:
    # A block at rows [2*i, 2*i+1], cols [2*i, 2*i+1]
    # B block at rows [2*i, 2*i+1], cols [2*j, 2*j+1]
    # C block at rows [2*i, 2*i+1], cols [2*j, 2*j+1]
    
    col_offset = 0
    for i in range(2):
        for j in range(3):
            r0 = 2 * i
            c0 = 2 * j
            terms = strassen_terms(r0, c0)
            for u, v, w in terms:
                for row in range(16):
                    U[row][col_offset] += u[row]
                for row in range(24):
                    V[row][col_offset] += v[row]
                for row in range(24):
                    W[row][col_offset] += w[row]
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
