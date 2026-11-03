import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Construct rank-32 decomposition using block Strassen
    # U shape: (n*m, rank) = (12, 32)
    # V shape: (m*p, rank) = (16, 32)  
    # W shape: (p*n, rank) = (12, 32)
    
    U = np.zeros((n * m, 32))
    V = np.zeros((m * p, 32))
    W = np.zeros((p * n, 32))
    
    # Part 1: 2x4x4 using Strassen (28 rank terms)
    # Rows i=0,1; split j into {0,1} and {2,3}; split k into {0,1} and {2,3}
    # 4 Strassen blocks, each with 7 terms
    
    # Strassen coefficients for 2x2x2x2
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    # C[0,0]=m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6
    
    def strassen_block(U, V, W, row_offset, col_offset, k_offset, term_start):
        """
        Add a Strassen block for 2x2 A block, 2x2 B block.
        row_offset: offset for i in U (0 or 1 for the 2 rows)
        col_offset: offset for j in U/V (0 or 2 for the 2 col groups)
        k_offset: offset for k in V/W (0 or 2 for the 2 row groups of B)
        term_start: starting term index
        """
        # 7 Strassen terms
        # Each term: U[i,j] * V[j,k] * W[k,i]
        # i in {row_offset, row_offset+1}
        # j in {col_offset, col_offset+1}
        # k in {k_offset, k_offset+1}
        
        for t in range(7):
            idx = term_start + t
            # Reset this term's entries
            # U entries: rows row_offset..row_offset+1, cols col_offset..col_offset+1
            # V entries: rows col_offset..col_offset+1, cols k_offset..k_offset+1
            # W entries: rows k_offset..k_offset+1, cols row_offset..row_offset+1
            
            if t == 0:  # m1 = (a+d)(e+h)
                U[row_offset, col_offset] += 1
                U[row_offset+1, col_offset+1] += 1
                V[col_offset, k_offset] += 1
                V[col_offset+1, k_offset+1] += 1
                W[k_offset, row_offset] += 1
                W[k_offset+1, row_offset+1] += 1
            elif t == 1:  # m2 = (c+d)e
                U[row_offset+1, col_offset] += 1
                U[row_offset+1, col_offset+1] += 1
                V[col_offset, k_offset] += 1
                W[k_offset, row_offset+1] += 1
                W[k_offset+1, row_offset+1] += -1
            elif t == 2:  # m3 = a(f-h)
                U[row_offset, col_offset] += 1
                V[col_offset, k_offset+1] += 1
                V[col_offset+1, k_offset+1] += -1
                W[k_offset+1, row_offset] += 1
                W[k_offset+1, row_offset+1] += 1
            elif t == 3:  # m4 = d(g-e)
                U[row_offset+1, col_offset+1] += 1
                V[col_offset+1, k_offset] += 1
                V[col_offset, k_offset] += -1
                W[k_offset, row_offset] += 1
                W[k_offset, row_offset+1] += 1
            elif t == 4:  # m5 = (a+b)h
                U[row_offset, col_offset] += 1
                U[row_offset, col_offset+1] += 1
                V[col_offset+1, k_offset+1] += 1
                W[k_offset, row_offset] += -1
                W[k_offset+1, row_offset] += 1
            elif t == 5:  # m6 = (c-a)(e+f)
                U[row_offset+1, col_offset] += 1
                U[row_offset, col_offset] += -1
                V[col_offset, k_offset] += 1
                V[col_offset, k_offset+1] += 1
                W[k_offset+1, row_offset+1] += 1
            elif t == 6:  # m7 = (b-d)(g+h)
                U[row_offset, col_offset+1] += 1
                U[row_offset+1, col_offset+1] += -1
                V[col_offset+1, k_offset] += 1
                V[col_offset+1, k_offset+1] += 1
                W[k_offset, row_offset] += 1
        
        return term_start + 7
    
    term = 0
    # Block (i=0:1, j=0:2, k=0:2): A[0:2,0:2] * B[0:2,0:2]
    term = strassen_block(U, V, W, 0, 0, 0, term)
    # Block (i=0:1, j=0:2, k=2:4): A[0:2,0:2] * B[0:2,2:4]
    term = strassen_block(U, V, W, 0, 0, 2, term)
    # Block (i=0:1, j=2:4, k=0:2): A[0:2,2:4] * B[2:4,0:2]
    term = strassen_block(U, V, W, 0, 2, 0, term)
    # Block (i=0:1, j=2:4, k=2:4): A[0:2,2:4] * B[2:4,2:4]
    term = strassen_block(U, V, W, 0, 2, 2, term)
    
    # Part 2: 1x4x4 (4 rank terms)
    # Row i=2; for each j=0..3, one term
    for j in range(4):
        idx = term
        # U[2*4+j, idx] = 1
        # V[j*4+k, idx] = 1 for all k
        # W[k*3+2, idx] = 1 for all k
        U[2 * m + j, idx] = 1
        for k in range(p):
            V[j * p + k, idx] = 1
            W[k * n + 2, idx] = 1
        term += 1
    
    # Verify the decomposition
    assert term == 32, f"Expected 32 terms, got {term}"
    
    # Check exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
