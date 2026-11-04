"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses a block decomposition with Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    
    # Dimensions for U, V, W
    U_rows = n * m  # 12
    V_rows = m * p  # 32
    W_rows = p * n  # 24
    
    # We will build the decomposition column by column.
    # We'll store the coefficients for each column in lists, then transpose to matrix form.
    # Or just build the matrices directly.
    
    # Initialize empty lists for columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i0, j0, k0):
        """
        Add a Strassen 2x2x2 decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        i0, j0, k0 are the top-left indices of the block.
        """
        # Local indices within the 2x2 block
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # Strassen 7 products
        # M1 = (a11+a22)(b11+b22)
        u1 = [0]*U_rows; v1 = [0]*V_rows; w1 = [0]*W_rows
        u1[i0*m + j0] = 1
        u1[(i0+1)*m + (j0+1)] = 1
        v1[j0*p + k0] = 1
        v1[(j0+1)*p + (k0+1)] = 1
        w1[k0*n + i0] = 1       # c11
        w1[(k0+1)*n + (i0+1)] = 1 # c22
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # M2 = (a21+a22)b11
        u2 = [0]*U_rows; v2 = [0]*V_rows; w2 = [0]*W_rows
        u2[(i0+1)*m + j0] = 1
        u2[(i0+1)*m + (j0+1)] = 1
        v2[j0*p + k0] = 1
        w2[k0*n + (i0+1)] = 1   # c21
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # M3 = a11(b12-b22)
        u3 = [0]*U_rows; v3 = [0]*V_rows; w3 = [0]*W_rows
        u3[i0*m + j0] = 1
        v3[j0*p + (k0+1)] = 1
        v3[(j0+1)*p + (k0+1)] = -1
        w3[(k0+1)*n + i0] = 1   # c12
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # M4 = a22(b21-b11)
        u4 = [0]*U_rows; v4 = [0]*V_rows; w4 = [0]*W_rows
        u4[(i0+1)*m + (j0+1)] = 1
        v4[(j0+1)*p + k0] = 1
        v4[j0*p + k0] = -1
        w4[k0*n + i0] = 1       # c11
        w4[k0*n + (i0+1)] = 1   # c21
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
        # M5 = (a11+a12)b22
        u5 = [0]*U_rows; v5 = [0]*V_rows; w5 = [0]*W_rows
        u5[i0*m + j0] = 1
        u5[i0*m + (j0+1)] = 1
        v5[(j0+1)*p + (k0+1)] = 1
        w5[(k0+1)*n + i0] = 1   # c12
        w5[(k0+1)*n + (i0+1)] = -1 # c22 (Note: c22 = M1 - M2 + M3 + M6, wait. Let's re-verify Strassen formulas)
        
        # Re-verifying Strassen:
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # My M5 definition: (a11+a12)b22.
        # In c12: coeff of M5 is 1.
        # In c22: coeff of M5 is 0.
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M5:
        # w5 for C12 (i0, k0+1) is 1.
        # w5 for C22 (i0+1, k0+1) is 0.
        # w5 for C11 (i0, k0) is -1.
        # w5 for C21 (i0+1, k0) is 0.
        
        w5[(k0+1)*n + i0] = 1
        w5[k0*n + i0] = -1
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)
        
        # M6 = (a21-a11)(b11+b12)
        u6 = [0]*U_rows; v6 = [0]*V_rows; w6 = [0]*W_rows
        u6[(i0+1)*m + j0] = 1
        u6[i0*m + j0] = -1
        v6[j0*p + k0] = 1
        v6[j0*p + (k0+1)] = 1
        w6[(k0+1)*n + (i0+1)] = 1 # C22
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)
        
        # M7 = (a12-a22)(b21+b22)
        u7 = [0]*U_rows; v7 = [0]*V_rows; w7 = [0]*W_rows
        u7[i0*m + (j0+1)] = 1
        u7[(i0+1)*m + (j0+1)] = -1
        v7[(j0+1)*p + k0] = 1
        v7[(j0+1)*p + (k0+1)] = 1
        w7[k0*n + i0] = 1 # C11
        U_cols.append(u7); V_cols.append(v7); W_cols.append(w7)
        
    def add_schoolbook_block_1x2x2(i0, j0, k0):
        """
        Add schoolbook 1x2x2 decomposition for A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]
        """
        # A is 1x2: a1 (i0, j0), a2 (i0, j0+1)
        # B is 2x2: b11 (j0, k0), b12 (j0, k0+1), b21 (j0+1, k0), b22 (j0+1, k0+1)
        # C is 1x2: c1 (i0, k0), c2 (i0, k0+1)
        
        # c1 = a1*b11 + a2*b21
        # c2 = a1*b12 + a2*b22
        
        # M1 = a1*b11
        u1 = [0]*U_rows; v1 = [0]*V_rows; w1 = [0]*W_rows
        u1[i0*m + j0] = 1
        v1[j0*p + k0] = 1
        w1[k0*n + i0] = 1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # M2 = a1*b12
        u2 = [0]*U_rows; v2 = [0]*V_rows; w2 = [0]*W_rows
        u2[i0*m + j0] = 1
        v2[j0*p + (k0+1)] = 1
        w2[(k0+1)*n + i0] = 1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # M3 = a2*b21
        u3 = [0]*U_rows; v3 = [0]*V_rows; w3 = [0]*W_rows
        u3[i0*m + (j0+1)] = 1
        v3[(j0+1)*p + k0] = 1
        w3[k0*n + i0] = 1
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # M4 = a2*b22
        u4 = [0]*U_rows; v4 = [0]*V_rows; w4 = [0]*W_rows
        u4[i0*m + (j0+1)] = 1
        v4[(j0+1)*p + (k0+1)] = 1
        w4[(k0+1)*n + i0] = 1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
    # Define the blocks
    # i splits: 0, 1 (2 rows), 2 (1 row)
    # j splits: 0, 1 (2 cols), 2, 3 (2 cols)
    # k splits: 0, 1 (2 cols), 2, 3 (2 cols), 4, 5 (2 cols), 6, 7 (2 cols)
    
    i_splits = [(0, 2), (2, 3)] # (start, end) -> size 2, size 1
    j_splits = [(0, 2), (2, 4)]
    k_splits = [(0, 2), (2, 4), (4, 6), (6, 8)]
    
    for i_start, i_end in i_splits:
        i_size = i_end - i_start
        for j_start, j_end in j_splits:
            j_size = j_end - j_start
            for k_start, k_end in k_splits:
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    add_strassen_block(i_start, j_start, k_start)
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    add_schoolbook_block_1x2x2(i_start, j_start, k_start)
                else:
                    # Fallback to schoolbook if unexpected size
                    # Should not happen with this specific split
                    pass
                    
    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T  # Transpose to get shape (U_rows, rank)
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
