"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    current_col = 0
    
    def add_term(u_idx, v_idx, w_idx, sign):
        nonlocal current_col
        U[u_idx, current_col] += sign
        V[v_idx, current_col] += sign
        W[w_idx, current_col] += sign
        current_col += 1

    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        """
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base, current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[j_base*p+k_base, current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        current_col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base, current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[j_base*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = -1
        current_col += 1
        
        # M2: A00(B01-B11)
        U[i_base*m+j_base, current_col] = 1
        V[j_base*p+(k_base+1), current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = -1
        W[(k_base+1)*n+i_base, current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        current_col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[(j_base+1)*p+k_base, current_col] = 1
        V[j_base*p+k_base, current_col] = -1
        W[k_base*n+i_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        current_col += 1
        
        # M4: (A00+A01)B11
        U[i_base*m+j_base, current_col] = 1
        U[i_base*m+(j_base+1), current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = -1
        W[(k_base+1)*n+i_base, current_col] = 1
        current_col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base, current_col] = 1
        U[i_base*m+j_base, current_col] = -1
        V[j_base*p+k_base, current_col] = 1
        V[j_base*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        current_col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1), current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = -1
        V[(j_base+1)*p+k_base, current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        current_col += 1

    def add_outer_product_block(i_base, j_single, k_base, k_width):
        """
        Adds rank (n * k_width) decomposition for a n x 1 * 1 x k_width block.
        """
        nonlocal current_col
        for i in range(n):
            for k in range(k_base, k_base + k_width):
                U[i * m + j_single, current_col] = 1
                V[j_single * p + k, current_col] = 1
                W[k * n + i, current_col] = 1
                current_col += 1

    # Partition j (middle dimension) into 0,1 (Strassen), 2,3 (Strassen), 4 (Outer)
    # Partition k (output cols) into 0,1 / 2,3 / 4,5
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Block 3: A[:, 4] * B[4, :]
    add_outer_product_block(0, 4, 0, 6)

    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
