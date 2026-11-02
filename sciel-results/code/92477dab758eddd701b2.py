import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target rank 88
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: 2x4x8 (Top 2 rows of A)
    # A_top is 2x4, B is 4x8
    # Split A_top into A_L (2x2), A_R (2x2)
    # Split B into B_L (2x8), B_R (2x8)
    # C_top = A_L B_L + A_R B_R
    # Each is 2x2x8, which is 4 blocks of 2x2x2
    
    # Helper to add a Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # i_offset: offset in A's row index (0..1 for 2x2)
        # j_offset: offset in A's col index / B's row index (0..1 for 2x2)
        # k_offset: offset in B's col index (0..1 for 2x2)
        
        # Strassen factors for 2x2x2
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[i, c], V[j, c], W[k, c]
        
        # M1 = (A11+A22)(B11+B22)
        U[i_offset*2+0, col_start+0] = 1
        U[i_offset*2+1, col_start+0] = 1
        V[j_offset*2+0, col_start+0] = 1
        V[j_offset*2+1, col_start+0] = 1
        W[k_offset*2+0, col_start+0] = 1
        W[k_offset*2+1, col_start+0] = 1
        
        # M2 = (A21+A22)B11
        U[i_offset*2+1, col_start+1] = 1
        V[j_offset*2+0, col_start+1] = 1
        W[k_offset*2+0, col_start+1] = 1
        
        # M3 = A11(B21-B22)
        U[i_offset*2+0, col_start+2] = 1
        V[j_offset*2+1, col_start+2] = 1
        V[j_offset*2+0, col_start+2] = -1 # Wait, B21 is j=1, B22 is j=0? 
        # Strassen: B21 is bottom-left? 
        # Standard: 
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M3 = A11(B21-B22)
        # B21 is row 1, col 0. B22 is row 1, col 1.
        # So V[j=1, k=0] = 1, V[j=1, k=1] = -1.
        # My V index is j*p+k. Here p=2.
        # V[j_offset*2+1, col_start+2] = 1 (B21)
        # V[j_offset*2+1, col_start+2] is the same entry? No.
        # V is (m*p, rank). 
        # For a 2x2x2 block, m=2, p=2.
        # V index is j*2+k.
        # B21 is j=1, k=0. Index 2.
        # B22 is j=1, k=1. Index 3.
        # So V[2, c] = 1, V[3, c] = -1.
        
        # Let's restart the Strassen factor mapping carefully.
        pass

    # I will implement the full logic in the final code block.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
