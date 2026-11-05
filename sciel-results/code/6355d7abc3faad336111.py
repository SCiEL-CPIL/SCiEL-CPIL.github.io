import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS
    
    def _strassen_2x2(U, V, W, base_r, base_c, base_k, col):
        """Add Strassen rank-7 terms for a 2x2x2 block at positions (base_r, base_c, base_k)."""
        # M1 = (A11+A22)(B11+B22)
        U[base_r*5+base_c, col] = 1
        U[(base_r+1)*5+(base_c+1), col] = 1
        V[base_c*5+base_k, col] = 1
        V[(base_c+1)*5+(base_k+1), col] = 1
        W[base_k*5+base_r, col] = 1
        W[(base_k+1)*5+(base_r+1), col] = 1
        W[base_k*5+(base_r+1), col] = -1
        W[(base_k+1)*5+base_r, col] = -1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(base_r+1)*5+base_c, col] = 1
        U[(base_r+1)*5+(base_c+1), col] = 1
        V[base_c*5+base_k, col] = 1
        W[(base_k+1)*5+base_r, col] = 1
        col += 1
        
        # M3 = A11(B21-B11)
        U[base_r*5+base_c, col] = 1
        V[(base_c+1)*5+base_k, col] = 1
        V[base_c*5+base_k, col] = -1
        W[base_k*5+(base_r+1), col] = 1
        col += 1
        
        # M4 = A22(B22-B12)
        U[(base_r+1)*5+(base_c+1), col] = 1
        V[(base_c+1)*5+(base_k+1), col] = 1
        V[(base_c+1)*5+base_k, col] = -1
        W[(base_k+1)*5+base_r, col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[base_r*5+base_c, col] = 1
        U[base_r*5+(base_c+1), col] = 1
        V[(base_c+1)*5+(base_k+1), col] = 1
        W[base_k*5+base_r, col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(base_r+1)*5+base_c, col] = 1
        U[base_r*5+base_c, col] = -1
        V[base_c*5+base_k, col] = 1
        V[base_c*5+(base_k+1), col] = 1
        W[base_k*5+(base_r+1), col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[base_r*5+(base_c+1), col] = 1
        U[(base_r+1)*5+(base_c+1), col] = -1
        V[(base_c+1)*5+base_k, col] = 1
        V[(base_c+1)*5+(base_k+1), col] = 1
        W[(base_k+1)*5+base_r, col] = 1
        col += 1
        
        return col
    
    # Rank 123: schoolbook with 2 Strassen blocks on diagonal 2x2 sub-blocks
    rank = 123
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    col = 0
    
    # Strassen for block at (0,0,0)
    col = _strassen_2x2(U, V, W, 0, 0, 0, col)
    
    # Schoolbook for remaining entries in block (0,0,0) region that aren't covered
    # Actually, Strassen covers all 8 entries of the 2x2x2 block
    # Now do schoolbook for all other entries
    
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Skip entries covered by Strassen blocks
                if (i in [0,1] and j in [0,1] and k in [0,1]):
                    continue
                if (i in [2,3] and j in [2,3] and k in [2,3]):
                    continue
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*5+i, col] = 1
                col += 1
    
    # Strassen for block at (2,2,2) - but we already did schoolbook for those entries
    # Need to redo: let me restructure
    
    # Actually, let me redo this properly
    col = 0
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    # Strassen for block (0,0,0)
    col = _strassen_2x2(U, V, W, 0, 0, 0, col)
    
    # Strassen for block (2,2,2)
    col = _strassen_2x2(U, V, W, 2, 2, 2, col)
    
    # Schoolbook for all other entries
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if (i in [0,1] and j in [0,1] and k in [0,1]):
                    continue
                if (i in [2,3] and j in [2,3] and k in [2,3]):
                    continue
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*5+i, col] = 1
                col += 1
    
    # col should be 14 + 109 = 123
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
