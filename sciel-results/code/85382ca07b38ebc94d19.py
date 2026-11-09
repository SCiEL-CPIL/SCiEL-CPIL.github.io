"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = 120.0  # 120 seconds limit, keep safety margin
    
    n, m, p = DIMENSIONS
    # Target rank: 54
    # 6 blocks of 2x2x2 (Strassen, rank 7 each) -> 42
    # 3 blocks of 2x1x2 (Outer, rank 4 each) -> 12
    # Total 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_start, j_start, k_start, col):
        # Local indices 0,1
        # A: i in {0,1}, j in {0,1} -> global (i_start+i)*m + (j_start+j)
        # B: j in {0,1}, k in {0,1} -> global (j_start+j)*p + (k_start+k)
        # C: i in {0,1}, k in {0,1} -> global (k_start+k)*n + (i_start+i)
        
        # M1: (A00+A11)(B00+B11)
        U[i_start*m + j_start][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[j_start*p + k_start][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[k_start*n + i_start][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[(i_start+1)*m + j_start][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[j_start*p + k_start][col] += 1
        W[k_start*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[i_start*m + j_start][col] += 1
        V[j_start*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += -1
        W[(k_start+1)*n + i_start][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + k_start][col] += 1
        V[j_start*p + k_start][col] += -1
        W[k_start*n + i_start][col] += 1
        W[k_start*n + (i_start+1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[i_start*m + j_start][col] += 1
        U[i_start*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[k_start*n + i_start][col] += -1
        W[(k_start+1)*n + i_start][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_start+1)*m + j_start][col] += 1
        U[i_start*m + j_start][col] += -1
        V[j_start*p + k_start][col] += 1
        V[j_start*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i_start*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += -1
        V[(j_start+1)*p + k_start][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[k_start*n + i_start][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += -1
        col += 1
        
        return col

    # Helper to add an Outer product block (2x1x2)
    def add_outer(i_start, j_start, k_start, col):
        # A: i in {0,1}, j in {0} -> global (i_start+i)*m + j_start
        # B: j in {0}, k in {0,1} -> global j_start*p + (k_start+k)
        # C: i in {0,1}, k in {0,1} -> global (k_start+k)*n + (i_start+i)
        
        # Term 1: A0 * B0 -> C00
        U[i_start*m + j_start][col] += 1
        V[j_start*p + k_start][col] += 1
        W[k_start*n + i_start][col] += 1
        col += 1
        
        # Term 2: A0 * B1 -> C01
        U[i_start*m + j_start][col] += 1
        V[j_start*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + i_start][col] += 1
        col += 1
        
        # Term 3: A1 * B0 -> C10
        U[(i_start+1)*m + j_start][col] += 1
        V[j_start*p + k_start][col] += 1
        W[k_start*n + (i_start+1)][col] += 1
        col += 1
        
        # Term 4: A1 * B1 -> C11
        U[(i_start+1)*m + j_start][col] += 1
        V[j_start*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        col += 1
        
        return col

    # Block 1: i=0,1; j=0,1; k=0,1
    col = add_strassen(0, 0, 0, col)
    # Block 2: i=0,1; j=0,1; k=2,3
    col = add_strassen(0, 0, 2, col)
    # Block 3: i=0,1; j=0,1; k=4,5
    col = add_strassen(0, 0, 4, col)
    
    # Block 4: i=0,1; j=2,3; k=0,1
    col = add_strassen(0, 2, 0, col)
    # Block 5: i=0,1; j=2,3; k=2,3
    col = add_strassen(0, 2, 2, col)
    # Block 6: i=0,1; j=2,3; k=4,5
    col = add_strassen(0, 2, 4, col)
    
    # Block 7: i=0,1; j=4; k=0,1
    col = add_outer(0, 4, 0, col)
    # Block 8: i=0,1; j=4; k=2,3
    col = add_outer(0, 4, 2, col)
    # Block 9: i=0,1; j=4; k=4,5
    col = add_outer(0, 4, 4, col)
    
    # Check time
    if time.time() > deadline - 10:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
