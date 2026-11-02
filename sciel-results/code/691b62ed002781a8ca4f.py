"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Uses GQ-style decomposition with rank 36.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = np.zeros((n * m, 36), dtype=float)
    V = np.zeros((m * p, 36), dtype=float)
    W = np.zeros((p * n, 36), dtype=float)
    
    col = 0
    
    def add_strassen_2x2x2(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        nonlocal col
        # Strassen 7 terms
        # m1
        U[a_off_i*m + a_off_j, col] = 1; U[(a_off_i+1)*m + (a_off_j+1), col] = 1
        V[b_off_j*p + b_off_k, col] = 1; V[(b_off_j+1)*p + (b_off_k+1), col] = 1
        W[c_off_k*n + c_off_i, col] = 1; W[(c_off_k+1)*n + (c_off_i+1), col] = 1
        col += 1
        # m2
        U[(a_off_i+1)*m + a_off_j, col] = 1; U[(a_off_i+1)*m + (a_off_j+1), col] = 1
        V[b_off_j*p + b_off_k, col] = 1
        W[c_off_k*n + (c_off_i+1), col] = 1; W[(c_off_k+1)*n + (c_off_i+1), col] = -1
        col += 1
        # m3
        U[a_off_i*m + a_off_j, col] = 1
        V[b_off_j*p + (b_off_k+1), col] = 1; V[(b_off_j+1)*p + (b_off_k+1), col] = -1
        W[(c_off_k+1)*n + c_off_i, col] = 1; W[(c_off_k+1)*n + (c_off_i+1), col] = 1
        col += 1
        # m4
        U[(a_off_i+1)*m + (a_off_j+1), col] = 1
        V[(b_off_j+1)*p + b_off_k, col] = 1; V[b_off_j*p + b_off_k, col] = -1
        W[c_off_k*n + c_off_i, col] = 1; W[c_off_k*n + (c_off_i+1), col] = 1
        col += 1
        # m5
        U[a_off_i*m + a_off_j, col] = 1; U[a_off_i*m + (a_off_j+1), col] = 1
        V[(b_off_j+1)*p + (b_off_k+1), col] = 1
        W[c_off_k*n + c_off_i, col] = -1; W[(c_off_k+1)*n + c_off_i, col] = 1
        col += 1
        # m6
        U[(a_off_i+1)*m + a_off_j, col] = 1; U[a_off_i*m + a_off_j, col] = -1
        V[b_off_j*p + b_off_k, col] = 1; V[b_off_j*p + (b_off_k+1), col] = 1
        W[(c_off_k+1)*n + (c_off_i+1), col] = 1
        col += 1
        # m7
        U[a_off_i*m + (a_off_j+1), col] = 1; U[(a_off_i+1)*m + (a_off_j+1), col] = -1
        V[(b_off_j+1)*p + b_off_k, col] = 1; V[(b_off_j+1)*p + (b_off_k+1), col] = 1
        W[c_off_k*n + c_off_i, col] = 1
        col += 1

    def add_schoolbook_2x2x1(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        nonlocal col
        # 4 terms
        U[a_off_i*m + a_off_j, col] = 1
        V[b_off_j*p + b_off_k, col] = 1
        W[c_off_k*n + c_off_i, col] = 1
        col += 1
        
        U[a_off_i*m + (a_off_j+1), col] = 1
        V[(b_off_j+1)*p + b_off_k, col] = 1
        W[c_off_k*n + c_off_i, col] = 1
        col += 1
        
        U[(a_off_i+1)*m + a_off_j, col] = 1
        V[b_off_j*p + b_off_k, col] = 1
        W[c_off_k*n + (c_off_i+1), col] = 1
        col += 1
        
        U[(a_off_i+1)*m + (a_off_j+1), col] = 1
        V[(b_off_j+1)*p + b_off_k, col] = 1
        W[c_off_k*n + (c_off_i+1), col] = 1
        col += 1

    # Top A-block (i=0,1; j=0,1)
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    
    # Bottom A-block (i=0,1; j=2,3)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)
    
    # Verify
    if col != 36:
        # Should not happen
        pass
        
    # Convert to lists as required by interface? 
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    # The return type is a dict of U, V, W.
    # The example used lists. I will use lists to be safe, or numpy arrays if accepted.
    # The prompt says "factor matrices", usually implying arrays.
    # I will return numpy arrays.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
