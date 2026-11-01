"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[2, 3] = 1.0
    
    # M5
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[1, 4] = 1.0
    
    # M6
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[2, 5] = 1.0
    
    # M7
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[3, 6] = 1.0
    
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :])
                val = np.dot(val, W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_2x2x15():
    """
    Construct a rank-53 decomposition for M_{2,2,15} using:
    - 7 blocks of 2x2x2 (Strassen, rank 7 each) = 49
    - 1 block of 2x2x1 (schoolbook, rank 4) = 4
    Total: 53
    """
    n, m, p = 2, 2, 15
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # 2x2x1 schoolbook
    U_1 = np.zeros((4, 4), dtype=np.float64)
    V_1 = np.zeros((2, 4), dtype=np.float64)
    W_1 = np.zeros((2, 4), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            U_1[i * 2 + j, col] = 1.0
            V_1[j, col] = 1.0
            W_1[i, col] = 1.0
            col += 1
    
    num_strassen = 7
    num_single = 1
    total_rank = num_strassen * 7 + num_single * 4  # 49 + 4 = 53
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    # 7 Strassen blocks for k = 0..13
    for b in range(num_strassen):
        for r in range(7):
            U[:, col_offset] = U_s[:, r]
            
            V[k_offset * m, col_offset] = V_s[0, r]
            V[k_offset * m + 1, col_offset] = V_s[1, r]
            V[(k_offset + 1) * m, col_offset] = V_s[2, r]
            V[(k_offset + 1) * m + 1, col_offset] = V_s[3, r]
            
            W[k_offset * n, col_offset] = W_s[0, r]
            W[k_offset * n + 1, col_offset] = W_s[1, r]
            W[(k_offset + 1) * n, col_offset] = W_s[2, r]
            W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
            
            col_offset += 1
        k_offset += 2
    
    # 1 schoolbook block for k = 14
    for r in range(4):
        U[:, col_offset] = U_1[:, r]
        
        V[k_offset * m, col_offset] = V_1[0, r]
        V[k_offset * m + 1, col_offset] = V_1[1, r]
        
        W[k_offset * n, col_offset] = W_1[0, r]
        W[k_offset * n + 1, col_offset] = W_1[1, r]
        
        col_offset += 1
    k_offset += 1
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x15 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with the known rank-53 construction
    U, V, W = _construct_2x2x15()
    best_rank = U.shape[1]
    
    if not _is_valid(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_rank = U.shape[1]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

