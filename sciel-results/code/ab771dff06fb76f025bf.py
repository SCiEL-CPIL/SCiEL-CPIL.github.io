"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _rank11_solution():
    """Return the known rank-11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Term 0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # Term 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # Term 2
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # Term 3
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # Term 4
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # Term 5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # Term 6
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W entries for Strassen part
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Terms 7-10 for the 3rd column (k=2)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0] B[0,2]
    U[0][7] = 1
    V[0*3+2][7] = 1  # V[2, 7]
    W[2*2+0][7] = 1  # W[4, 7]
    
    # Term 8: A[0,1] B[1,2]
    U[1][8] = 1
    V[1*3+2][8] = 1  # V[5, 8]
    W[2*2+0][8] = 1  # W[4, 8]
    
    # Term 9: A[1,0] B[0,2]
    U[2][9] = 1
    V[0*3+2][9] = 1  # V[2, 9]
    W[2*2+1][9] = 1  # W[5, 9]
    
    # Term 10: A[1,1] B[1,2]
    U[3][10] = 1
    V[1*3+2][10] = 1 # V[5, 10]
    W[2*2+1][10] = 1 # W[5, 10]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Return the optimal rank-11 solution immediately
    return _rank11_solution()
# EVOLVE-BLOCK-END
