import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 6 blocks
    # (j_offset, k_offset, is_strassen)
    blocks = [
        (0, 0, True),
        (0, 2, True),
        (0, 4, False),
        (2, 0, True),
        (2, 2, True),
        (2, 4, False)
    ]
    
    col = 0
    for j_off, k_off, is_strassen in blocks:
        if is_strassen:
            # 7 Strassen terms
            # Term 1
            U[0*m + j_off+0][col] = 1
            U[1*m + j_off+1][col] = 1
            V[(j_off+0)*p + k_off+0][col] = 1
            V[(j_off+1)*p + k_off+1][col] = 1
            W[(k_off+0)*n + 0][col] = 1
            W[(k_off+1)*n + 1][col] = 1
            col += 1
            
            # Term 2
            U[1*m + j_off+0][col] = 1
            U[1*m + j_off+1][col] = 1
            V[(j_off+0)*p + k_off+0][col] = 1
            W[(k_off+0)*n + 1][col] = 1
            W[(k_off+1)*n + 1][col] = -1
            col += 1
            
            # Term 3
            U[0*m + j_off+0][col] = 1
            V[(j_off+0)*p + k_off+1][col] = 1
            V[(j_off+1)*p + k_off+1][col] = -1
            W[(k_off+1)*n + 0][col] = 1
            W[(k_off+1)*n + 1][col] = 1
            col += 1
            
            # Term 4
            U[1*m + j_off+1][col] = 1
            V[(j_off+0)*p + k_off+0][col] = -1
            V[(j_off+1)*p + k_off+0][col] = 1
            W[(k_off+0)*n + 0][col] = 1
            W[(k_off+0)*n + 1][col] = 1
            col += 1
            
            # Term 5
            U[0*m + j_off+0][col] = 1
            U[0*m + j_off+1][col] = 1
            V[(j_off+1)*p + k_off+1][col] = 1
            W[(k_off+0)*n + 0][col] = -1
            W[(k_off+1)*n + 0][col] = 1
            col += 1
            
            # Term 6
            U[0*m + j_off+0][col] = -1
            U[1*m + j_off+0][col] = 1
            V[(j_off+0)*p + k_off+0][col] = 1
            V[(j_off+0)*p + k_off+1][col] = 1
            W[(k_off+1)*n + 1][col] = 1
            col += 1
            
            # Term 7
            U[0*m + j_off+1][col] = 1
            U[1*m + j_off+1][col] = -1
            V[(j_off+1)*p + k_off+0][col] = 1
            V[(j_off+1)*p + k_off+1][col] = 1
            W[(k_off+0)*n + 0][col] = 1
            col += 1
            
        else:
            # 4 Schoolbook terms
            # Term 1
            U[0*m + j_off+0][col] = 1
            V[(j_off+0)*p + k_off][col] = 1
            W[k_off*n + 0][col] = 1
            col += 1
            
            # Term 2
            U[0*m + j_off+1][col] = 1
            V[(j_off+1)*p + k_off][col] = 1
            W[k_off*n + 0][col] = 1
            col += 1
            
            # Term 3
            U[1*m + j_off+0][col] = 1
            V[(j_off+0)*p + k_off][col] = 1
            W[k_off*n + 1][col] = 1
            col += 1
            
            # Term 4
            U[1*m + j_off+1][col] = 1
            V[(j_off+1)*p + k_off][col] = 1
            W[k_off*n + 1][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
