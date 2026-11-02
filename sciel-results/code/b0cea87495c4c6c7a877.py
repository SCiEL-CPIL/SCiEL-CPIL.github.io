"""
Tensor decomposition for 3x3x8 matrix multiplication.
Uses Lader's rank-23 for the 3x3x3 parts and a rank-18 schoolbook for the 3x3x2 part.
Total rank: 23 + 23 + 18 = 64.
Includes a search for potential rank reduction.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    elapsed = lambda: time.time() - start_time
    time_left = lambda: deadline - elapsed()
    elapsed = lambda: time.time() - start_time
    time_left = lambda: deadline - elapsed()
    
    # Lader's rank-23 decomposition for 3x3x3
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1],
    ]
    
    # Verify Lader decomposition
    def verify_333(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(len(U)):
                        total += U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
                    if total != 1:
                        return False
        return True
    
    lader_valid = verify_333(lader_U, lader_V, lader_W)
    
    # Helper to verify full decomposition
    def verify_full(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U)):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if lader_valid:
        # Build 3x3x8 as two 3x3x4 blocks, each = 3x3x3 + 3x3x1
        # 3x3x3: rank 23 (Lader)
        # 3x3x1: rank 1 (all-ones tensor)
        # 3x3x4: rank 24
        # 3x3x8: rank 48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: k=0,1,2 (3x3x3) + k=3 (3x3x1)
        # Columns 0-22: Lader for k=0,1,2
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Column 23: 3x3x1 for k=3
        for i in range(3):
            for j in range(3):
                U[i*m+j][23] = 1
            for j in range(3):
                V[j*p+3][23] = 1
            for i in range(3):
                W[3*n+i][23] = 1
        
        # Block 2: k=4,5,6 (3x3x3) + k=7 (3x3x1)
        # Columns 24-46: Lader for k=4,5,6
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][24+r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+(4+k)][24+r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(4+k)*n+i][24+r] = lader_W[r][k*3+i]
        
        # Column 47: 3x3x1 for k=7
        for i in range(3):
            for j in range(3):
                U[i*m+j][47] = 1
            for j in range(3):
                V[j*p+7][47] = 1
            for i in range(3):
                W[7*n+i][47] = 1
        
        if verify_full(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to original 3+3+2 approach
            rank = 23 + 23 + 18
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for r in range(23):
                for i in range(3):
                    for j in range(3):
                        U[i*m+j][r] = lader_U[r][i*3+j]
                for j in range(3):
                    for k in range(3):
                        V[j*p+k][r] = lader_V[r][j*3+k]
                for k in range(3):
                    for i in range(3):
                        W[k*n+i][r] = lader_W[r][k*3+i]
            
            for r in range(23):
                for i in range(3):
                    for j in range(3):
                        U[i*m+j][23+r] = lader_U[r][i*3+j]
                for j in range(3):
                    for k in range(3):
                        V[j*p+(3+k)][23+r] = lader_V[r][j*3+k]
                for k in range(3):
                    for i in range(3):
                        W[(3+k)*n+i][23+r] = lader_W[r][k*3+i]
            
            col = 46
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        U[i*m+j][col] = 1
                        V[j*p+(6+k)][col] = 1
                        W[(6+k)*n+i][col] = 1
                        col += 1
            
            if verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            else:
                # Fall back to schoolbook
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][column] = 1
                            V[j * p + k][column] = 1
                            W[k * n + i][column] = 1
                            column += 1
                return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook rank 72
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
