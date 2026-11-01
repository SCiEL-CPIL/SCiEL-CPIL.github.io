import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 7]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    R = 7
    U = [[0] * R for _ in range(4)]
    V = [[0] * R for _ in range(4)]
    W = [[0] * R for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    return U, V, W

def _schoolbook_2x1():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (column).
    U: (4, 4)
    V: (2, 4)
    W: (2, 4)
    Indices:
    U[i*2+j, c]
    V[j, c]
    W[i, c]
    """
    R = 4
    U = [[0] * R for _ in range(4)]
    V = [[0] * R for _ in range(2)]
    W = [[0] * R for _ in range(2)]
    
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 1: A00 * B0 -> C0
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 2: A01 * B1 -> C0
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # Term 3: A10 * B0 -> C1
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # Term 4: A11 * B1 -> C1
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Global shapes
    # U: (n*m, R) = (4, R)
    # V: (m*p, R) = (14, R)
    # W: (p*n, R) = (14, R)
    
    # Blocks:
    # 1. k=0,1 (Strassen)
    # 2. k=2,3 (Strassen)
    # 3. k=4,5 (Strassen)
    # 4. k=6   (Schoolbook)
    
    blocks = [
        {'type': 'strassen', 'k_offset': 0},
        {'type': 'strassen', 'k_offset': 2},
        {'type': 'strassen', 'k_offset': 4},
        {'type': 'schoolbook', 'k_offset': 6},
    ]
    
    total_rank = 0
    for b in blocks:
        if b['type'] == 'strassen':
            total_rank += 7
        else:
            total_rank += 4
            
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for b in blocks:
        k_off = b['k_offset']
        
        if b['type'] == 'strassen':
            U_loc, V_loc, W_loc = _strassen_2x2()
            r_loc = 7
            
            # U is same for all blocks, just copy columns
            for i in range(n * m):
                for c in range(r_loc):
                    U_global[i][col_offset + c] = U_loc[i][c]
            
            # V: local index j*2 + k_loc -> global index j*p + (k_off + k_loc)
            for j in range(m):
                for k_loc in range(2):
                    v_idx_loc = j * 2 + k_loc
                    v_idx_glob = j * p + (k_off + k_loc)
                    for c in range(r_loc):
                        V_global[v_idx_glob][col_offset + c] = V_loc[v_idx_loc][c]
            
            # W: local index k_loc*2 + i -> global index (k_off + k_loc)*n + i
            for k_loc in range(2):
                for i in range(n):
                    w_idx_loc = k_loc * 2 + i
                    w_idx_glob = (k_off + k_loc) * n + i
                    for c in range(r_loc):
                        W_global[w_idx_glob][col_offset + c] = W_loc[w_idx_loc][c]
                        
            col_offset += r_loc
            
        else: # schoolbook
            U_loc, V_loc, W_loc = _schoolbook_2x1()
            r_loc = 4
            
            # U
            for i in range(n * m):
                for c in range(r_loc):
                    U_global[i][col_offset + c] = U_loc[i][c]
            
            # V: local index j -> global index j*p + k_off
            for j in range(m):
                v_idx_loc = j
                v_idx_glob = j * p + k_off
                for c in range(r_loc):
                    V_global[v_idx_glob][col_offset + c] = V_loc[v_idx_loc][c]
            
            # W: local index i -> global index k_off*n + i
            for i in range(n):
                w_idx_loc = i
                w_idx_glob = k_off * n + i
                for c in range(r_loc):
                    W_global[w_idx_glob][col_offset + c] = W_loc[w_idx_loc][c]
                        
            col_offset += r_loc

    return {"U": U_global, "V": V_global, "W": W_global}
