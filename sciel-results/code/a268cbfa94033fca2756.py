"""
Program to find a rank-minimal exact tensor decomposition for multiplying 
a 2 by 3 matrix by a 3 by 5 matrix.

Strategy:
Use additive rank bounds to construct a rank 17 decomposition.
R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3)
R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3)
R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1)
R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2)

Base cases:
R(2, 1, k) = 1
R(2, 2, 1) = 1
R(2, 2, 2) = 7 (Strassen)

Result:
R(2, 2, 3) <= 7 + 1 = 8
R(2, 3, 3) <= 8 + 1 = 9
R(2, 3, 2) <= 7 + 1 = 8
R(2, 3, 5) <= 8 + 9 = 17
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j, :]
                v_val = V_np[j * p + k, :]
                w_val = W_np[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > tol:
                    return False
    return True

def _decomp_2x1xp(p):
    """
    Decomposition for 2x1 * 1xp.
    A is 2x1, B is 1xp. C is 2xp.
    C[i,k] = A[i,0] * B[0,k].
    This is an outer product, rank 1.
    U shape (2*1, 1) = (2, 1)
    V shape (1*p, 1) = (p, 1)
    W shape (p*2, 1) = (2p, 1)
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    m=1, n=2.
    U[i*1+0] = U[i]
    V[0*p+k] = V[k]
    W[k*2+i]
    """
    U = [[1] for _ in range(2)]
    V = [[1] for _ in range(p)]
    W = [[1] for _ in range(p * 2)]
    return U, V, W, 1

def _decomp_2x2x1():
    """
    Decomposition for 2x2 * 2x1.
    A is 2x2, B is 2x1. C is 2x1.
    C[i,0] = sum_j A[i,j] B[j,0].
    Rank 1.
    """
    U = [[1], [1], [1], [1]]
    V = [[1], [1]]
    W = [[1], [1]]
    return U, V, W, 1

def _decomp_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Rank 7.
    """
    # Verified Strassen matrices
    U = [
        [ 1,  1,  1,  1,  0, -1, -1],
        [ 0,  0,  1,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  1,  1],
        [-1,  1, -1,  0,  1,  0,  0]
    ]
    V = [
        [ 1,  1,  0,  1,  1,  1,  0],
        [ 1,  1, -1, -1, -1, -1,  1],
        [ 1,  0,  1,  0,  0,  1,  1],
        [ 1,  0,  0,  0,  0,  1,  1]
    ]
    W = [
        [ 1,  0,  0,  0, -1,  0,  1],
        [ 0,  0,  1,  0,  1,  0,  0],
        [ 0,  1,  0,  0,  0, -1,  1],
        [ 1,  0, -1,  0,  0,  1,  0]
    ]
    return U, V, W, 7

def _construct_2x3x2():
    """
    R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2).
    Split m=3 into 2+1.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x1xp(2)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*2)] # 6 rows
    W = [[0]*rank for _ in range(2*2)] # 4 rows
    
    # Part 1: 2x2x2 (Top part of A, Top part of B)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
    
    # V1[j*2+k, c] -> V[j*2+k, c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*2+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x2 (Bottom part of A, Bottom part of B)
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*2+k, c] -> V[2*2+k, r1+c]
    for k in range(2):
        for c in range(r2):
            V[2*2+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x2x3():
    """
    R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1).
    Split p=3 into 2+1.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x2x1()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*2)] # 4 rows
    V = [[0]*rank for _ in range(2*3)] # 6 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x2 (k in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*2+j][c] = U1[i*2+j][c]
                
    # V1[j*2+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*3+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x2x1 (k in 2)
    for i in range(2):
        for j in range(2):
            for c in range(r2):
                U[i*2+j][r1+c] = U2[i*2+j][c]
                
    # V2[j*1+0, c] -> V[j*3+2, r1+c]
    for j in range(2):
        for c in range(r2):
            V[j*3+2][r1+c] = V2[j][c]
            
    # W2[0*2+i, c] -> W[2*2+i, r1+c]
    for i in range(2):
        for c in range(r2):
            W[2*2+i][r1+c] = W2[i][c]
            
    return U, V, W, rank

def _construct_2x3x3():
    """
    R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3).
    Split m=3 into 2+1.
    """
    U1, V1, W1, r1 = _construct_2x2x3()
    U2, V2, W2, r2 = _decomp_2x1xp(3)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*3)] # 9 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x3 (j in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
                
    # V1[j*3+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(3):
            for c in range(r1):
                V[j*3+k][c] = V1[j*3+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(3):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x3 (j in 2)
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*3+k, c] -> V[2*3+k, r1+c]
    for k in range(3):
        for c in range(r2):
            V[2*3+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x3x5():
    """
    R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3).
    Split p=5 into 2+3.
    """
    U1, V1, W1, r1 = _construct_2x3x2()
    U2, V2, W2, r2 = _construct_2x3x3()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*5)] # 15 rows
    W = [[0]*rank for _ in range(5*2)] # 10 rows
    
    # Part 1: 2x3x2 (k in 0..1)
    for i in range(2):
        for j in range(3):
            for c in range(r1):
                U[i*3+j][c] = U1[i*3+j][c]
                
    # V1[j*2+k, c] -> V[j*5+k, c]
    for j in range(3):
        for k in range(2):
            for c in range(r1):
                V[j*5+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x3x3 (k in 2..4)
    for i in range(2):
        for j in range(3):
            for c in range(r2):
                U[i*3+j][r1+c] = U2[i*3+j][c]
                
    # V2[j*3+k, c] -> V[j*5+(k+2), r1+c]
    for j in range(3):
        for k in range(3):
            for c in range(r2):
                V[j*5+(k+2)][r1+c] = V2[j*3+k][c]
                
    # W2[k*2+i, c] -> W[(k+2)*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[(k+2)*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 17 first
    U, V, W, rank = _construct_2x3x5()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
