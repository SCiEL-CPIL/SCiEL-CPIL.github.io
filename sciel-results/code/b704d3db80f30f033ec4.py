import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def validate_decomp(U, V, W, n, m, p, rank):
    """Check if the decomposition reconstructs the MM tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if total != 1:
                    return False
    return True

def schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def get_3x3x3_decomp():
    """
    Returns (U, V, W, rank) for 3x3x3.
    Currently uses Schoolbook (27) as a safe baseline.
    """
    n, m, p = 3, 3, 3
    U_fb, V_fb, W_fb, rank_fb = schoolbook_decomp(n, m, p)
    return U_fb, V_fb, W_fb, rank_fb

def get_3x3x2_decomp():
    """
    Returns (U, V, W, rank) for 3x3x2.
    Currently uses Schoolbook (18) as a safe baseline.
    """
    n, m, p = 3, 3, 2
    U_fb, V_fb, W_fb, rank_fb = schoolbook_decomp(n, m, p)
    return U_fb, V_fb, W_fb, rank_fb

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose 3x3x8 into 3x3x3 + 3x3x3 + 3x3x2
    U1, V1, W1, r1 = get_3x3x3_decomp()
    U2, V2, W2, r2 = get_3x3x3_decomp()
    U3, V3, W3, r3 = get_3x3x2_decomp()
    
    total_rank = r1 + r2 + r3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Block 1 (k=0,1,2)
    for i in range(n * m):
        for r in range(r1):
            U[i][r] = U1[i][r]
    for j in range(m):
        for k in range(3):
            for r in range(r1):
                V[j*p+k][r] = V1[j*3+k][r]
    for k in range(3):
        for i in range(n):
            for r in range(r1):
                W[k*n+i][r] = W1[k*3+i][r]
                
    # Fill Block 2 (k=3,4,5)
    for i in range(n * m):
        for r in range(r2):
            U[i][r1 + r] = U2[i][r]
    for j in range(m):
        for k in range(3, 6):
            for r in range(r2):
                V[j*p+k][r1 + r] = V2[j*3 + (k-3)][r]
    for k in range(3, 6):
        for i in range(n):
            for r in range(r2):
                W[k*n+i][r1 + r] = W2[(k-3)*3+i][r]
                
    # Fill Block 3 (k=6,7)
    for i in range(n * m):
        for r in range(r3):
            U[i][r1 + r2 + r] = U3[i][r]
    for j in range(m):
        for k in range(6, 8):
            for r in range(r3):
                V[j*p+k][r1 + r2 + r] = V3[j*2 + (k-6)][r]
    for k in range(6, 8):
        for i in range(n):
            for r in range(r3):
                W[k*n+i][r1 + r2 + r] = W3[(k-6)*3+i][r]
                
    if validate_decomp(U, V, W, n, m, p, total_rank):
        return {"U": U, "V": V, "W": W}
    else:
        U_fb, V_fb, W_fb, _ = schoolbook_decomp(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
