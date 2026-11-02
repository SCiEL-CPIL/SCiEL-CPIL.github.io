"""Matrix multiplication 2x2x15 tensor decomposition"""
import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Use rank 60 (schoolbook) as a guaranteed correct baseline
    # Then we could try to optimize, but let's start with something that works
    rank = 60  # n*m*p = 2*2*15 = 60
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 60
    V = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # 30 x 60
    
    # A indices: 0=a(A00), 1=b(A01), 2=c(A10), 3=d(A11)
    # B indices: row*p + col, so B[0,k] is index k, B[1,k] is index p+k
    # C indices: k*n + i = C[i,k], so C[0,k] is index 2k, C[1,k] is index 2k+1
    
    # Schoolbook: each output C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    r = 0
    for k in range(p):
        # Term: a * B[0,k] -> contributes to C[0,k]
        U[0][r] = 1
        V[0 * p + k][r] = 1
        W[2 * k][r] = 1
        r += 1
        
        # Term: b * B[1,k] -> contributes to C[0,k]
        U[1][r] = 1
        V[1 * p + k][r] = 1
        W[2 * k][r] = 1
        r += 1
        
        # Term: c * B[0,k] -> contributes to C[1,k]
        U[2][r] = 1
        V[0 * p + k][r] = 1
        W[2 * k + 1][r] = 1
        r += 1
        
        # Term: d * B[1,k] -> contributes to C[1,k]
        U[3][r] = 1
        V[1 * p + k][r] = 1
        W[2 * k + 1][r] = 1
        r += 1
    
    return {"U": U, "V": V, "W": W}
