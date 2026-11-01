import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21  # 3 pairs * 7 Strassen multiplications
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 21
    
    # For each pair of columns (k1, k2), use Strassen
    # Pairs: (0,1), (2,3), (4,5)
    pairs = [(0, 1), (2, 3), (4, 5)]
    
    for pair_idx, (k1, k2) in enumerate(pairs):
        base = pair_idx * 7
        
        # Strassen multiplications for 2x2 A * 2x2 B (columns k1, k2)
        # A = [[A00, A01], [A10, A11]]
        # B = [[B0k1, B0k2], [B1k1, B1k2]]
        
        # M1 = (A00+A11) * (B0k1+B1k2)
        r = base + 0
        U[0][r] += 1.0  # A00
        U[3][r] += 1.0  # A11
        V[0 * p + k1][r] += 1.0  # B[0,k1]
        V[1 * p + k2][r] += 1.0  # B[1,k2]
        # C[0,k1] += M1, C[1,k2] += M1
        W[k1 * n + 0][r] += 1.0
        W[k2 * n + 1][r] += 1.0
        
        # M2 = (A10+A11) * B0k1
        r = base + 1
        U[2][r] += 1.0  # A10
        U[3][r] += 1.0  # A11
        V[0 * p + k1][r] += 1.0  # B[0,k1]
        # C[1,k1] += M2
        W[k1 * n + 1][r] += 1.0
        
        # M3 = A00 * (B0k2-B1k2)
        r = base + 2
        U[0][r] += 1.0  # A00
        V[0 * p + k2][r] += 1.0  # B[0,k2]
        V[1 * p + k2][r] += -1.0  # B[1,k2]
        # C[0,k2] += M3
        W[k2 * n + 0][r] += 1.0
        
        # M4 = A11 * (B1k1-B0k1)
        r = base + 3
        U[3][r] += 1.0  # A11
        V[1 * p + k1][r] += 1.0  # B[1,k1]
        V[0 * p + k1][r] += -1.0  # B[0,k1]
        # C[0,k1] += M4, C[1,k1] += M4
        W[k1 * n + 0][r] += 1.0
        W[k1 * n + 1][r] += 1.0
        
        # M5 = (A00+A01) * B1k2
        r = base + 4
        U[0][r] += 1.0  # A00
        U[1][r] += 1.0  # A01
        V[1 * p + k2][r] += 1.0  # B[1,k2]
        # C[0,k1] -= M5, C[0,k2] += M5
        W[k1 * n + 0][r] += -1.0
        W[k2 * n + 0][r] += 1.0
        
        # M6 = (A10-A00) * (B0k1+B1k1)
        r = base + 5
        U[2][r] += 1.0  # A10
        U[0][r] += -1.0  # A00
        V[0 * p + k1][r] += 1.0  # B[0,k1]
        V[1 * p + k1][r] += 1.0  # B[1,k1]
        # C[1,k2] += M6
        W[k2 * n + 1][r] += 1.0
        
        # M7 = (A01-A11) * (B1k1+B1k2)
        r = base + 6
        U[1][r] += 1.0  # A01
        U[3][r] += -1.0  # A11
        V[1 * p + k1][r] += 1.0  # B[1,k1]
        V[1 * p + k2][r] += 1.0  # B[1,k2]
        # C[0,k1] += M7
        W[k1 * n + 0][r] += 1.0
    
    return {"U": U, "V": V, "W": W}
