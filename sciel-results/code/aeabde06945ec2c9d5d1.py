import numpy as np
import time

DIMENSIONS = [2, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible
    # Decompose m=4 into 2x2 blocks, p=7 into 3x2 + 1
    # For 2x2x2: Strassen gives rank 7
    # For 2x2x1: Schoolbook gives rank 4
    # Total: 2 (inner blocks) * [3 (col blocks of 2) * 7 + 1 (last col) * 4] = 2 * [21 + 4] = 50
    
    # But let's try a smarter approach:
    # Use 2x3x7 blocks: 
    #   2x3x2: can use a 2x2x2 Strassen (7) + 2x1x2 schoolbook (4) = 11? No, that's not right.
    #   Actually 2x3x2: split inner dim 3 into 2+1: 2x2x2 (7) + 2x1x2 (4) = 11
    #   2x3x7: split cols into 2+2+2+1: 3*11 + 4 = 37? No...
    
    # Let me think more carefully about the tensor structure.
    # The multiplication tensor for (n x m) * (m x p) has entries T[a,k,b] = delta_{a,m} * delta_{m,b}
    # We want to find rank r such that T = sum_{i=1}^r U_i V_i W_i where:
    #   U_i has shape (n*m), V_i has shape (m*p), W_i has shape (n*p)
    
    # For 2x4x7, the naive rank is 56.
    # Strassen for 2x2x2 gives rank 7 instead of 8.
    
    # Let's use a block decomposition:
    # Split m into 2+2, p into 2+2+2+1
    # For each 2x2x2 block: rank 7
    # For each 2x2x1 block: rank 4
    # Total: 2 * (3*7 + 1*4) = 2 * 25 = 50
    
    # This is the same as before. Let's try to do better.
    
    # Alternative: use a 2x4x7 decomposition directly
    # The border rank of 2x4x7 is at most the sum of border ranks of sub-blocks
    
    # Let me try: split p into 3+4
    # 2x4x3: border rank? 
    #   2x2x3: 2x2x2 (7) + 2x1x2 (4) = 11? No, 2x1x2 is 2*1*2=4 scalar mults, but border rank is 4
    #   So 2x2x3: 7 + 4 = 11
    #   2x4x3: 2 * 11 = 22
    # 2x4x4: 2 * 2 * 7 = 28 (using 2x2x2 Strassen four times)
    # Total: 22 + 28 = 50
    
    # Same result. Let's try p split as 4+3:
    # 2x4x4: 28
    # 2x4x3: 22
    # Total: 50
    
    # It seems 50 is a natural bound from this approach.
    
    # Can we do better? The theoretical lower bound for 2x4x7 is...
    # The tensor is of format 2x4x7. The minimum rank is at least max(2*4, 4*7, 2*7) = max(8, 28, 14) = 28
    # But that's a very loose bound.
    
    # Let's stick with the rank 50 construction but implement it cleanly.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 50)
    V = [[0] * rank for _ in range(m * p)]   # shape (28, 50)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 50)
    
    col = 0
    
    def add_strassen_222(j0, k0):
        """Strassen for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[:, k0:k0+2]"""
        nonlocal col
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + 0
        c01 = (k0 + 1) * n + 0
        c10 = k0 * n + 1
        c11 = (k0 + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    def add_schoolbook_221(j0, k0):
        """Schoolbook for A[:, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[:, k0:k0+1]"""
        nonlocal col
        a00 = 0 * m + j0
        a01 = 0 * m + (j0 + 1)
        a10 = 1 * m + j0
        a11 = 1 * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0
        
        c00 = k0 * n + 0
        c10 = k0 * n + 1
        
        U[a00][col] += 1
        V[b00][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[a01][col] += 1
        V[b10][col] += 1
        W[c00][col] += 1
        col += 1
        
        U[a10][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        U[a11][col] += 1
        V[b10][col] += 1
        W[c10][col] += 1
        col += 1
    
    # Process inner blocks j=0..1 and j=2..3
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(j_start, k_start)
        add_schoolbook_221(j_start, 6)
    
    return {"U": U, "V": V, "W": W}
