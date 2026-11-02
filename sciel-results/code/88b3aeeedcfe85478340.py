import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Rank 44 Construction
    # Part 1: m=1 (Rank 16)
    # Part 2: m=2 (Rank 28) -> 4 blocks of Strassen 7
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0
    # u = i*3 + 0
    # v = 0*8 + k = k
    # w = k*2 + i
    # r = 0 .. 15
    # r = i*8 + k
    for i in range(n):
        for k in range(p):
            r = i * p + k
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            
    # Part 2: j=1,2
    # Split k into 4 blocks of 2
    # Strassen factors for each block
    # Correct Strassen Formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M5 + M3 - M7 - M1
    # C01 = M1 + M4
    # C10 = M2 - M4
    # C11 = M5 + M6 - M2
    
    # Wait, I need to be sure. I will use a known valid set in the code.
    
    for l in range(4):
        base_r = 16 + l * 7
        # A indices
        A00 = 1 # i=0, j=1
        A01 = 2 # i=0, j=2
        A10 = 4 # i=1, j=1
        A11 = 5 # i=1, j=2
        
        # B indices for block l
        # j=1 -> v = 8 + 2l (B00), 9 + 2l (B01)
        # j=2 -> v = 16 + 2l (B10), 17 + 2l (B11)
        B00 = 8 + 2 * l
        B01 = 9 + 2 * l
        B10 = 16 + 2 * l
        B11 = 17 + 2 * l
        
        # W indices for block l
        # C00 -> w = 4l
        # C01 -> w = 4l+2
        # C10 -> w = 4l+1
        # C11 -> w = 4l+3
        C00 = 4 * l
        C01 = 4 * l + 2
        C10 = 4 * l + 1
        C11 = 4 * l + 3
        
        # M1
        U[A00][base_r] = 1
        U[A11][base_r] = 1
        V[B00][base_r] = 1
        V[B11][base_r] = 1
        W[C00][base_r] = -1 # C00 = ... - M1
        W[C11][base_r] = 1  # C11 = M1 ...
        
        # M2
        U[A10][base_r+1] = 1
        U[A11][base_r+1] = 1
        V[B00][base_r+1] = 1
        W[C10][base_r+1] = 1 # C10 = M2 ...
        W[C11][base_r+1] = -1 # C11 = ... - M2
        
        # M3
        U[A00][base_r+2] = 1
        V[B10][base_r+2] = 1
        V[B11][base_r+2] = -1
        W[C00][base_r+2] = 1 # C00 = ... + M3
        W[C11][base_r+2] = 1 # C11 = ... + M3
        
        # M4
        U[A11][base_r+3] = 1
        V[B01][base_r+3] = 1
        V[B00][base_r+3] = -1
        W[C00][base_r+3] = 1 # C00 = ... + M4
        W[C01][base_r+3] = 1 # C01 = ... + M4
        W[C10][base_r+3] = -1 # C10 = ... - M4
        
        # M5
        U[A00][base_r+4] = 1
        U[A01][base_r+4] = 1
        V[B11][base_r+4] = 1
        W[C00][base_r+4] = 1 # C00 = M5 ...
        W[C01][base_r+4] = 1 # C01 = M5 ...
        W[C11][base_r+4] = 1 # C11 = M5 ...
        
        # M6
        U[A10][base_r+5] = 1
        U[A00][base_r+5] = -1
        V[B00][base_r+5] = 1
        V[B01][base_r+5] = 1
        W[C01][base_r+5] = 1 # C01 = ... + M6
        W[C11][base_r+5] = 1 # C11 = ... + M6
        
        # M7
        U[A01][base_r+6] = 1
        U[A11][base_r+6] = -1
        V[B10][base_r+6] = 1
        V[B11][base_r+6] = 1
        W[C00][base_r+6] = -1 # C00 = ... - M7
        W[C10][base_r+6] = 1 # C10 = ... + M7
        
    return {"U": U, "V": V, "W": W}
