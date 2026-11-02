import numpy as np

def construct():
    n, m, p = 2, 3, 7
    # Total rank expected: 40
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x7 (j=2)
    # C[i,k] = A[i,2] * B[2,k]
    # 14 terms
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2][col] = 1
            V[2 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x2x7 (j=0,1)
    # We use recursive decomposition.
    # L(2,2,7) <= L(2,2,6) + L(2,2,1)
    # L(2,2,6) <= L(2,2,3) + L(2,2,3)
    # L(2,2,3) <= L(2,2,2) + L(2,2,1)
    
    # Define Strassen vectors for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # u, v, w vectors
    strassen = [
        # M1: (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10 gets M2? 
        # Let's verify C formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M2 contributes to C10 (index 2) and C11 (index 3) with -1.
        # So w2 = [0, 0, 1, -1]
        # M3: A00(B01-B11)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # Contributes to C01 (idx 1) and C11 (idx 3).
        # w3 = [0, 1, 0, 1]
        # M4: A11(B10-B00)
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1] # B10 - B00
        # Contributes to C00 (idx 0) and C10 (idx 2).
        # w4 = [1, 0, 1, 0]
        # M5: (A00+A01)B11
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # Contributes to C00 (idx 0) with -1 and C01 (idx 1) with 1.
        # w5 = [-1, 1, 0, 0]
        # M6: (A01-A10)(B00+B01)
        # u6 = [0, 1, -1, 0]
        # v6 = [1, 1, 0, 0]
        # Contributes to C11 (idx 3) with 1.
        # w6 = [0, 0, 0, 1]
        # M7: (A10-A00)(B11+B10)
        # u7 = [0, 0, 1, -1] # A10 - A00
        # v7 = [0, 0, 1, 1] # B10 + B11
        # Contributes to C00 (idx 0) with 1.
        # w7 = [1, 0, 0, 0]
    ]
    
    # Correcting Strassen w vectors based on standard decomposition:
    # M1: C00(+1), C11(+1) -> [1, 0, 0, 1]
    # M2: C10(+1), C11(-1) -> [0, 0, 1, -1]
    # M3: C01(+1), C11(+1) -> [0, 1, 0, 1]
    # M4: C00(+1), C10(+1) -> [1, 0, 1, 0]
    # M5: C00(-1), C01(+1) -> [-1, 1, 0, 0]
    # M6: C11(+1) -> [0, 0, 0, 1]
    # M7: C00(+1) -> [1, 0, 0, 0]
    
    strassen_corrected = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 1, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Schoolbook for 2x2x1 (k=0 only)
    # C[i,k] = sum A[i,j] B[j,k]
    # Terms: (i,j)
    schoolbook_221 = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]), # i=0, j=0 -> C00
        ([0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]), # i=0, j=1 -> C01
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]), # i=1, j=0 -> C10
        ([0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1])  # i=1, j=1 -> C11
    ]
    
    def add_223_block(k_offset):
        nonlocal col
        # Add 7 Strassen terms for k_offset, k_offset+1
        for u, v, w in strassen_corrected:
            # Map u (len 4) to U (len 6, indices 0,1 for j=0,1)
            # u[0]->A00, u[1]->A01, u[2]->A10, u[3]->A11
            # U indices: i*3+j. i=0,1. j=0,1.
            # 00->0, 01->1, 10->3, 11->4
            U[0][col] += u[0]
            U[1][col] += u[1]
            U[3][col] += u[2]
            U[4][col] += u[3]
            
            # Map v (len 4) to V (len 21, indices j*7+k)
            # v[0]->B00, v[1]->B01, v[2]->B10, v[3]->B11
            # V indices: j*7+k. j=0,1. k=k_offset, k_offset+1.
            # 00->0*7+k_offset, 01->0*7+k_offset+1
            # 10->1*7+k_offset, 11->1*7+k_offset+1
            V[0 * 7 + k_offset][col] += v[0]
            V[0 * 7 + k_offset + 1][col] += v[1]
            V[1 * 7 + k_offset][col] += v[2]
            V[1 * 7 + k_offset + 1][col] += v[3]
            
            # Map w (len 4) to W (len 14, indices k*2+i)
            # w[0]->C00, w[1]->C01, w[2]->C10, w[3]->C11
            # W indices: k*2+i.
            # C00 -> k_offset*2+0
            # C01 -> (k_offset+1)*2+0
            # C10 -> k_offset*2+1
            # C11 -> (k_offset+1)*2+1
            W[k_offset * 2 + 0][col] += w[0]
            W[(k_offset + 1) * 2 + 0][col] += w[1]
            W[k_offset * 2 + 1][col] += w[2]
            W[(k_offset + 1) * 2 + 1][col] += w[3]
            col += 1
            
        # Add 4 Schoolbook terms for k_offset+2
        for u, v, w in schoolbook_221:
            U[0][col] += u[0]
            U[1][col] += u[1]
            U[3][col] += u[2]
            U[4][col] += u[3]
            
            # B is 2x1 here, so only B[j, k_offset+2]
            # v[0] is B0, v[1] is B1 (mapped from 2x2 schoolbook where k=0)
            # In 2x2x1 schoolbook, B is 2x1.
            # v[0] corresponds to B[0,0], v[1] to B[1,0].
            V[0 * 7 + k_offset + 2][col] += v[0]
            V[1 * 7 + k_offset + 2][col] += v[1]
            
            # W for 2x1 output? No, C is 2x1.
            # w[0] C00, w[1] C01 (dummy), w[2] C10, w[3] C11 (dummy)
            # In 2x2x1, C is 2x1. Indices 0,1.
            # w[0] -> C[0,0], w[2] -> C[1,0].
            # Global W index: k*2+i.
            # k = k_offset+2.
            # C[0,0] -> (k_offset+2)*2+0
            # C[1,0] -> (k_offset+2)*2+1
            W[(k_offset + 2) * 2 + 0][col] += w[0]
            W[(k_offset + 2) * 2 + 1][col] += w[2]
            col += 1

    def add_221_block(k_offset):
        nonlocal col
        for u, v, w in schoolbook_221:
            U[0][col] += u[0]
            U[1][col] += u[1]
            U[3][col] += u[2]
            U[4][col] += u[3]
            
            V[0 * 7 + k_offset][col] += v[0]
            V[1 * 7 + k_offset][col] += v[1]
            
            W[k_offset * 2 + 0][col] += w[0]
            W[k_offset * 2 + 1][col] += w[2]
            col += 1

    # Build 2x2x7
    # k=0,1,2 -> 2x2x3 block
    add_223_block(0)
    # k=3,4,5 -> 2x2x3 block
    add_223_block(3)
    # k=6 -> 2x2x1 block
    add_221_block(6)
    
    return {"U": U, "V": V, "W": W}
