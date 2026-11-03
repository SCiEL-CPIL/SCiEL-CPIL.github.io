import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Rank 44 decomposition
    # Split A into:
    # a = A[0,0] (1x1)
    # b = A[0,1:3] (1x2)
    # c = A[1:3,0] (2x1)
    # d = A[1:3,1:3] (2x2)
    #
    # Split B into:
    # B1 = B[0,:] (1x5)
    # B2 = B[1:3,:] (2x5)
    #
    # C = [a*B1 + b*B2]  (1x5)
    #     [c*B1 + d*B2]  (2x5)
    #
    # Terms:
    # 1. a*B1: 1x1x5 -> 5 terms (Schoolbook)
    # 2. b*B2: 1x2x5 -> 10 terms (Schoolbook)
    # 3. c*B1: 2x1x5 -> 10 terms (Schoolbook)
    # 4. d*B2: 2x2x5 -> Split B2 into B2a (2x2) and B2b (2x3)
    #    d*B2a: 2x2x2 -> 7 terms (Strassen)
    #    d*B2b: 2x2x3 -> 12 terms (Schoolbook)
    # Total Rank: 5 + 10 + 10 + 7 + 12 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w):
        nonlocal col
        for idx, coeff in zip(u_indices, coeffs_u):
            U[idx][col] = coeff
        for idx, coeff in zip(v_indices, coeffs_v):
            V[idx][col] = coeff
        for idx, coeff in zip(w_indices, coeffs_w):
            W[idx][col] = coeff
        col += 1

    # 1. a * B1 (A[0,0] * B[0, k] for k=0..4)
    # C[0, k] += A[0,0] * B[0, k]
    for k in range(5):
        add_term(
            [0 * m + 0],  # A[0,0]
            [0 * p + k],  # B[0, k]
            [k * n + 0],  # C[0, k]
            [1], [1], [1]
        )
        
    # 2. b * B2 (A[0, j] * B[j, k] for j in {1,2}, k=0..4)
    # C[0, k] += A[0,1]*B[1,k] + A[0,2]*B[2,k]
    for j in [1, 2]:
        for k in range(5):
            add_term(
                [0 * m + j],  # A[0, j]
                [j * p + k],  # B[j, k]
                [k * n + 0],  # C[0, k]
                [1], [1], [1]
            )

    # 3. c * B1 (A[i, 0] * B[0, k] for i in {1,2}, k=0..4)
    # C[i, k] += A[i,0] * B[0, k]
    for i in [1, 2]:
        for k in range(5):
            add_term(
                [i * m + 0],  # A[i, 0]
                [0 * p + k],  # B[0, k]
                [k * n + i],  # C[i, k]
                [1], [1], [1]
            )

    # 4. d * B2 (A[i, j] * B[j, k] for i,j in {1,2}, k=0..4)
    # Split B2 into B2a (cols 0,1) and B2b (cols 2,3,4)
    
    # 4a. d * B2a (2x2x2) using Strassen
    # d = [[A[1,1], A[1,2]], [A[2,1], A[2,2]]]
    # B2a = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # Result contributes to C[1,0], C[1,1], C[2,0], C[2,1]
    
    # Local indices for d: (0,0)->A[1,1], (0,1)->A[1,2], (1,0)->A[2,1], (1,1)->A[2,2]
    # Local indices for B2a: (0,0)->B[1,0], (0,1)->B[1,1], (1,0)->B[2,0], (1,1)->B[2,1]
    # Local indices for C: (0,0)->C[1,0], (0,1)->C[1,1], (1,0)->C[2,0], (1,1)->C[2,1]
    
    # Strassen Terms:
    # M1 = (d00+d11)(b00+b11) -> C00 += 1, C11 += 1
    # M2 = (d10+d11)b00 -> C10 += 1
    # M3 = d00(b10-b11) -> C01 += 1, C10 += 1
    # M4 = d11(b01-b00) -> C00 -= 1, C01 += 1
    # M5 = (d00+d01)b11 -> C00 += 1
    # M6 = (d10-d00)(b00+b01) -> C11 += 1
    # M7 = (d01-d11)(b10+b11) -> C00 -= 1, C10 += 1
    
    # M1
    add_term(
        [1*m+1, 2*m+2], # A[1,1], A[2,2]
        [1*p+0, 2*p+1], # B[1,0], B[2,1]
        [0*n+1, 1*n+2], # C[1,0], C[2,1]
        [1, 1], [1, 1], [1, 1]
    )
    # M2
    add_term(
        [2*m+1, 2*m+2], # A[2,1], A[2,2]
        [1*p+0],        # B[1,0]
        [0*n+2],        # C[2,0]
        [1, 1], [1], [1]
    )
    # M3
    add_term(
        [1*m+1],        # A[1,1]
        [2*p+0, 2*p+1],# B[2,0], B[2,1]
        [1*n+1, 0*n+2],# C[1,1], C[2,0]
        [1], [1, -1], [1, 1]
    )
    # M4
    add_term(
        [2*m+2],        # A[2,2]
        [1*p+1, 1*p+0],# B[1,1], B[1,0]
        [0*n+1, 1*n+1],# C[1,0], C[1,1]
        [1], [1, -1], [-1, 1]
    )
    # M5
    add_term(
        [1*m+1, 1*m+2],# A[1,1], A[1,2]
        [2*p+1],       # B[2,1]
        [0*n+1],       # C[1,0]
        [1, 1], [1], [1]
    )
    # M6
    add_term(
        [2*m+1, 1*m+1],# A[2,1], A[1,1]
        [1*p+0, 1*p+1],# B[1,0], B[1,1]
        [1*n+2],       # C[2,1]
        [1, -1], [1, 1], [1]
    )
    # M7
    add_term(
        [1*m+2, 2*m+2],# A[1,2], A[2,2]
        [2*p+0, 2*p+1],# B[2,0], B[2,1]
        [0*n+1, 0*n+2],# C[1,0], C[2,0]
        [1, -1], [1, 1], [-1, 1]
    )

    # 4b. d * B2b (2x2x3) using Schoolbook
    # B2b cols are 2,3,4
    for i in [1, 2]:
        for j in [1, 2]:
            for k in [2, 3, 4]:
                add_term(
                    [i*m+j],  # A[i,j]
                    [j*p+k],  # B[j,k]
                    [k*n+i],  # C[i,k]
                    [1], [1], [1]
                )

    return {"U": U, "V": V, "W": W}
