import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Total rank
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Build 3x3x2 block decomposition
    def build_3x3x2(k_off):
        # k_off is 0, 3, 6? No, we split 7 into 2,2,2,1.
        # So k_off will be 0, 2, 4.
        
        # 1. Strassen core for A[0:2, 0:2] * B[0:2, k_off:k_off+2]
        # A indices: 0,1,3,4
        # B indices: j*7 + k_off + k_local for j in 0,1 and k_local in 0,1
        # C indices: (k_off + k_local)*3 + i for i in 0,1 and k_local in 0,1
        
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        # A: 0, 4. B: 0*7+k_off+0, 1*7+k_off+1. C: (k_off+0)*3+0, (k_off+1)*3+1
        add_term([0, 4], [1, 1], 
                 [0*7 + k_off + 0, 1*7 + k_off + 1], [1, 1],
                 [(k_off+0)*3 + 0, (k_off+1)*3 + 1], [1, 1])
                 
        # M2 = (A10 + A11)B00 -> C10
        # A: 3, 4. B: 0*7+k_off+0. C: (k_off+0)*3+1
        add_term([3, 4], [1, 1], 
                 [0*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 1], [1])
                 
        # M3 = A00(B10 + B11) -> C01
        # A: 0. B: 1*7+k_off+0, 1*7+k_off+1. C: (k_off+1)*3+0
        add_term([0], [1], 
                 [1*7 + k_off + 0, 1*7 + k_off + 1], [1, 1],
                 [(k_off+1)*3 + 0], [1])
                 
        # M4 = A11(B00 - B11) -> C00 + C10
        # A: 4. B: 0*7+k_off+0, 1*7+k_off+1. C: (k_off+0)*3+0, (k_off+0)*3+1
        add_term([4], [1], 
                 [0*7 + k_off + 0, 1*7 + k_off + 1], [1, -1],
                 [(k_off+0)*3 + 0, (k_off+0)*3 + 1], [1, 1])
                 
        # M5 = (A00 + A10)(B10 + B11) -> C00 - C01? No, Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # So M5 contributes -1 to C00 and +1 to C01.
        # A: 0, 3. B: 1*7+k_off+0, 1*7+k_off+1. C: (k_off+0)*3+0, (k_off+1)*3+0
        add_term([0, 3], [1, 1], 
                 [1*7 + k_off + 0, 1*7 + k_off + 1], [1, 1],
                 [(k_off+0)*3 + 0, (k_off+1)*3 + 0], [-1, 1])
                 
        # M6 = (A10 - A00)(B00 + B10) -> C11
        # A: 3, 0. B: 0*7+k_off+0, 1*7+k_off+0. C: (k_off+1)*3+1
        add_term([3, 0], [1, -1], 
                 [0*7 + k_off + 0, 1*7 + k_off + 0], [1, 1],
                 [(k_off+1)*3 + 1], [1])
                 
        # M7 = (A11 - A10)(B10 + B11) -> C00 + C11
        # A: 4, 3. B: 1*7+k_off+0, 1*7+k_off+1. C: (k_off+0)*3+0, (k_off+1)*3+1
        add_term([4, 3], [1, -1], 
                 [1*7 + k_off + 0, 1*7 + k_off + 1], [1, 1],
                 [(k_off+0)*3 + 0, (k_off+1)*3 + 1], [1, 1])

        # 2. Correction A12 * B2 (4 terms)
        # A12: i in 0,1; j=2. Indices: 2, 5.
        # B2: j=2; k in k_off, k_off+1. Indices: 2*7+k_off, 2*7+k_off+1.
        # C top: i in 0,1; k in k_off, k_off+1.
        
        # A02 * B2_k0 -> C0_k0
        add_term([2], [1], 
                 [2*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 0], [1])
                 
        # A02 * B2_k1 -> C0_k1
        add_term([2], [1], 
                 [2*7 + k_off + 1], [1],
                 [(k_off+1)*3 + 0], [1])
                 
        # A12 * B2_k0 -> C1_k0
        add_term([5], [1], 
                 [2*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 1], [1])
                 
        # A12 * B2_k1 -> C1_k1
        add_term([5], [1], 
                 [2*7 + k_off + 1], [1],
                 [(k_off+1)*3 + 1], [1])

        # 3. Correction A21 * B1 (4 terms)
        # A21: i=2; j in 0,1. Indices: 6, 7.
        # B1: j in 0,1; k in k_off, k_off+1. Indices: j*7+k_off+kl.
        # C bot: i=2; k in k_off, k_off+1.
        
        # A20 * B0_k0 -> C2_k0
        add_term([6], [1], 
                 [0*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 2], [1])
                 
        # A20 * B0_k1 -> C2_k1
        add_term([6], [1], 
                 [0*7 + k_off + 1], [1],
                 [(k_off+1)*3 + 2], [1])
                 
        # A21 * B1_k0 -> C2_k0
        add_term([7], [1], 
                 [1*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 2], [1])
                 
        # A21 * B1_k1 -> C2_k1
        add_term([7], [1], 
                 [1*7 + k_off + 1], [1],
                 [(k_off+1)*3 + 2], [1])

        # 4. Correction A22 * B2 (2 terms)
        # A22: i=2, j=2. Index: 8.
        # B2: j=2; k in k_off, k_off+1.
        # C bot: i=2; k in k_off, k_off+1.
        
        # A22 * B2_k0 -> C2_k0
        add_term([8], [1], 
                 [2*7 + k_off + 0], [1],
                 [(k_off+0)*3 + 2], [1])
                 
        # A22 * B2_k1 -> C2_k1
        add_term([8], [1], 
                 [2*7 + k_off + 1], [1],
                 [(k_off+1)*3 + 2], [1])

    # Build 3x3x1 block decomposition (schoolbook)
    def build_3x3x1(k_off):
        for i in range(3):
            for j in range(3):
                add_term([i*m + j], [1], 
                         [j*p + k_off], [1],
                         [k_off*n + i], [1])

    # Execute construction
    build_3x3x2(0)
    build_3x3x2(2)
    build_3x3x2(4)
    build_3x3x1(6)
    
    return {"U": U, "V": V, "W": W}
