import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # (6, 34)
    V = [[0] * rank for _ in range(m * p)]  # (18, 34)
    W = [[0] * rank for _ in range(p * n)]  # (12, 34)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, values):
        nonlocal col
        for u_idx, val in zip(u_indices, values):
            U[u_idx][col] += val
        for v_idx, val in zip(v_indices, values):
            V[v_idx][col] += val
        for w_idx, val in zip(w_indices, values):
            W[w_idx][col] += val
        col += 1

    # Block 1: k in {0, 1, 2}
    # Strassen for A_L (j=0,1) * B_T1 (j=0,1, k=0,1)
    # U indices: i*3+j
    # V indices: j*6+k
    # W indices: k*2+i
    
    # M1: (a00+a10)(b00+b10) -> C00, C11
    add_term([0, 3], [0, 6], [0, 3], [1, 1, 1, 1])
    # M2: (a10+a00)b00 -> C10, C11
    add_term([3, 0], [0], [1, 3], [1, 1, 1, 1])
    # M3: a00(b10-b00) -> C01, C11
    add_term([0], [6, 0], [2, 3], [1, 1, -1, 1, 1])
    # M4: a10(b01-b00) -> C00, C10
    add_term([3], [7, 0], [0, 1], [1, 1, -1, 1, 1])
    # M5: (a00+a01)b11 -> C00, C01
    add_term([0, 1], [7], [0, 2], [1, 1, 1, -1, 1])
    # M6: (a10-a00)(b00+b10) -> C11
    add_term([3, 0], [0, 6], [3], [1, -1, 1, 1, 1])
    # M7: (a01-a10)(b10+b11) -> C00, C11
    add_term([1, 3], [6, 7], [0, 3], [1, -1, 1, 1, 1, 1])

    # Schoolbook for A_L (j=0,1) * B_T2 (j=0,1, k=2)
    # S1: a00 b02 -> C02 (W4)
    add_term([0], [2], [4], [1, 1, 1])
    # S2: a01 b12 -> C02 (W4)
    add_term([1], [8], [4], [1, 1, 1])
    # S3: a10 b02 -> C12 (W5)
    add_term([3], [2], [5], [1, 1, 1])
    # S4: a11 b12 -> C12 (W5)
    add_term([4], [8], [5], [1, 1, 1])

    # Schoolbook for A_R (j=2) * B_B (j=2, k=0,1,2)
    # R1: a02 b20 -> C00 (W0)
    add_term([2], [12], [0], [1, 1, 1])
    # R2: a02 b21 -> C01 (W2)
    add_term([2], [13], [2], [1, 1, 1])
    # R3: a02 b22 -> C02 (W4)
    add_term([2], [14], [4], [1, 1, 1])
    # R4: a12 b20 -> C10 (W1)
    add_term([5], [12], [1], [1, 1, 1])
    # R5: a12 b21 -> C11 (W3)
    add_term([5], [13], [3], [1, 1, 1])
    # R6: a12 b22 -> C12 (W5)
    add_term([5], [14], [5], [1, 1, 1])

    # Block 2: k in {3, 4, 5}
    # Strassen for A_L (j=0,1) * B_T1 (j=0,1, k=3,4)
    # V indices: j*6+k. j=0: 3,4. j=1: 9,10.
    # W indices: k*2+i. k=3: 6,7. k=4: 8,9.
    
    # M1: (a00+a10)(b03+b13) -> C03, C13
    add_term([0, 3], [3, 9], [6, 9], [1, 1, 1, 1])
    # M2: (a10+a00)b03 -> C13, C13? No, C13, C13?
    # C indices: C_{1,3} is W7, C_{1,3}? No.
    # C_{i,k}. i=1, k=3 -> W[3*2+1] = W7.
    # C_{i,k}. i=1, k=3? No, M2 contributes to C_{10} and C_{11} in local 2x2.
    # Local C_{0,0} -> Global C_{0,3} (W6).
    # Local C_{0,1} -> Global C_{0,4} (W8).
    # Local C_{1,0} -> Global C_{1,3} (W7).
    # Local C_{1,1} -> Global C_{1,4} (W9).
    
    # M1: Local C00, C11 -> W6, W9
    add_term([0, 3], [3, 9], [6, 9], [1, 1, 1, 1])
    # M2: Local C10, C11 -> W7, W9
    add_term([3, 0], [3], [7, 9], [1, 1, 1, 1])
    # M3: Local C01, C11 -> W8, W9
    add_term([0], [9, 3], [8, 9], [1, 1, -1, 1, 1])
    # M4: Local C00, C10 -> W6, W7
    add_term([3], [10, 3], [6, 7], [1, 1, -1, 1, 1])
    # M5: Local C00, C01 -> W6, W8
    add_term([0, 1], [10], [6, 8], [1, 1, 1, -1, 1])
    # M6: Local C11 -> W9
    add_term([3, 0], [3, 9], [9], [1, -1, 1, 1, 1])
    # M7: Local C00, C11 -> W6, W9
    add_term([1, 3], [9, 10], [6, 9], [1, -1, 1, 1, 1, 1])

    # Schoolbook for A_L (j=0,1) * B_T2 (j=0,1, k=5)
    # V indices: j*6+5. j=0: 5. j=1: 11.
    # W indices: k=5. i=0: W10. i=1: W11.
    
    # S1: a00 b05 -> C05 (W10)
    add_term([0], [5], [10], [1, 1, 1])
    # S2: a01 b15 -> C05 (W10)
    add_term([1], [11], [10], [1, 1, 1])
    # S3: a10 b05 -> C15 (W11)
    add_term([3], [5], [11], [1, 1, 1])
    # S4: a11 b15 -> C15 (W11)
    add_term([4], [11], [11], [1, 1, 1])

    # Schoolbook for A_R (j=2) * B_B (j=2, k=3,4,5)
    # V indices: 12+k. k=3: 15. k=4: 16. k=5: 17.
    # W indices: k*2+i.
    # k=3: W6, W7.
    # k=4: W8, W9.
    # k=5: W10, W11.
    
    # R1: a02 b23 -> C03 (W6)
    add_term([2], [15], [6], [1, 1, 1])
    # R2: a02 b24 -> C04 (W8)
    add_term([2], [16], [8], [1, 1, 1])
    # R3: a02 b25 -> C05 (W10)
    add_term([2], [17], [10], [1, 1, 1])
    # R4: a12 b23 -> C13 (W7)
    add_term([5], [15], [7], [1, 1, 1])
    # R5: a12 b24 -> C14 (W9)
    add_term([5], [16], [9], [1, 1, 1])
    # R6: a12 b25 -> C15 (W11)
    add_term([5], [17], [11], [1, 1, 1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
