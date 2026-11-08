import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize factor matrices
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_term(r_idx, u_entries, v_entries, w_entries):
        """
        Adds a rank-1 term to the decomposition.
        u_entries: list of (index, value) for U
        v_entries: list of (index, value) for V
        w_entries: list of (index, value) for W
        """
        for idx, val in u_entries:
            U[idx, r_idx] = val
        for idx, val in v_entries:
            V[idx, r_idx] = val
        for idx, val in w_entries:
            W[idx, r_idx] = val
            
    # Index mapping:
    # A (n x m): i in 0..1, j in 0..2 -> idx = i*m + j
    # B (m x p): j in 0..2, k in 0..2 -> idx = j*p + k
    # C (p x n): k in 0..2, i in 0..1 -> idx = k*n + i (Trace convention)
    
    # --- Part 1: Strassen for the 2x2 sub-blocks (cols 0,1 of A; rows 0,1 of B; cols 0,1 of C) ---
    # This handles C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]
    
    # M1: (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
    add_term(0, 
             [(0, 1), (1+2, 1)],  # U: A00, A11
             [(0, 1), (1+2, 1)],  # V: B00, B11
             [(0, 1), (1, 1)])    # W: C00, C11

    # M2: (A10 + A11) * B00 -> C10 += M2, C11 -= M2
    add_term(1,
             [(2, 1), (1+2, 1)],  # U: A10, A11
             [(0, 1)],            # V: B00
             [(1, 1), (0, -1)])   # W: C10, C11

    # M3: A00 * (B01 - B11) -> C01 += M3, C11 += M3
    add_term(2,
             [(0, 1)],            # U: A00
             [(1, 1), (2, -1)],   # V: B01, B11
             [(2, 1), (0, 1)])    # W: C01, C11

    # M4: A11 * (B10 - B00) -> C00 += M4, C10 += M4
    add_term(3,
             [(1+2, 1)],          # U: A11
             [(2, 1), (0, -1)],   # V: B10, B00
             [(0, 1), (1, 1)])    # W: C00, C10

    # M5: (A00 + A01) * B11 -> C00 -= M5, C01 += M5
    add_term(4,
             [(0, 1), (1, 1)],    # U: A00, A01
             [(2, 1)],            # V: B11
             [(0, -1), (2, 1)])   # W: C00, C01

    # M6: (A10 - A00) * (B00 + B01) -> C11 += M6
    add_term(5,
             [(2, 1), (0, -1)],   # U: A10, A00
             [(0, 1), (1, 1)],    # V: B00, B01
             [(0, 1)])            # W: C11

    # M7: (A01 - A11) * (B10 + B11) -> C00 += M7
    add_term(6,
             [(1, 1), (1+2, -1)], # U: A01, A11
             [(2, 1), (3, 1)],    # V: B10, B11
             [(0, 1)])            # W: C00

    # --- Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] ---
    # A is 2x1 (col 2), B is 1x2 (row 2). Result 2x2.
    # C[i, j] = A[i, 2] * B[2, j]
    
    # r=7: A02 * B20 -> C00
    add_term(7, 
             [(2, 1)],  # U: A02
             [(6, 1)],  # V: B20
             [(0, 1)])  # W: C00

    # r=8: A02 * B21 -> C01
    add_term(8,
             [(2, 1)],  # U: A02
             [(7, 1)],  # V: B21
             [(2, 1)])  # W: C01

    # r=9: A12 * B20 -> C10
    add_term(9,
             [(5, 1)],  # U: A12
             [(6, 1)],  # V: B20
             [(1, 1)])  # W: C10

    # r=10: A12 * B21 -> C11
    add_term(10,
             [(5, 1)],  # U: A12
             [(7, 1)],  # V: B21
             [(0, 1)])  # W: C11

    # --- Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2] ---
    # A is 2x3, B is 3x1 (col 2). Result 2x1 (col 2 of C).
    # C[i, 2] = sum_j A[i, j] * B[j, 2]
    
    # r=11: A00 * B02 -> C02
    add_term(11,
             [(0, 1)],  # U: A00
             [(2, 1)],  # V: B02
             [(4, 1)])  # W: C02

    # r=12: A01 * B12 -> C02
    add_term(12,
             [(1, 1)],  # U: A01
             [(5, 1)],  # V: B12
             [(4, 1)])  # W: C02

    # r=13: A02 * B22 -> C02
    add_term(13,
             [(2, 1)],  # U: A02
             [(8, 1)],  # V: B22
             [(4, 1)])  # W: C02

    # r=14: A10 * B02 -> C12
    add_term(14,
             [(3, 1)],  # U: A10
             [(2, 1)],  # V: B02
             [(5, 1)])  # W: C12

    # r=15: A11 * B12 -> C12
    add_term(15,
             [(4, 1)],  # U: A11
             [(5, 1)],  # V: B12
             [(5, 1)])  # W: C12

    # r=16: A12 * B22 -> C12
    add_term(16,
             [(5, 1)],  # U: A12
             [(8, 1)],  # V: B22
             [(5, 1)])  # W: C12

    if time.time() > deadline:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
