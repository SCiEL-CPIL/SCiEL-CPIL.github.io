import time
import sys

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 150.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # s: 0..6
    # U_coeffs[s] is a list of (index_in_2x2, coeff)
    # V_coeffs[s] is a list of (index_in_2x2, coeff)
    # Indices in 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_coeffs = [
        [(0, 1), (3, 1)],       # P0
        [(2, 1), (3, 1)],       # P1
        [(0, 1)],               # P2
        [(3, 1)],               # P3
        [(0, 1), (1, 1)],       # P4
        [(2, 1), (0, -1)],      # P5
        [(1, 1), (3, -1)]       # P6
    ]
    
    V_coeffs = [
        [(0, 1), (3, 1)],       # P0
        [(0, 1)],               # P1
        [(2, 1), (3, -1)],      # P2
        [(1, 1), (0, -1)],      # P3
        [(3, 1)],               # P4
        [(0, 1), (2, 1)],       # P5
        [(1, 1), (3, 1)]        # P6
    ]
    
    # Contributions to Z (2x2 result)
    # Z[ab][s] = coeff
    # a,b in {0,1}
    # Z00 = P0 + P3 - P4 + P6
    # Z01 = P2 + P4
    # Z10 = P1 + P3
    # Z11 = P0 - P1 + P2 + P5
    
    Z_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0, 1): {2: 1, 4: 1},
        (1, 0): {1: 1, 3: 1},
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base_r = i * 63 + j * 21 + k * 7
                
                for s in range(7):
                    r = base_r + s
                    
                    # Fill U
                    for idx, coeff in U_coeffs[s]:
                        # idx 0: (0,0) -> A[2i, 2j]
                        # idx 1: (0,1) -> A[2i, 2j+1]
                        # idx 2: (1,0) -> A[2i+1, 2j]
                        # idx 3: (1,1) -> A[2i+1, 2j+1]
                        row_a = 2 * i + (idx // 2)
                        col_a = 2 * j + (idx % 2)
                        u_idx = row_a * m + col_a
                        U[u_idx][r] += coeff
                        
                    # Fill V
                    for idx, coeff in V_coeffs[s]:
                        # idx 0: (0,0) -> B[2j, 2k]
                        # idx 1: (0,1) -> B[2j, 2k+1]
                        # idx 2: (1,0) -> B[2j+1, 2k]
                        # idx 3: (1,1) -> B[2j+1, 2k+1]
                        row_b = 2 * j + (idx // 2)
                        col_b = 2 * k + (idx % 2)
                        v_idx = row_b * p + col_b
                        V[v_idx][r] += coeff
                        
                    # Fill W
                    for (a, b), s_coeffs in Z_coeffs.items():
                        if s in s_coeffs:
                            coeff_w = s_coeffs[s]
                            # C row = 2i + a, C col = 2k + b
                            row_c = 2 * i + a
                            col_c = 2 * k + b
                            # W index is col_c * n + row_c
                            w_idx = col_c * n + row_c
                            W[w_idx][r] += coeff_w

    # Check time
    if time.time() > deadline - 5:
        pass # Return whatever we have, though construction is fast

    return {"U": U, "V": V, "W": W}
