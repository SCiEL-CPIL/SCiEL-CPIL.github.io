import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients
    # A-forms for M1..M7
    # Indices: 00, 01, 10, 11
    A_forms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # B-forms for M1..M7
    # Local indices: 00, 01, 10, 11
    B_forms_local = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [-1, 0, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # W-forms for M1..M7
    # Local output indices: P00, P10, P01, P11
    W_forms_local = [
        [1, 0, 0, 1], # M1 in P00(1), P11(1)
        [0, 1, 0, -1],# M2 in P10(1), P11(-1)
        [0, 0, 1, 1], # M3 in P01(1), P11(1)
        [1, 1, 0, 0], # M4 in P00(1), P10(1)
        [-1, 0, 1, 0],# M5 in P00(-1), P01(1)
        [0, 0, 0, 1], # M6 in P11(1)
        [1, 0, 0, 0]  # M7 in P00(1)
    ]
    
    col = 0
    for k in range(7):
        for s in range(7):
            # U
            for i in range(4):
                U[i, col] = A_forms[s][i]
            
            # V
            # Local indices 0,1,2,3 map to global:
            # 0 -> 2k
            # 1 -> 2k+1
            # 2 -> 14+2k
            # 3 -> 14+2k+1
            local_indices_v = [2*k, 2*k+1, 14+2*k, 14+2*k+1]
            for i in range(4):
                val = B_forms_local[s][i]
                if val != 0:
                    V[local_indices_v[i], col] = val
            
            # W
            # Local indices 0,1,2,3 map to global:
            # 0 -> 4k
            # 1 -> 4k+1
            # 2 -> 4k+2
            # 3 -> 4k+3
            local_indices_w = [4*k, 4*k+1, 4*k+2, 4*k+3]
            for i in range(4):
                val = W_forms_local[s][i]
                if val != 0:
                    W[local_indices_w[i], col] = val
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
