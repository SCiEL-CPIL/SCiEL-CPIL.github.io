import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_rank21_strassen():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # Uses 7 multiplications instead of 8
    # The 7 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We apply this to 3 blocks of B (cols 0-1, 2-3, 4-5)
    
    strassen_blocks = [
        # (U coeffs, V coeffs, W coeffs)
        # Indices:
        # U: 0=A00, 1=A01, 2=A10, 3=A11
        # V: 0=B00, 1=B01, 2=B10, 3=B11 (local to block)
        # W: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11 (local to block)
        
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                # Map local V index l to global V index
                # l=0 -> B00, l=1 -> B01, l=2 -> B10, l=3 -> B11
                # Global V index: 6*j + k
                # j=0 for l=0,1; j=1 for l=2,3
                # k = 2*b + (l % 2)
                if l == 0: v_idx = 0 * p + 2 * b
                elif l == 1: v_idx = 0 * p + 2 * b + 1
                elif l == 2: v_idx = 1 * p + 2 * b
                elif l == 3: v_idx = 1 * p + 2 * b + 1
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                # Map local W index (i, k_local) to global W index
                # Global W index: 2*k + i
                # k_global = 2*b + k_local
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _build_rank21_strassen()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
