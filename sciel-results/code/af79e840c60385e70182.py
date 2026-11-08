import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    def get_189_solution():
        # Strassen 2x2 patterns: (U_coeffs, V_coeffs, W_coeffs)
        # Each is a dict mapping (row, col) -> value
        # M1 = (A11+A22)(B11+B22): C11+=M1, C22+=M1
        # M2 = (A21+A22)B11: C21+=M2, C22-=M2
        # M3 = A11(B21-B22): C12+=M3, C22+=M3
        # M4 = A22(B11-B10): C11+=M4, C21+=M4  [note: B10 means B[1,0]]
        # M5 = (A11+A12)B22: C11-=M5, C12+=M5
        # M6 = (A21-A11)(B11+B12): C22+=M6
        # M7 = (A12-A22)(B21+B22): C11+=M7
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        rank = 189
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        
        col_idx = 0
        for I in range(3):
            for J in range(3):
                for K in range(3):
                    if time.time() > deadline:
                        break
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * 2 + pi
                            c = J * 2 + pj
                            U[r * 6 + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * 2 + pi
                            c = K * 2 + pj
                            V[r * 6 + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * 2 + pj
                            i = I * 2 + pi
                            W[k * 6 + i, col_idx] += val
                        col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    sol = get_189_solution()
    return sol
# EVOLVE-BLOCK-END
