import time

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2
    # Each term is defined by:
    # u_coeffs: dict {(i, j): val}
    # v_coeffs: dict {(j, k): val}
    # w_coeffs: dict {(k, i): val}
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (A10 + A11)B00
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}), # C10: 1, C11: -1
        # M3 = A00(B01 - B11)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}), # C01: 1, C11: 1
        # M4 = A11(B10 - B00)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}), # C00: 1, C10: 1
        # M5 = (A00 + A01)B11
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}), # C00: -1, C01: 1
        # M6 = (A10 - A00)(B00 + B01)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}), # C11: 1
        # M7 = (A01 - A11)(B10 + B11)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}), # C00: 1
    ]
    
    # 8 blocks: 2 j-blocks, 4 k-blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                if _time_left() < 0.1:
                    return {"U": U, "V": V, "W": W}
                
                # Add term
                for (i, j), val in u_coeffs.items():
                    if val != 0:
                        U[i * m + j][col] += val
                for (j, k), val in v_coeffs.items():
                    if val != 0:
                        V[j * p + k][col] += val
                for (k, i), val in w_coeffs.items():
                    if val != 0:
                        W[k * n + i][col] += val
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
