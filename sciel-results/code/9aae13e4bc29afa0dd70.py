import time

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a21+a22)(b11)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = (a11)(b12-b22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = (a22)(b21-b11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a11+a12)(b22)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (a21-a11)(b11+b12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a12-a22)(b21+b22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    return patterns

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    t0 = time.time()
    deadline = 120.0
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2()
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_off, b_j_off, b_k_off, c_off) in blocks:
        for a_pat, b_pat, c_pat in strassen_patterns:
            for local_a_idx, coeff in a_pat.items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_off + local_c
                U[global_i * m + global_j][col] += coeff
            
            for local_b_idx, coeff in b_pat.items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_j_off + local_r
                global_k = b_k_off + local_c
                V[global_j * p + global_k][col] += coeff
            
            for local_c_idx, coeff in c_pat.items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_off + local_c
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - t0 > deadline - 5:
            break
    
    return {"U": U, "V": V, "W": W}
