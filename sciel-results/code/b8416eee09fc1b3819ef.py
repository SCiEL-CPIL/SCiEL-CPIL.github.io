import time

def _strassen_2x2():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    For C = A*B where A,B,C are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    Strassen's 7 products:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)(b11)
    M3 = (a11)(b12-b22)
    M4 = (a22)(b21-b11)
    M5 = (a11+a12)(b22)
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    Returns list of 7 dicts with 'a', 'b', 'c' coefficient dicts.
    A entries indexed as: a11=0, a12=1, a21=2, a22=3
    B entries indexed as: b11=0, b12=1, b21=2, b22=3
    C entries indexed as: c11=0, c12=1, c21=2, c22=3
    """
    patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm on each of the four 2x2 block products:
      A_L * B_TL, A_R * B_BL, A_L * B_TR, A_R * B_BR
    
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention W[k*n+i])
    """
    t0 = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    strassen_patterns = _strassen_2x2()
    
    # We need to decompose C = A B
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    
    # Each term is a 2x2 matrix multiplication.
    # We map each 2x2 multiplication to Strassen patterns.
    
    # Block definitions: (a_j_offset, b_j_offset, b_k_offset, c_k_offset)
    # a_j_offset: column offset in A (0 for left, 2 for right)
    # b_j_offset: row offset in B
    # b_k_offset: col offset in B
    # c_k_offset: col offset in C
    
    blocks = [
        # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (0, 0, 0, 0),
        # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (2, 2, 0, 0),
        # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (0, 0, 2, 2),
        # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_off, b_j_off, b_k_off, c_off) in blocks:
        for pattern in strassen_patterns:
            # Fill U: A entries
            # Local A index 0..3 corresponds to:
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_off + local_c
                U[global_i * m + global_j][col] += coeff
            
            # Fill V: B entries
            # Local B index 0..3 corresponds to:
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_j_off + local_r
                global_k = b_k_off + local_c
                V[global_j * p + global_k][col] += coeff
            
            # Fill W: C entries
            # Local C index 0..3 corresponds to:
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_off + local_c
                # W index is k*n + i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
            # Deadline check
            if time.time() - t0 > deadline - 5:
                break
        if time.time() - t0 > deadline - 5:
            break
    
    return {"U": U, "V": V, "W": W}
