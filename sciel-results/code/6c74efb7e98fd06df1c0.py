import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _get_strassen_factors_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Block indices: 0=00, 1=01, 2=10, 3=11
    # A_coeffs[f] maps block_idx -> coefficient
    A_coeffs = [
        {0: 1, 3: 1}, # M1 = (A00+A11)(B00+B11)
        {2: 1, 3: 1}, # M2 = (A10+A11)B00
        {0: 1},       # M3 = A00(B01-B11)
        {3: 1},       # M4 = A11(B10-B00)
        {0: 1, 1: 1}, # M5 = (A00+A01)B11
        {2: 1, 0: -1},# M6 = (A10-A00)(B00+B01)
        {1: 1, 3: -1} # M7 = (A01-A11)(B10+B11)
    ]
    
    B_coeffs = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        {0: 1, 3: 1}, # M1: C00+1, C11+1
        {2: 1, 3: -1},# M2: C10+1, C11-1
        {1: 1, 3: 1}, # M3: C01+1, C11+1
        {0: 1, 2: 1}, # M4: C00+1, C10+1
        {0: -1, 1: 1},# M5: C00-1, C01+1
        {3: 1},       # M6: C11+1
        {0: 1}        # M7: C00+1
    ]
    
    # For 2x2, each block is 1x1, so U_half = V_half = W_half = [[1]]
    for f in range(7):
        for b_idx, coeff in A_coeffs[f].items():
            r = b_idx // 2
            c = b_idx % 2
            idx = r * 2 + c
            U[idx, f] += coeff
        
        for b_idx, coeff in B_coeffs[f].items():
            r = b_idx // 2
            c = b_idx % 2
            idx = r * 2 + c
            V[idx, f] += coeff
        
        for b_idx, coeff in C_coeffs[f].items():
            # W index is k*2+i where k is row, i is col of C block
            k = b_idx // 2
            i = b_idx % 2
            idx = k * 2 + i
            W[idx, f] += coeff
    
    return U, V, W

def _get_strassen_factors_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _get_strassen_factors_2x2()
    rank2 = 7
    n = 4
    n_half = 2
    
    U = np.zeros((n*n, 7 * rank2))
    V = np.zeros((n*n, 7 * rank2))
    W = np.zeros((n*n, 7 * rank2))
    
    A_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    B_coeffs = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
        {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
    ]
    C_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
        {0: -1, 1: 1}, {3: 1}, {0: 1}
    ]
    
    for f in range(7):
        col_offset = f * rank2
        for l in range(rank2):
            col = col_offset + l
            
            # U part: place U2[:, l] into blocks
            for b_idx, coeff in A_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        U[global_idx, col] += coeff * U2[local_idx, l]
            
            # V part
            for b_idx, coeff in B_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        V[global_idx, col] += coeff * V2[local_idx, l]
            
            # W part: W index is k*n+i
            for b_idx, coeff in C_coeffs[f].items():
                k_start = (b_idx // 2) * n_half
                i_start = (b_idx % 2) * n_half
                for k_local in range(n_half):
                    for i_local in range(n_half):
                        local_idx = k_local * n_half + i_local
                        global_idx = (k_start + k_local) * n + (i_start + i_local)
                        W[global_idx, col] += coeff * W2[local_idx, l]
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Convert to numpy for efficient computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Must have T[a,b,c] = 1 if (i,j,k) is valid, else 0
    
    # Check all 1 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check some 0 positions (sample)
    import random
    random.seed(42)
    for _ in range(5000):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        a = i * m + j
        b = j * p + k
        c_alt = random.randint(0, p*n-1)
        c_correct = k * n + i
        if c_alt == c_correct:
            continue
        s = np.dot(U_arr[a], V_arr[b] * W_arr[c_alt])
        if abs(s) > 1e-9:
            return False
    return True

def _construct_strassen_4x8():
    """
    Constructs a rank-98 decomposition for 4x4x8 matrix multiplication.
    View as two independent 4x4x4 multiplications.
    """
    n, m, p = 4, 4, 8
    
    # Get 4x4x4 Strassen factors
    U4, V4, W4 = _get_strassen_factors_4x4()
    rank4 = U4.shape[1] # Should be 49
    
    U = np.zeros((n*m, 2*rank4))
    V = np.zeros((m*p, 2*rank4))
    W = np.zeros((p*n, 2*rank4))
    
    # Left half (columns 0-3 of B, rows 0-3 of C)
    U[:, 0:rank4] = U4
    V[0:16, 0:rank4] = V4
    W[0:16, 0:rank4] = W4
    
    # Right half (columns 4-7 of B, rows 4-7 of C)
    U[:, rank4:2*rank4] = U4
    
    # V for right half: b = j*4 + k where k is 4..7
    # V4 row index: j*4 + k_local (k_local 0..3)
    # V row index: j*4 + (k_local + 4)
    for j in range(4):
        for k_local in range(4):
            v_row_in_V4 = j * 4 + k_local
            v_row_in_V = j * 4 + (k_local + 4)
            V[v_row_in_V, rank4:2*rank4] = V4[v_row_in_V4, :]
    
    # W for right half: c = k*4 + i where k is 4..7
    # W4 row index: k_local*4 + i
    # W row index: (k_local+4)*4 + i
    for k_local in range(4):
        for i in range(4):
            w_row_in_W4 = k_local * 4 + i
            w_row_in_W = (k_local + 4) * 4 + i
            W[w_row_in_W, rank4:2*rank4] = W4[w_row_in_W4, :]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try Strassen 4x8
    try:
        U, V, W = _construct_strassen_4x8()
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
