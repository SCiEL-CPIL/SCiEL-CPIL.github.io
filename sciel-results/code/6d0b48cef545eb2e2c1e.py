"""Matrix multiplication 4,4,8 rank-minimal search with Strassen decomposition."""
import random
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11 -> contributes to C21
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1               # C21
    
    # M3 = A11(B12-B22) -> contributes to C12
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1               # C12
    
    # M4 = A22(B21-B11) -> contributes to C21
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[2, 3] = 1               # C21
    
    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[1, 4] = 1               # C12
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11
    
    return U, V, W

def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # 4x4 divided into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # High-level Strassen:
    # M11 = (A00+A11)(B00+B11)
    # M12 = (A10+A11)B00
    # M13 = A00(B01-B11)
    # M14 = A11(B10-B00)
    # M15 = (A00+A01)B11
    # M16 = (A10-A00)(B00+B01)
    # M17 = (A01-A11)(B10+B11)
    
    # C00 = M11 + M14 - M15 + M17
    # C01 = M13 + M15
    # C10 = M12 + M14
    # C11 = M11 - M12 + M13 + M16
    
    # For each high-level factor h (0-6), and each low-level factor l (0-6):
    # The U part is from the high-level A combination, scaled by low-level U
    # The V part is from the high-level B combination, scaled by low-level V
    # The W part is the high-level C placement, scaled by low-level W
    
    # High-level factor definitions: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping block (bi, bj) -> coefficient
    # V_coeffs: dict mapping block (bi, bj) -> coefficient
    # W_coeffs: dict mapping block (bi, bj) -> coefficient
    
    high_level_factors = [
        # M11 = (A00+A11)(B00+B11) -> C00, C11
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M12 = (A10+A11)B00 -> C10
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        # M13 = A00(B01-B11) -> C01
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1}},
        # M14 = A11(B10-B00) -> C00, C10
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # M15 = (A00+A01)B11 -> C00, C01
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M16 = (A10-A00)(B00+B01) -> C11
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M17 = (A01-A11)(B10+B11) -> C00
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    for h, hf in enumerate(high_level_factors):
        for l in range(7):
            factor_idx = h * 7 + l
            
            # U part: for each block in U_coeffs, add low-level U contributions
            for (bi, bj), coeff_u in hf['U'].items():
                # Block (bi, bj) of A: rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
                for li in range(2):
                    for lj in range(2):
                        a_row = bi * 2 + li
                        a_col = bj * 2 + lj
                        U_idx = a_row * 4 + a_col  # 4x4 index
                        U[U_idx, factor_idx] += coeff_u * U2[li * 2 + lj, l]
            
            # V part: for each block in V_coeffs, add low-level V contributions
            for (bi, bj), coeff_v in hf['V'].items():
                for li in range(2):
                    for lj in range(2):
                        b_row = bi * 2 + li
                        b_col = bj * 2 + lj
                        V_idx = b_row * 4 + b_col  # 4x4 index
                        V[V_idx, factor_idx] += coeff_v * V2[li * 2 + lj, l]
            
            # W part: for each block in W_coeffs, add low-level W contributions
            for (bi, bj), coeff_w in hf['W'].items():
                for li in range(2):
                    for lj in range(2):
                        c_row = bi * 2 + li
                        c_col = bj * 2 + lj
                        W_idx = c_row * 4 + c_col  # 4x4 index
                        W[W_idx, factor_idx] += coeff_w * W2[li * 2 + lj, l]
    
    return U, V, W

def _adapt_to_4x8(U4, V4, W4):
    """
    Adapt 4x4 Strassen factors to 4x8 multiplication.
    Input: U4: 16x49, V4: 16x49, W4: 16x49 (for 4x4 * 4x4 -> 4x4)
    Output: U: 16x?, V: 32x?, W: 32x? (for 4x4 * 4x8 -> 4x8)
    
    For 4x4 * 4x8:
    A is 4x4, B is 4x8, C is 4x8
    U: (n*m, rank) = (16, rank)
    V: (m*p, rank) = (32, rank)  
    W: (p*n, rank) = (32, rank)
    
    We can view B as two 4x4 matrices side by side: B = [B_left, B_right]
    C = [C_left, C_right]
    
    C_left = A * B_left (4x4 * 4x4 -> 4x4)
    C_right = A * B_right (4x4 * 4x4 -> 4x4)
    
    So we can use the same 4x4 decomposition for both halves.
    """
    rank4 = U4.shape[1]
    
    # U stays the same: 16 x rank4
    U = U4.copy()
    
    # V: need 32 x rank4*2
    # For B_left (cols 0-3 of B), use V4
    # For B_right (cols 4-7 of B), use V4
    V = np.zeros((32, rank4 * 2))
    V[:16, :rank4] = V4  # B_left part
    V[16:, rank4:] = V4  # B_right part
    
    # W: need 32 x rank4*2
    # W indices: c = k*n + i where k is 0-7 (p=8), i is 0-3 (n=4)
    # For C_left (k=0-3): W4
    # For C_right (k=4-7): W4
    W = np.zeros((32, rank4 * 2))
    W[:16, :rank4] = W4  # C_left part (k=0-3)
    W[16:, rank4:] = W4  # C_right part (k=4-7)
    
    return U.tolist(), V.tolist(), W.tolist()

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact for the 4x4 * 4x8 tensor."""
    rank = len(U[0]) if U else 0
    # T[a, b, c] should be 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check some zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_alt = (k + 1) % p * n + i
                if c_alt != c:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c_alt][r]
                    if abs(s) > 1e-9:
                        return False
    return True

def _construct_strassen_4x8():
    """Construct Strassen-based decomposition for 4x4 * 4x8."""
    U4, V4, W4 = _strassen_4x4_factors()
    U, V, W = _adapt_to_4x8(U4, V4, W4)
    return {"U": U, "V": V, "W": W}

def _try_random_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition via random search with local optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    start = time.time()
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            return None
        # Random initialization with entries from {0, 1, -1}
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(pn)]
        
        # Local search: try flipping entries
        for iteration in range(30):
            if time.time() > deadline - 10:
                break
            if _validate(U, V, W, n, m, p):
                return U, V, W
            # Pick a random entry to flip
            mat = random.choice([U, V, W])
            row = random.randrange(len(mat))
            col = random.randrange(r)
            mat[row][col] = -mat[row][col] if mat[row][col] != 0 else 1
            if time.time() > deadline - 10:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # First, try the Strassen-based decomposition (rank 98 for 4x8)
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        if _validate(U, V, W, n, m, p):
            return result
        else:
            # Strassen decomposition might have issues, fall back to search
            pass
    except Exception:
        pass
    
    # Try ranks from low to high using random search
    # Start from rank 98 (Strassen-based) and try to improve
    for r in range(98, n * m * p + 1):
        if time.time() > deadline - 15:
            break
        result = _try_random_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
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
