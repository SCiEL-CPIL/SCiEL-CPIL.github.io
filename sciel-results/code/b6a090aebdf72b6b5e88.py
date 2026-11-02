import time
import numpy as np

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2x16():
    n, m, p = 2, 2, 16
    num_blocks = p // 2 # 8 blocks of 2 columns
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)] # 4 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Strassen formulas for A(2x2) * B(2x2) -> C(2x2)
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices (local): 00, 01, 10, 11
    # C indices (local): 00, 01, 10, 11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for U (A), V (B), W (C) for each of the 7 terms
    # Term 1: M1
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1)
    
    # Term 2: M2
    # U: A10(1), A11(1)
    # V: B00(1)
    # W: C10(1), C11(-1)
    
    # Term 3: M3
    # U: A00(1)
    # V: B01(1), B11(-1)
    # W: C01(1), C11(1)
    
    # Term 4: M4
    # U: A11(1)
    # V: B10(1), B00(-1)
    # W: C00(1), C10(1)
    
    # Term 5: M5
    # U: A00(1), A01(1)
    # V: B11(1)
    # W: C00(-1), C01(1)
    
    # Term 6: M6
    # U: A10(1), A00(-1)
    # V: B00(1), B01(1)
    # W: C11(1)
    
    # Term 7: M7
    # U: A01(1), A11(-1)
    # V: B10(1), B11(1)
    # W: C00(1)
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (idx, coeff), V: (idx, coeff), W: (idx, coeff)
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)], # Local B indices
            'W': [(0, 1), (3, 1)]  # Local C indices
        },
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]
    
    col = 0
    for t in range(num_blocks):
        # Map local indices to global indices
        # Local B: 00->2t, 01->2t+1, 10->16+2t, 11->16+2t+1
        # Local C: 00->2t, 01->2t+1, 10->16+2t, 11->16+2t+1
        
        b_map = {0: 2*t, 1: 2*t+1, 2: 16+2*t, 3: 16+2*t+1}
        c_map = {0: 2*t, 1: 2*t+1, 2: 16+2*t, 3: 16+2*t+1}
        
        for term in terms:
            for idx, coeff in term['U']:
                U[idx][col] += coeff
            for idx, coeff in term['V']:
                V[b_map[idx]][col] += coeff
            for idx, coeff in term['W']:
                W[c_map[idx]][col] += coeff
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 2, 2, 16
    start_time = time.time()
    
    # Try Strassen (Rank 56)
    try:
        U, V, W = _strassen_2x2x16()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to Schoolbook (Rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
