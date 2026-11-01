import time
import sys

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm for 2x2 * 2x2 has rank 7.
    # The 2x14 matrix is 7 blocks of 2x2.
    # Total rank = 7 blocks * 7 terms = 49.
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: A[i,j], B[j,k], C[i,k]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for U (A part), V (B part), W (C part)
    # For each term s in 0..6, and block b in 0..6
    # col = b * 7 + s
    
    # A coefficients: [ (A00 coeff, A11 coeff), (A10 coeff, A11 coeff) ... ]
    # Actually, let's just list the non-zero entries for U, V, W for each term.
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    # Term 1 (M2): (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1, C11=-1
    # Term 2 (M3): A00(B01-B11)
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01=1, C11=1
    # Term 3 (M4): A11(B10-B00)
    # U: A11=1
    # V: B10=1, B00=-1
    # W: C00=1, C10=1
    # Term 4 (M5): (A00+A01)B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00=-1, C01=1
    # Term 5 (M6): (A10-A00)(B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1
    # Term 6 (M7): (A01-A11)(B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1
    
    # We store these as lists of (index, value) for U, V, W
    # U index: i*m + j
    # V index: j*p + k (local k in 0..1, global k = 2*b + k_local)
    # W index: k*n + i (local k in 0..1, global k = 2*b + k_local)
    
    terms = [
        # (U_entries, V_entries, W_entries)
        # U: (i,j, val), V: (j, k_local, val), W: (k_local, i, val)
        {
            'U': [(0, 1), (3, 1)],       # A00, A11
            'V': [(0, 0, 1), (1, 1, 1)], # B00, B11
            'W': [(0, 0, 1), (1, 1, 1)]  # C00, C11
        },
        {
            'U': [(2, 1), (3, 1)],       # A10, A11
            'V': [(0, 0, 1)],            # B00
            'W': [(0, 1, 1), (1, 1, -1)] # C10, C11
        },
        {
            'U': [(0, 1)],               # A00
            'V': [(0, 1, 1), (1, 1, -1)],# B01, B11
            'W': [(1, 0, 1), (1, 1, 1)]  # C01, C11
        },
        {
            'U': [(3, 1)],               # A11
            'V': [(1, 0, 1), (0, 0, -1)],# B10, B00
            'W': [(0, 0, 1), (0, 1, 1)]  # C00, C10
        },
        {
            'U': [(0, 1), (1, 1)],       # A00, A01
            'V': [(1, 1, 1)],            # B11
            'W': [(0, 0, -1), (1, 0, 1)] # C00, C01
        },
        {
            'U': [(2, 1), (0, -1)],      # A10, A00
            'V': [(0, 0, 1), (0, 1, 1)], # B00, B01
            'W': [(1, 1, 1)]             # C11
        },
        {
            'U': [(1, 1), (3, -1)],      # A01, A11
            'V': [(1, 0, 1), (1, 1, 1)], # B10, B11
            'W': [(0, 0, 1)]             # C00
        }
    ]
    
    for b in range(7):
        for s in range(7):
            col = b * 7 + s
            t = terms[s]
            
            # Fill U
            for i, j, val in t['U']:
                U[i * m + j][col] = val
            
            # Fill V
            for j, k_local, val in t['V']:
                k_global = 2 * b + k_local
                V[j * p + k_global][col] = val
            
            # Fill W
            for k_local, i, val in t['W']:
                k_global = 2 * b + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
