"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7 * (p // 2)  # 8 blocks of 2x2x2, each rank 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
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
    
    # We define the contributions of each M to U, V, W
    # U indices: i*m + j (i, j in 0..1)
    # V indices: j*p + k (j in 0..1, k in 0..1 for local block)
    # W indices: k*n + i (k in 0..1 for local block, i in 0..1)
    
    # For each block b, we have 7 terms
    # Let's define the local indices for A, B, C
    # A[i,j], B[j,k], C[k,i]
    
    # We will create a list of (u_idx, v_idx, w_idx, coeff) for each term
    # But it's easier to just iterate and assign.
    
    # Term definitions:
    # Each term t has:
    #   U contributions: list of (i, j, coeff)
    #   V contributions: list of (j, k, coeff)
    #   W contributions: list of (k, i, coeff)
    
    # M1
    t1 = {
        'U': [(0, 0, 1), (1, 1, 1)],
        'V': [(0, 0, 1), (1, 1, 1)],
        'W': [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] # Wait, C[k,i]
        # C00 gets M1, C11 gets M1
        # C00 is k=0, i=0. C11 is k=1, i=1.
        # So W contributions for M1: (0,0,1), (1,1,1)
    }
    t1['W'] = [(0, 0, 1), (1, 1, 1)]
    
    # M2
    t2 = {
        'U': [(1, 0, 1), (1, 1, 1)],
        'V': [(0, 0, 1)],
        'W': [(0, 1, 1), (1, 1, -1)] # C10 gets M2, C11 gets -M2
        # C10 is k=0, i=1. C11 is k=1, i=1.
    }
    
    # M3
    t3 = {
        'U': [(0, 0, 1)],
        'V': [(0, 1, 1), (1, 1, -1)],
        'W': [(1, 0, 1), (1, 1, 1)] # C01 gets M3, C11 gets M3
        # C01 is k=1, i=0. C11 is k=1, i=1.
    }
    
    # M4
    t4 = {
        'U': [(1, 1, 1)],
        'V': [(1, 0, 1), (0, 0, -1)],
        'W': [(0, 0, 1), (0, 1, 1)] # C00 gets M4, C10 gets M4
        # C00 is k=0, i=0. C10 is k=0, i=1.
    }
    
    # M5
    t5 = {
        'U': [(0, 0, 1), (0, 1, 1)],
        'V': [(1, 1, 1)],
        'W': [(0, 0, -1), (1, 0, 1)] # C00 gets -M5, C01 gets M5
        # C00 is k=0, i=0. C01 is k=1, i=0.
    }
    
    # M6
    t6 = {
        'U': [(1, 0, 1), (0, 0, -1)],
        'V': [(0, 0, 1), (0, 1, 1)],
        'W': [(1, 1, 1)] # C11 gets M6
    }
    
    # M7
    t7 = {
        'U': [(0, 1, 1), (1, 1, -1)],
        'V': [(1, 0, 1), (1, 1, 1)],
        'W': [(0, 0, 1)] # C00 gets M7
    }
    
    terms = [t1, t2, t3, t4, t5, t6, t7]
    
    for b in range(p // 2):
        base_col = b * 7
        for t_idx in range(7):
            col = base_col + t_idx
            t = terms[t_idx]
            
            for (i, j, coeff) in t['U']:
                U[i * m + j][col] += coeff
                
            for (j, k, coeff) in t['V']:
                # k is local 0..1, global is 2*b + k
                V[j * p + (2 * b + k)][col] += coeff
                
            for (k, i, coeff) in t['W']:
                # k is local 0..1, global is 2*b + k
                W[(2 * b + k) * n + i][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
