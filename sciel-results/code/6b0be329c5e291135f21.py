"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 using Strassen for 5 blocks of 2x2x2
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    strassen = [
        {
            'A': [1, 0, 0, 1],
            'B': [1, 0, 0, 1],
            'C': [1, 0, 0, 1]
        },
        # M2: (A10+A11)B00 -> C10+=1
        {
            'A': [0, 0, 1, 1],
            'B': [1, 0, 0, 0],
            'C': [0, 0, 1, 0]
        },
        # M3: A00(B01-B11) -> C01+=1
        {
            'A': [1, 0, 0, 0],
            'B': [0, 1, 0, -1],
            'C': [0, 1, 0, 0]
        },
        # M4: A11(B10-B00) -> C00+=1, C10+=1
        {
            'A': [0, 0, 0, 1],
            'B': [0, 0, 1, -1],
            'C': [1, 0, 1, 0]
        },
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        {
            'A': [1, 1, 0, 0],
            'B': [0, 0, 0, 1],
            'C': [-1, 1, 0, 0]
        },
        # M6: (A10-A00)(B00+B01) -> C11+=1
        {
            'A': [0, 0, 1, -1],
            'B': [1, 1, 0, 0],
            'C': [0, 0, 0, 1]
        },
        # M7: (A01-A11)(B10+B11) -> C00+=1, C11+=1
        {
            'A': [0, 1, 0, -1],
            'B': [0, 0, 1, 1],
            'C': [1, 0, 0, 1]
        }
    ]
    
    for t in range(5):
        for s in range(7):
            c = t * 7 + s
            params = strassen[s]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_A = i * m + j
                    U[idx_A][c] = params['A'][i * 2 + j]
            
            # Fill V
            for j in range(2):
                for r in range(2):
                    k = t * 2 + r
                    idx_B = j * p + k
                    V[idx_B][c] = params['B'][j * 2 + r]
            
            # Fill W
            for r in range(2):
                k = t * 2 + r
                for i in range(2):
                    idx_C = k * n + i
                    W[idx_C][c] = params['C'][i * 2 + r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
