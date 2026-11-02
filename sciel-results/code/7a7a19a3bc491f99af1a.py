"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Block Strassen (27 blocks * 7 Strassen ranks)
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrices
    # Each factor is (U_entries, V_entries, W_entries)
    # U_entries: list of (local_row, local_col, val)
    # V_entries: list of (local_row, local_col, val)
    # W_entries: list of (local_row, local_col, val)
    
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11)
        {
            'U': [((0,0),1), ((1,1),1)],
            'V': [((0,0),1), ((1,1),1)],
            'W': [((0,0),1), ((0,1),1), ((1,0),1), ((1,1),1)] # Wait, W is for C.
            # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
            # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
            # So W entries for M1 are: C00:1, C11:1. C01:0, C10:0.
        },
        # M2 = (A10+A11)B00
        {
            'U': [((1,0),1), ((1,1),1)],
            'V': [((0,0),1)],
            'W': [((1,0),1), ((1,1),1)] # C10 += M2, C11 += -M2.
            # Correction: W10 += M2, W11 -= M2.
        },
        # M3 = A00(B01-B11)
        {
            'U': [((0,0),1)],
            'V': [((0,1),1), ((1,1),-1)],
            'W': [((0,1),1), ((1,1),1)] # C01 += M3, C11 += M3.
        },
        # M4 = A11(B10-B00)
        {
            'U': [((1,1),1)],
            'V': [((1,0),1), ((0,0),-1)],
            'W': [((0,0),1), ((1,0),1)] # C00 += M4, C10 += M4.
        },
        # M5 = (A00+A01)B11
        {
            'U': [((0,0),1), ((0,1),1)],
            'V': [((1,1),1)],
            'W': [((0,0),-1), ((0,1),1)] # C00 -= M5, C01 += M5.
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'U': [((1,0),1), ((0,0),-1)],
            'V': [((0,0),1), ((0,1),1)],
            'W': [((1,1),1)] # C11 += M6.
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'U': [((0,1),1), ((1,1),-1)],
            'V': [((1,0),1), ((1,1),1)],
            'W': [((0,0),1), ((0,1),1)] # C00 += M7, C01 += M7?
            # C00 += M7. C01? No.
            # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
            # M7 only contributes to C00.
        }
    ]
    
    # Let's re-verify the W contributions carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: W00 += 1, W11 += 1
    # M2: W10 += 1, W11 -= 1
    # M3: W01 += 1, W11 += 1
    # M4: W00 += 1, W10 += 1
    # M5: W00 -= 1, W01 += 1
    # M6: W11 += 1
    # M7: W00 += 1
    
    # My previous W definitions in the dict were messy. I will hardcode the W contributions directly in the loop for clarity and correctness.
    
    # Define Strassen U, V, W contributions
    # U: (x,y) -> val
    # V: (y,z) -> val
    # W: (x,z) -> val
    
    s_U = [
        { (0,0):1, (1,1):1 }, # M1
        { (1,0):1, (1,1):1 }, # M2
        { (0,0):1 },           # M3
        { (1,1):1 },           # M4
        { (0,0):1, (0,1):1 }, # M5
        { (1,0):1, (0,0):-1 },# M6
        { (0,1):1, (1,1):-1 } # M7
    ]
    
    s_V = [
        { (0,0):1, (1,1):1 }, # M1
        { (0,0):1 },           # M2
        { (0,1):1, (1,1):-1 },# M3
        { (1,0):1, (0,0):-1 },# M4
        { (1,1):1 },           # M5
        { (0,0):1, (0,1):1 }, # M6
        { (1,0):1, (1,1):1 }  # M7
    ]
    
    s_W = [
        { (0,0):1, (1,1):1 }, # M1
        { (1,0):1, (1,1):-1 },# M2
        { (0,1):1, (1,1):1 }, # M3
        { (0,0):1, (1,0):1 }, # M4
        { (0,0):-1, (0,1):1 },# M5
        { (1,1):1 },           # M6
        { (0,0):1 }            # M7
    ]

    col = 0
    for a in range(3):
        for c in range(3):
            for b in range(3):
                for s in range(7):
                    # Update U
                    for (x, y), val in s_U[s].items():
                        i = 2*a + x
                        j = 2*c + y
                        u_idx = i * m + j
                        U[u_idx][col] += val
                    
                    # Update V
                    for (y, z), val in s_V[s].items():
                        j = 2*c + y
                        k = 2*b + z
                        v_idx = j * p + k
                        V[v_idx][col] += val
                    
                    # Update W
                    for (x, z), val in s_W[s].items():
                        i = 2*a + x
                        k = 2*b + z
                        w_idx = k * n + i
                        W[w_idx][col] += val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
