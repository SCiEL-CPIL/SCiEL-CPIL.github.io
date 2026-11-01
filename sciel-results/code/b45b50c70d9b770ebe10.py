"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2[r, c] where r is index in A (0..3), c is mode (0..6)
    # A = [[a00, a01], [a10, a11]] -> indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B = [[b00, b01], [b10, b11]] -> indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C = [[c00, c01], [c10, c11]] -> indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    # M2 = (a10+a11)b00 -> C10 += 1
    # M3 = a00(b01-b11) -> C01 += 1, C11 -= 1
    # M4 = a11(b10-b00) -> C10 += 1, C11 -= 1
    # M5 = (a00+a01)b11 -> C01 += 1, C11 += 1
    # M6 = (a10-a00)(b00+b01) -> C10 += 1, C00 -= 1
    # M7 = (a01-a11)(b10+b11) -> C00 += 1, C11 -= 1
    
    # Let's verify C coefficients:
    # C00 = M1 + M4 - M5 + M7? 
    # Strassen standard:
    # C00 = M1 - M4 + M6 - M7 ? 
    # Let's use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # Let's check U2, V2, W2 based on this.
    # U2 (A part)
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, -A00
    # M7: A01, -A11
    
    # V2 (B part)
    # M1: B00, B11
    # M2: B00
    # M3: B01, -B11
    # M4: B10, -B00
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    
    # W2 (C part)
    # C00: M1(1), M4(-1), M6(1), M7(-1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M5(1)
    
    U2 = {
        0: {0: 1, 3: 1},   # M1: A00, A11
        1: {2: 1, 3: 1},   # M2: A10, A11
        2: {0: 1},         # M3: A00
        3: {3: 1},         # M4: A11
        4: {0: 1, 1: 1},   # M5: A00, A01
        5: {2: 1, 0: -1},  # M6: A10, -A00
        6: {1: 1, 3: -1}   # M7: A01, -A11
    }
    
    V2 = {
        0: {0: 1, 3: 1},   # M1: B00, B11
        1: {0: 1},         # M2: B00
        2: {1: 1, 3: -1},  # M3: B01, -B11
        3: {2: 1, 0: -1},  # M4: B10, -B00
        4: {3: 1},         # M5: B11
        5: {0: 1, 1: 1},   # M6: B00, B01
        6: {2: 1, 3: 1}    # M7: B10, B11
    }
    
    W2 = {
        0: {0: 1, 3: -1, 5: 1, 6: -1},  # C00: M1, -M4, M6, -M7
        1: {2: 1, 4: 1},                # C01: M3, M5
        2: {1: 1, 3: 1},                # C10: M2, M4
        3: {0: 1, 1: -1, 2: 1, 4: 1}    # C11: M1, -M2, M3, M5
    }

    # Build 4x4 factors
    # Rank is 7 * 7 = 49
    # U4, V4, W4 will be lists of lists, or we can build sparse and convert
    # Shape: (16, 49)
    
    U4_sparse = defaultdict(lambda: defaultdict(int))
    V4_sparse = defaultdict(lambda: defaultdict(int))
    W4_sparse = defaultdict(lambda: defaultdict(int))
    
    for c1 in range(7):
        for c0 in range(7):
            c = 7 * c1 + c0
            
            # U4 column c
            for r1, v1 in U2[c1].items():
                for r0, v0 in U2[c0].items():
                    # r1, r0 are 2x2 indices (0..3)
                    # Decompose r1, r0 into i1, j1 and i0, j0
                    i1 = r1 // 2
                    j1 = r1 % 2
                    i0 = r0 // 2
                    j0 = r0 % 2
                    
                    # Standard U row index: i * m + j = i * 4 + j
                    # i = 2*i1 + i0
                    # j = 2*j1 + j0
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    row_u = i * 4 + j
                    val = v1 * v0
                    U4_sparse[row_u][c] += val
                    
            # V4 column c
            for r1, v1 in V2[c1].items():
                for r0, v0 in V2[c0].items():
                    j1 = r1 // 2
                    k1 = r1 % 2
                    j0 = r0 // 2
                    k0 = r0 % 2
                    
                    # Standard V row index: j * p + k = j * 4 + k
                    # j = 2*j1 + j0
                    # k = 2*k1 + k0
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    row_v = j * 4 + k
                    val = v1 * v0
                    V4_sparse[row_v][c] += val
                    
            # W4 column c
            for r1, v1 in W2[c1].items():
                for r0, v0 in W2[c0].items():
                    k1 = r1 // 2
                    i1 = r1 % 2
                    k0 = r0 // 2
                    i0 = r0 % 2
                    
                    # Standard W row index: k * n + i = k * 4 + i
                    # k = 2*k1 + k0
                    # i = 2*i1 + i0
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    row_w = k * 4 + i
                    val = v1 * v0
                    W4_sparse[row_w][c] += val

    # Convert sparse to dense
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for row in U4_sparse:
        for col, val in U4_sparse[row].items():
            U[row][col] = val
            
    for row in V4_sparse:
        for col, val in V4_sparse[row].items():
            V[row][col] = val
            
    for row in W4_sparse:
        for col, val in W4_sparse[row].items():
            W[row][col] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
