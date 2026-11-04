"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block-Strassen: partition 6 = 3*2, so 3x3 blocks of 2x2
    # For each (i0, j0, k0) in {0,1,2}^3, apply Strassen to 2x2 sub-matrices
    # Total rank = 27 * 7 = 189
    
    U_terms = []  # list of (row_index, coeff) per term
    V_terms = []
    W_terms = []
    
    # Strassen 7 products for 2x2 matrices
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b00-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term: U has entries for A, V has entries for B, W has entries for C
    # A index: row = i, col = k -> flat index i*m + k = i*6 + k
    # B index: row = k, col = j -> flat index k*p + j = k*6 + j
    # C index: row = i, col = j -> but W uses trace convention W[k*n+i, r]
    # Wait, the problem says W[k*n+i, r] which is unusual. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) with flat index k*n + i, where k is the "column" index of B (which is row of C) and i is row of A (which is col of C)?
    # Actually for matrix mult C = AB, C[i,j] = sum_k A[i,k] B[k,j]
    # U has shape (n*m, rank) = (36, rank), indexed by (i, k) -> i*m + k
    # V has shape (m*p, rank) = (36, rank), indexed by (k, j) -> k*p + j
    # W has shape (p*n, rank) = (36, rank), indexed by (j, i) -> j*n + i (trace convention)
    # So W[j*n + i, r] corresponds to C[i,j]
    
    # Let me define the Strassen terms for each (i0, j0, k0)
    # i = i0*2 + i1, k = k0*2 + k1, j = j0*2 + j1
    # where i1, k1, j1 in {0, 1}
    
    strassen_A = [
        # Each is a list of (i1, k1, coeff)
        [(0,0,1), (1,1,1)],   # M1: a00 + a11
        [(1,0,1), (1,1,1)],   # M2: a10 + a11
        [(0,0,1)],             # M3: a00
        [(1,1,1)],             # M4: a11
        [(0,0,1), (0,1,1)],   # M5: a00 + a01
        [(1,0,1), (0,0,-1)],  # M6: a10 - a00
        [(0,1,1), (1,1,-1)],  # M7: a01 - a11
    ]
    
    strassen_B = [
        [(0,0,1), (1,1,1)],   # M1: b00 + b11
        [(0,0,1)],             # M2: b00
        [(1,0,1), (1,1,-1)],  # M3: b10 - b11
        [(0,0,1), (0,1,-1)],  # M4: b00 - b01
        [(1,1,1)],             # M5: b11
        [(0,0,1), (0,1,1)],   # M6: b00 + b01
        [(1,0,1), (1,1,1)],   # M7: b10 + b11
    ]
    
    strassen_C = [
        # Each is a list of (i1, j1, coeff)
        [(0,0,1), (0,1,0), (1,0,0), (1,1,0)],  # placeholder, fill below
    ]
    
    # C contributions per Strassen product:
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    strassen_C = [
        [(0,0,1)],   # M1 -> C00
        [(1,0,1)],   # M2 -> C10
        [(0,1,1)],   # M3 -> C01
        [(0,0,1)],   # M4 -> C00
        [(0,1,1)],   # M5 -> C01
        [(1,1,1)],   # M6 -> C11
        [(0,0,1)],   # M7 -> C00
    ]
    
    # Wait, I need to be more careful. Let me redo this.
    # For each Strassen product s (0-6), it contributes to C entries with certain coefficients.
    # Let me define C_contrib[s] = list of (i1, j1, coeff)
    
    C_contrib = [
        [(0,0,1)],           # M1 -> C00
        [(1,0,1)],           # M2 -> C10
        [(0,1,1)],           # M3 -> C01
        [(0,0,1)],           # M4 -> C00
        [(0,1,1)],           # M5 -> C01
        [(1,1,1)],           # M6 -> C11
        [(0,0,1)],           # M7 -> C00
    ]
    
    # But we also need negative coefficients and multiple C entries per product.
    # Let me redefine properly:
    # M1 contributes: C00 += 1, C11 += 1
    # M2 contributes: C10 += 1, C11 -= 1
    # M3 contributes: C01 += 1, C11 += 1
    # M4 contributes: C00 += 1, C10 += 1
    # M5 contributes: C00 -= 1, C01 += 1
    # M6 contributes: C11 += 1
    # M7 contributes: C00 += 1
    
    C_contrib = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1)],                # M7
    ]
    
    U_rows = n * m  # 36
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    rank = 27 * 7   # 189
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    for i0 in range(3):
        for j0 in range(3):
            for k0 in range(3):
                for s in range(7):
                    # A part: strassen_A[s]
                    for (i1, k1, coeff) in strassen_A[s]:
                        i = i0 * 2 + i1
                        k = k0 * 2 + k1
                        U[i * m + k][col] = coeff
                    
                    # B part: strassen_B[s]
                    for (k1, j1, coeff) in strassen_B[s]:
                        k = k0 * 2 + k1
                        j = j0 * 2 + j1
                        V[k * p + j][col] = coeff
                    
                    # C part: C_contrib[s]
                    for (i1, j1, coeff) in C_contrib[s]:
                        i = i0 * 2 + i1
                        j = j0 * 2 + j1
                        # W trace convention: W[k*n+i, r] where k is col of B / row of C = j, i is row of A / col of C = i
                        # So W[j*n + i, col]
                        W[j * n + i][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
