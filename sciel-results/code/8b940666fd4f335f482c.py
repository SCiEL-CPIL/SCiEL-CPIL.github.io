"""
Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Approach:
- The 2x4 * 4x8 product has 2*4*8 = 64 scalar multiplications (schoolbook).
- We split the 4 intermediate indices j into two groups of 2.
- We split the 8 output indices k into four groups of 2.
- For each (j-group, k-group) pair, we have a 2x2 * 2x2 sub-multiplication.
- We use Strassen's rank-7 algorithm for each 2x2 * 2x2 sub-problem.
- Total rank = 2 * 4 * 7 = 56.

The factor matrices U, V, W are built so that:
  C[i,k] = sum_j A[i,j] * B[j,k]
is exactly reproduced via the tensor decomposition.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen's 7 intermediate products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen coefficients:
    # For each of 7 terms, list of (A_idx, coeff) and (B_idx, coeff)
    # A_idx: 0=a, 1=b, 2=c, 3=d
    # B_idx: 0=e, 1=f, 2=g, 3=h
    
    strassen_A = [
        {0: 1, 3: 1},   # M1: a+d
        {2: 1, 3: 1},   # M2: c+d
        {0: 1},         # M3: a
        {3: 1},         # M4: d
        {0: 1, 1: 1},   # M5: a+b
        {2: 1, 0: -1},  # M6: c-a
        {1: 1, 3: -1},  # M7: b-d
    ]
    strassen_B = [
        {0: 1, 3: 1},   # M1: e+h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f-h
        {2: 1, 0: -1},  # M4: g-e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e+f
        {2: 1, 3: 1},   # M7: g+h
    ]
    # C combination: C_local[i][k] = sum_s coeff[s] * M_s
    # C11 = M1 + M4 - M5 + M7 -> coeffs: M1:1, M4:1, M5:-1, M7:1
    # C12 = M3 + M5 -> coeffs: M3:1, M5:1
    # C21 = M2 + M4 -> coeffs: M2:1, M4:1
    # C22 = M1 - M2 + M3 + M6 -> coeffs: M1:1, M2:-1, M3:1, M6:1
    
    strassen_C = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C11: M1+M4-M5+M7 (indices 0,3,4,6)
        {2: 1, 4: 1},                # C12: M3+M5 (indices 2,4)
        {1: 1, 3: 1},                # C21: M2+M4 (indices 1,3)
        {0: 1, 1: -1, 2: 1, 5: 1},  # C22: M1-M2+M3+M6 (indices 0,1,2,5)
    ]
    
    # Total rank
    rank = 2 * 4 * 7  # 2 j-groups * 4 k-groups * 7 Strassen terms = 56
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # n*m = 8
    V = [[0] * rank for _ in range(m * p)]  # m*p = 32
    W = [[0] * rank for _ in range(p * n)]  # p*n = 16
    
    # For each (jg, kg, s) where jg in {0,1}, kg in {0,1,2,3}, s in {0,...,6}
    # The rank index is: c = (jg * 4 + kg) * 7 + s
    
    for jg in range(2):
        for kg in range(4):
            for s in range(7):
                c = (jg * 4 + kg) * 7 + s
                
                # Fill U: for each i in {0,1}, j_local in {0,1}
                # A[i, jg*2 + j_local] has index i*m + (jg*2 + j_local) = i*4 + jg*2 + j_local
                for i in range(2):
                    for j_local in range(2):
                        # A_idx for Strassen: i*2 + j_local (0=a,1=b,2=c,3=d)
                        a_idx = i * 2 + j_local
                        if a_idx in strassen_A[s]:
                            U[i * m + jg * 2 + j_local][c] += strassen_A[s][a_idx]
                
                # Fill V: for each j_local in {0,1}, k_local in {0,1}
                # B[jg*2 + j_local, kg*2 + k_local] has index (jg*2+j_local)*p + (kg*2+k_local)
                for j_local in range(2):
                    for k_local in range(2):
                        # B_idx for Strassen: j_local*2 + k_local (0=e,1=f,2=g,3=h)
                        b_idx = j_local * 2 + k_local
                        if b_idx in strassen_B[s]:
                            V[(jg * 2 + j_local) * p + kg * 2 + k_local][c] += strassen_B[s][b_idx]
                
                # Fill W: for each i in {0,1}, k_local in {0,1}
                # C[i, kg*2 + k_local] has index (kg*2+k_local)*n + i = (kg*2+k_local)*2 + i
                for i in range(2):
                    for k_local in range(2):
                        # C_idx for Strassen: i*2 + k_local (0=C11,1=C12,2=C21,3=C22)
                        c_idx = i * 2 + k_local
                        if s in strassen_C[c_idx]:
                            W[(kg * 2 + k_local) * n + i][c] += strassen_C[c_idx][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
