"""Strassen-Winograd block decomposition for 4x5 by 5x6 matrix multiplication."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _build_2x2_strassen():
    """Return list of (alpha_list, beta_list, gamma) for 2x2x2 Strassen.
    alpha_list[i] = coefficient for A[i][j], beta_list[k] = coefficient for B[j][k],
    gamma = coefficient for C[i][k]."""
    # M1=(A11+A22)(B11+B22): C11 +=1, C22 -=1
    # M2=(A21+A22)B11: C21 +=1, C22 -=1
    # M3=A11(B21-B11): C12 +=1, C21 +=1
    # M4=A22(B12-B22): C11 +=1, C21 +=1
    # M5=(A11+A12)B22: C11 -=1, C12 +=1
    # M6=(A21-A11)(B11+B12): C22 +=1
    # M7=(A11+A12)(B21+B22): C11 +=1, C22 +=1
    return [
        # (alpha[4], beta[4], gamma[4])
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, -1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 1, 0]),   # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]),    # M7
    ]

def _build_core_4x4x4():
    """Build the 28 rank terms for 4x4x4 via 2x2 Strassen recursion.
    Each term: (alpha[16], beta[16], gamma[16]) where alpha[i*4+j] for A[i][j],
    beta[j*4+k] for B[j][k], gamma[k*4+i] for C[i][k] (trace convention C[k*n+i])."""
    strassen = _build_2x2_strassen()
    terms = []
    for bp in range(4):  # block position (0,1,2,3)
        bi, bj = bp // 2, bp % 2  # which 2x2 block
        for (a2, b2, g2) in strassen:
            alpha = [0] * 16
            beta = [0] * 16
            gamma = [0] * 16
            for r in range(4):
                if a2[r] != 0:
                    i = 2 * bi + (r // 2)
                    j = 2 * bj + (r % 2)
                    alpha[i * 4 + j] = a2[r]
            for r in range(4):
                if b2[r] != 0:
                    j = 2 * bj + (r // 2)
                    k = 2 * bj + (r % 2)
                    beta[j * 4 + k] = b2[r]
            for r in range(4):
                if g2[r] != 0:
                    i = 2 * bi + (r // 2)
                    k = 2 * bj + (r % 2)
                    gamma[k * 4 + i] = g2[r]  # C[k*n+i] with n=4
            terms.append((alpha, beta, gamma))
    return terms

def _build_tail_terms():
    """Build the 4 rank terms for A1 (4x1) * B1 (1x6).
    A1[i][0] for i in 0..3, B1[0][k] for k in 0..5.
    Each term: A[i][4] * B[4][k] contributes to C[i][k].
    alpha[i*5+4]=1, beta[4*6+k]=1, gamma[k*4+i]=1."""
    terms = []
    for i in range(4):
        alpha = [0] * 20  # 4x5
        beta = [0] * 30   # 5x6
        gamma = [0] * 24  # 6x4
        alpha[i * 5 + 4] = 1
        for k in range(6):
            beta[4 * 6 + k] = 1
            gamma[k * 4 + i] = 1
        terms.append((alpha, beta, gamma))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    # All rank terms
    core_terms = _build_core_4x4x4()
    tail_terms = _build_tail_terms()
    all_terms = core_terms + tail_terms
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    for c, (alpha, beta, gamma) in enumerate(all_terms):
        for i in range(n * m):
            U[i][c] = alpha[i]
        for j in range(m * p):
            V[j][c] = beta[j]
        for k in range(p * n):
            W[k][c] = gamma[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
