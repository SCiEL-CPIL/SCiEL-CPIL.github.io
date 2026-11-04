"""Matrix multiplication 3x4x8 rank-minimal decomposition using block Strassen."""
DIMENSIONS = [3, 4, 8]
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120.0  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2x2(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1, ri0, ri1):
        # M1 = (P00+P11)*(Q00+Q11) -> contributes to R00 and R11
        add_term(
            {(pi0*m+pj0): 1, (pi1*m+pj1): 1},
            {(qj0*p+qk0): 1, (qj1*p+qk1): 1},
            {(qk0*n+pi0): 1, (qk1*n+pi1): 1}
        )
        # M2 = (P10+P11)*Q00 -> R10
        add_term(
            {(pi1*m+pj0): 1, (pi1*m+pj1): 1},
            {(qj0*p+qk0): 1},
            {(qk1*n+pi0): 1}
        )
        # M3 = P00*(Q01-Q11) -> R01
        add_term(
            {(pi0*m+pj0): 1},
            {(qj0*p+qk1): 1, (qj1*p+qk1): -1},
            {(qk0*n+pi1): 1}
        )
        # M4 = P11*(Q10-Q00) -> R00 and R10
        add_term(
            {(pi1*m+pj1): 1},
            {(qj1*p+qk0): 1, (qj0*p+qk0): -1},
            {(qk0*n+pi0): 1, (qk1*n+pi0): 1}
        )
        # M5 = (P00+P01)*Q11 -> -R00 and +R01
        add_term(
            {(pi0*m+pj0): 1, (pi0*m+pj1): 1},
            {(qj1*p+qk1): 1},
            {(qk0*n+pi0): -1, (qk0*n+pi1): 1}
        )
        # M6 = (P10-P00)*(Q00+Q01) -> R11
        add_term(
            {(pi1*m+pj0): 1, (pi0*m+pj0): -1},
            {(qj0*p+qk0): 1, (qj0*p+qk1): 1},
            {(qk1*n+pi1): 1}
        )
        # M7 = (P01-P11)*(Q10+Q11) -> R00
        add_term(
            {(pi0*m+pj1): 1, (pi1*m+pj1): -1},
            {(qj1*p+qk0): 1, (qj1*p+qk1): 1},
            {(qk0*n+pi0): 1}
        )
    
    def mul_2x4x2(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1):
        # A is 2x4 (rows pi0,pi1; cols pj0..pj1), B is 4x2 (rows qj0..qj1; cols qk0,qk1)
        # Split A into left 2 cols and right 2 cols
        strassen_2x2x2(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1, pi0, pi1)
        strassen_2x2x2(pi0, pi1, pj0+2, pj1+2, qj0+2, qj1+2, qk0, qk1, pi0, pi1)
    
    def mul_2x4x4(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1):
        # A is 2x4, B is 4x4, C is 2x4
        # Split B into left 2 cols and right 2 cols
        mul_2x4x2(pi0, pi1, pj0, pj1, qj0, qj1, qk0, qk1)
        mul_2x4x2(pi0, pi1, pj0, pj1, qj0, qj1, qk0+2, qk1+2)
    
    def mul_1x4x4(pi, pj0, pj1, qj0, qj1, qk0, qk1):
        # A is 1x4, B is 4x4, C is 1x4 (4 outputs)
        for k in range(4):
            u = {}
            v = {}
            for j in range(4):
                u[pi*m + (pj0+j)] = 1
                v[(qj0+j)*p + (qk0+k)] = 1
            w = {(qk0+k)*n + pi: 1}
            add_term(u, v, w)
    
    # Build full 3x4x8:
    # A rows 0,1 (A1: 2x4) and row 2 (A2: 1x4)
    # B cols 0-3 (B1: 4x4) and cols 4-7 (B2: 4x4)
    
    mul_2x4x4(0, 1, 0, 3, 0, 3, 0, 3)   # A1 x B1 -> C[0:4, 0:2]
    mul_2x4x4(0, 1, 0, 3, 0, 3, 4, 7)   # A1 x B2 -> C[4:8, 0:2]
    mul_1x4x4(2, 0, 3, 0, 3, 0, 3)       # A2 x B1 -> C[0:4, 2]
    mul_1x4x4(2, 0, 3, 0, 3, 4, 7)       # A2 x B2 -> C[4:8, 2]
    
    rank = len(terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][c] += val
        for idx, val in v.items():
            V[idx][c] += val
        for idx, val in w.items():
            W[idx][c] += val
    
    return {"U": U, "V": V, "W": W}
