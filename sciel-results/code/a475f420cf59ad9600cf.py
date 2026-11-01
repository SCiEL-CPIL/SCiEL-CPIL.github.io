import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank 42 using Strassen on 6 blocks of 2x2
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # P00 = M1 - M4 + M5 - M7
    # P01 = M3 + M4
    # P10 = M2 + M7
    # P11 = M1 - M2 + M3 + M6
    
    # U indices: i*2 + j
    # V indices: j*12 + k
    # W indices: k*2 + i
    
    # Coefficients for U (A part)
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # M1: a=1, d=1
    # M2: c=1, d=1
    # M3: a=1
    # M4: d=1
    # M5: a=1, b=1
    # M6: c=1, a=-1
    # M7: b=1, d=-1
    
    # Coefficients for V (B part)
    # e=V[0*12+2l], f=V[0*12+2l+1]
    # g=V[1*12+2l], h=V[1*12+2l+1]
    # M1: e=1, h=1
    # M2: e=1
    # M3: f=1, h=-1
    # M4: g=1, e=-1
    # M5: h=1
    # M6: e=1, f=1
    # M7: g=1, h=1
    
    # Coefficients for W (C part)
    # C00 = W[2l*2+0] = W[4l]
    # C01 = W[2l*2+1] ? No.
    # W index is k*n + i.
    # C_{i, k}.
    # C_{0, 2l} -> i=0, k=2l -> idx = 2l*2 + 0 = 4l
    # C_{0, 2l+1} -> i=0, k=2l+1 -> idx = (2l+1)*2 + 0 = 4l+2
    # C_{1, 2l} -> i=1, k=2l -> idx = 2l*2 + 1 = 4l+1
    # C_{1, 2l+1} -> i=1, k=2l+1 -> idx = (2l+1)*2 + 1 = 4l+3
    
    # M1: C00=1, C11=1
    # M2: C10=1, C11=-1
    # M3: C01=1, C11=1
    # M4: C00=-1, C01=1
    # M5: C00=1
    # M6: C11=1
    # M7: C00=-1, C10=1

    for l in range(p // 2):
        base_col = l * 7
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # M1
        r = base_col + 0
        U[0][r] = 1
        U[3][r] = 1
        V[0 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = 1
        W[k1 * n + 1][r] = 1
        
        # M2
        r = base_col + 1
        U[2][r] = 1
        U[3][r] = 1
        V[0 * p + k0][r] = 1
        W[k0 * n + 1][r] = 1
        W[k1 * n + 1][r] = -1
        
        # M3
        r = base_col + 2
        U[0][r] = 1
        V[0 * p + k1][r] = 1
        V[1 * p + k1][r] = -1
        W[k1 * n + 0][r] = 1
        W[k1 * n + 1][r] = 1
        
        # M4
        r = base_col + 3
        U[3][r] = 1
        V[1 * p + k0][r] = 1
        V[0 * p + k0][r] = -1
        W[k0 * n + 0][r] = -1
        W[k1 * n + 0][r] = 1
        
        # M5
        r = base_col + 4
        U[0][r] = 1
        U[1][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = 1
        
        # M6
        r = base_col + 5
        U[2][r] = 1
        U[0][r] = -1
        V[0 * p + k0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 1][r] = 1
        
        # M7
        r = base_col + 6
        U[1][r] = 1
        U[3][r] = -1
        V[1 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = -1
        W[k0 * n + 1][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
