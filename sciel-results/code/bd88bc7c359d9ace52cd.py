"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # We view the 2x16 matrix B as 8 blocks of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 8 * 7 = 56.
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Base column index for this block
        base_c = t * 7
        
        # Strassen terms for 2x2 multiplication:
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f+h)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # U indices: i*m + j
        # 0: a (0,0), 1: b (0,1), 2: c (1,0), 3: d (1,1)
        
        # V indices: j*p + k
        # e = b[0, k0] -> 0*p + k0 = k0
        # f = b[0, k1] -> 0*p + k1 = k1
        # g = b[1, k0] -> 1*p + k0 = p + k0
        # h = b[1, k1] -> 1*p + k1 = p + k1
        
        # W indices: k*n + i
        # C[0, k0] -> k0*n + 0 = 2*k0
        # C[0, k1] -> k1*n + 0 = 2*k1
        # C[1, k0] -> k0*n + 1 = 2*k0 + 1
        # C[1, k1] -> k1*n + 1 = 2*k1 + 1
        
        w_c00 = k0 * n
        w_c01 = k1 * n
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        # s=0: M1 = (a+d)(e+h)
        c = base_c + 0
        U[0][c] = 1  # a
        U[3][c] = 1  # d
        V[v_e][c] = 1
        V[v_h][c] = 1
        W[w_c00][c] = 1
        W[w_c11][c] = 1
        
        # s=1: M2 = (c+d)e
        c = base_c + 1
        U[2][c] = 1  # c
        U[3][c] = 1  # d
        V[v_e][c] = 1
        W[w_c10][c] = 1
        W[w_c11][c] = -1
        
        # s=2: M3 = a(g-h)
        c = base_c + 2
        U[0][c] = 1  # a
        V[v_g][c] = 1
        V[v_h][c] = -1
        W[w_c01][c] = 1
        W[w_c11][c] = 1
        
        # s=3: M4 = d(f+h)
        c = base_c + 3
        U[3][c] = 1  # d
        V[v_f][c] = 1
        V[v_h][c] = 1
        W[w_c00][c] = 1
        W[w_c10][c] = 1
        
        # s=4: M5 = (a+b)h
        c = base_c + 4
        U[0][c] = 1  # a
        U[1][c] = 1  # b
        V[v_h][c] = 1
        W[w_c00][c] = -1
        W[w_c01][c] = 1
        
        # s=5: M6 = (c-a)(e+f)
        c = base_c + 5
        U[2][c] = 1  # c
        U[0][c] = -1 # a
        V[v_e][c] = 1
        V[v_f][c] = 1
        W[w_c11][c] = 1
        
        # s=6: M7 = (b-d)(g+h)
        c = base_c + 6
        U[1][c] = 1  # b
        U[3][c] = -1 # d
        V[v_g][c] = 1
        V[v_h][c] = 1
        W[w_c00][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
