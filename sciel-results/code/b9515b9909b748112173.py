"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p = 12, so 6 blocks of 2
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u_r[i, j]
    u_coeffs = [
        [1, 0, 0, 1], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [-1, 0, 1, 0],# m6: c-a
        [0, 1, 0, -1] # m7: b-d
    ]
    
    # v_r[j, k]
    # B = [[e, f], [g, h]]
    # e=0, f=1, g=2, h=3
    v_coeffs = [
        [1, 0, 0, 1], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1],# m3: f-h
        [0, -1, 1, 0],# m4: g-f
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    
    # w_r[k, i]
    # C = [[c00, c01], [c10, c11]]
    # Index k*n+i
    # k=0, i=0 -> c00
    # k=0, i=1 -> c10
    # k=1, i=0 -> c01
    # k=1, i=1 -> c11
    w_coeffs = [
        [1, 0, 0, 1], # m1: c00 + c11
        [0, 1, 0, -1],# m2: c10 - c11
        [0, 0, 1, 1], # m3: c01 + c11
        [1, 1, 0, 0], # m4: c00 + c10
        [-1, 0, 1, 0],# m5: -c00 + c01
        [0, 0, 0, 1], # m6: c11
        [1, 0, 0, 0]  # m7: c00
    ]
    
    for l in range(num_blocks):
        for r in range(7):
            col = l * 7 + r
            
            # Fill U
            # U index i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u_coeffs[r][i * m + j]
            
            # Fill V
            # V index j*p + k
            # For block l, k ranges from 2*l to 2*l+1
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * l + k_local
                    V[j * p + k_global][col] = v_coeffs[r][j * 2 + k_local]
            
            # Fill W
            # W index k*n + i
            # For block l, k ranges from 2*l to 2*l+1
            for k_local in range(2):
                for i in range(n):
                    k_global = 2 * l + k_local
                    W[k_global * n + i][col] = w_coeffs[r][k_local * n + i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
