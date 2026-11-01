"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42 based on Strassen decomposition of 2x2x2 blocks
    # 3 column blocks * 2 (top/bottom) * 7 (Strassen) = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms definition:
    # Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict { (i,j): coeff } for X
    # v_coeffs: dict { (i,j): coeff } for Y
    # w_coeffs: dict { (i,j): coeff } for Z
    
    # M1: (x00+x11)(y00+y11) -> z00+=1, z11+=1
    M1 = (
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    )
    # M2: (x10+x11)y00 -> z10+=1, z11+=-1
    M2 = (
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}, 
        {(1,0): 1, (1,1): -1}
    )
    # M3: x00(y01-y11) -> z01+=1, z11+=1
    M3 = (
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}, 
        {(0,1): 1, (1,1): 1}
    )
    # M4: x11(y00-y01) -> z00+=1, z10+=1
    M4 = (
        {(1,1): 1}, 
        {(0,0): 1, (0,1): -1}, 
        {(0,0): 1, (1,0): 1}
    )
    # M5: (x00+x01)y11 -> z00+=-1, z01+=1
    M5 = (
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}, 
        {(0,0): -1, (0,1): 1}
    )
    # M6: (x10-x00)(y00+y01) -> z11+=1
    M6 = (
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}
    )
    # M7: (x01-x11)(y10+y11) -> z00+=1
    M7 = (
        {(0,1): 1, (1,1): -1}, 
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}
    )
    
    strassen_terms = [M1, M2, M3, M4, M5, M6, M7]
    
    for src in ['L', 'R']:
        for k in range(3):
            for term in strassen_terms:
                u_coeffs, v_coeffs, w_coeffs = term
                
                # Fill U
                for (i, j), coeff in u_coeffs.items():
                    if src == 'L':
                        u_row = i * m + j
                    else:
                        u_row = i * m + (2 + j)
                    U[u_row][col] += coeff
                    
                # Fill V
                for (i, j), coeff in v_coeffs.items():
                    if src == 'L':
                        v_row = i * p + (3 * k + j)
                    else:
                        v_row = (2 + i) * p + (3 * k + j)
                    V[v_row][col] += coeff
                    
                # Fill W
                for (i, j), coeff in w_coeffs.items():
                    # Output C[i, 3k+j]
                    # W index: k_global * n + i_global
                    # k_global is column of B, which is 3k+j
                    # i_global is row of A/C, which is i
                    w_row = (3 * k + j) * n + i
                    W[w_row][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
