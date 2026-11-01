"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (Strassen) * (p / 2) blocks = 7 * 8 = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A = [[a, b], [c, d]]
    # U indices: 0=a, 1=b, 2=c, 3=d
    A_forms = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # Strassen linear forms for B (2x2) for each block
    # B_l = [[e, f], [g, h]]
    # V indices for block l: 
    # j=0: k=0,1 -> indices 0, 1 (e, f)
    # j=1: k=0,1 -> indices 2, 3 (g, h)
    # Global V index for block l, local (j, k_local):
    # k_global = 2*l + k_local
    # v_index = j * 16 + k_global
    
    B_forms = [
        [1, 1, 0, 0], # M1: e+h (e is 0,0; h is 1,1) -> v(0, 2l), v(1, 2l+1)
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, -1, 1, 0],# M4: g-f
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # Reconstruction coefficients for C (2x2)
    # C = [[C00, C01], [C10, C11]]
    # W indices for block l:
    # i=0, k=0 -> w(2l, 0)
    # i=0, k=1 -> w(2l+1, 0)
    # i=1, k=0 -> w(2l, 1)
    # i=1, k=1 -> w(2l+1, 1)
    # Global W index: k_global * 2 + i
    
    C_forms = [
        [1, 0, 0, 1], # C00: M1, M4, -M5, M7 -> coeffs for M1..M7
        [0, 1, 1, 0], # C01: M3, M5
        [0, 0, 1, 0], # C10: M2, M4
        [1, -1, 1, 1] # C11: M1, -M2, M3, M6
    ]
    # Wait, C_forms needs to be a list of 4 lists, each of length 7
    # C00 uses M1, M4, M5, M7. Coeffs: 1, 0, -1, 1 for indices 0, 3, 4, 6.
    # Let's define W_coeffs[k, i, m] where m is the Strassen index 0..6
    
    W_coeffs = [
        [1, 0, -1, 1, 0, 0, 0], # C00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],  # C01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],  # C10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0]  # C11: M1 - M2 + M3 + M6
    ]
    
    for l in range(8):
        for m in range(7):
            r = l * 7 + m
            
            # Fill U
            for i in range(4):
                U[i][r] = A_forms[m][i]
                
            # Fill V
            # B_forms[m] is [e, f, g, h] coeffs
            # e is at V[0 * 16 + 2*l]
            # f is at V[0 * 16 + 2*l + 1]
            # g is at V[1 * 16 + 2*l]
            # h is at V[1 * 16 + 2*l + 1]
            v_base = 2 * l
            V[0 * 16 + v_base][r] = B_forms[m][0]
            V[0 * 16 + v_base + 1][r] = B_forms[m][1]
            V[1 * 16 + v_base][r] = B_forms[m][2]
            V[1 * 16 + v_base + 1][r] = B_forms[m][3]
            
            # Fill W
            # W index = k_global * 2 + i
            # C00 is i=0, k=0 -> k_global=2l
            # C01 is i=0, k=1 -> k_global=2l+1
            # C10 is i=1, k=0 -> k_global=2l
            # C11 is i=1, k=1 -> k_global=2l+1
            
            W[2*l * 2 + 0][r] = W_coeffs[0][m]
            W[(2*l+1) * 2 + 0][r] = W_coeffs[1][m]
            W[2*l * 2 + 1][r] = W_coeffs[2][m]
            W[(2*l+1) * 2 + 1][r] = W_coeffs[3][m]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
