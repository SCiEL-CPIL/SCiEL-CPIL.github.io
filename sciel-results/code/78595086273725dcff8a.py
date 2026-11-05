import time

def construct():
    n = 6
    m = 6
    p = 6
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms
    # Each term is a dict:
    # 'A': {(br, bc): coeff}, 'B': {(br, bc): coeff}, 'C': {(br, bc): coeff}
    terms = [
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    col = 0
    for term in terms:
        A_coeffs = term['A']
        B_coeffs = term['B']
        C_coeffs = term['C']
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # U part
                    for (br_A, bc_A), c_A in A_coeffs.items():
                        # Local i, j
                        # Global I = 3*br_A + i, J = 3*bc_A + j
                        I = 3 * br_A + i
                        J = 3 * bc_A + j
                        U[I * m + J][col] += c_A
                    
                    # V part
                    for (br_B, bc_B), c_B in B_coeffs.items():
                        # Local j, k
                        # Global I = 3*br_B + j, J = 3*bc_B + k
                        I = 3 * br_B + j
                        J = 3 * bc_B + k
                        V[I * p + J][col] += c_B
                    
                    # W part
                    for (br_C, bc_C), c_C in C_coeffs.items():
                        # Local k, i (W index k*n+i, here n=3 locally)
                        # Global R = 3*br_C + k, C = 3*bc_C + i
                        R = 3 * br_C + k
                        C_idx = 3 * bc_C + i
                        W[R * n + C_idx][col] += c_C
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
