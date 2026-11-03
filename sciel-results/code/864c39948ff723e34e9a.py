import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 12 = 84
    rank = 84
    
    # U is n*m x rank
    U = [[0] * rank for _ in range(n * m)]
    # V is m*p x rank
    V = [[0] * rank for _ in range(m * p)]
    # W is p*n x rank
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps
    # Each step is (alpha_dict, beta_dict, gamma_dict)
    # alpha: block -> coeff for A
    # beta: block -> coeff for B
    # gamma: block -> coeff for C
    # Blocks: 00, 01, 10, 11
    
    steps = [
        # M1
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        # M3
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        # M4
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        # M5
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        # M6
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        # M7
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})
    ]
    
    # Helper to get row index in U for A_block[i1, j1]
    def get_u_row(block, i1, j1):
        # A is 4x4
        # 00: i in 0-1, j in 0-1
        # 01: i in 0-1, j in 2-3
        # 10: i in 2-3, j in 0-1
        # 11: i in 2-3, j in 2-3
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError("Invalid block")
        return i * m + j
        
    # Helper to get row index in V for B_block[j1, k1]
    def get_v_row(block, j1, k1):
        # B is 4x6
        # 00: j in 0-1, k in 0-2
        # 01: j in 0-1, k in 3-5
        # 10: j in 2-3, k in 0-2
        # 11: j in 2-3, k in 3-5
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return j * p + k
        
    # Helper to get row index in W for C_block[i1, k1]
    # Convention: W[k*n+i, r]
    def get_w_row(block, i1, k1):
        # C is 4x6
        # 00: i in 0-1, k in 0-2
        # 01: i in 0-1, k in 3-5
        # 10: i in 2-3, k in 0-2
        # 11: i in 2-3, k in 3-5
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    # U contributions
                    for b, val in alpha.items():
                        r = get_u_row(b, i1, j1)
                        U[r][col] += val
                        
                    # V contributions
                    for b, val in beta.items():
                        r = get_v_row(b, j1, k1)
                        V[r][col] += val
                        
                    # W contributions
                    for b, val in gamma.items():
                        r = get_w_row(b, i1, k1)
                        W[r][col] += val
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
