import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 7 steps for 2x2 block multiplication
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})
    ]
    
    # Each Strassen step involves a 2x2 by 2x3 block multiplication
    # The 2x2 x 2x3 tensor has 12 terms in its naive decomposition
    # Total rank = 7 * 12 = 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_u_row(block, i1, j1):
        if block == '00':
            return i1 * m + j1
        elif block == '01':
            return i1 * m + (j1 + 2)
        elif block == '10':
            return (i1 + 2) * m + j1
        elif block == '11':
            return (i1 + 2) * m + (j1 + 2)
        else:
            raise ValueError("Invalid block")
    
    def get_v_row(block, j1, k1):
        if block == '00':
            return j1 * p + k1
        elif block == '01':
            return j1 * p + (k1 + 3)
        elif block == '10':
            return (j1 + 2) * p + k1
        elif block == '11':
            return (j1 + 2) * p + (k1 + 3)
        else:
            raise ValueError("Invalid block")
    
    def get_w_row(block, i1, k1):
        if block == '00':
            return k1 * n + i1
        elif block == '01':
            return (k1 + 3) * n + i1
        elif block == '10':
            return k1 * n + (i1 + 2)
        elif block == '11':
            return (k1 + 3) * n + (i1 + 2)
        else:
            raise ValueError("Invalid block")
    
    col = 0
    for alpha, beta, gamma in steps:
        for i1 in range(2):
            for j1 in range(2):
                for k1 in range(3):
                    for b, val in alpha.items():
                        r = get_u_row(b, i1, j1)
                        U[r][col] += val
                    for b, val in beta.items():
                        r = get_v_row(b, j1, k1)
                        V[r][col] += val
                    for b, val in gamma.items():
                        r = get_w_row(b, i1, k1)
                        W[r][col] += val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
