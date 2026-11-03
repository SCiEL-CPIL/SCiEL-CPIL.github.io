import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 11 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps
    # alpha: block -> coeff for A
    # beta: block -> coeff for B
    # gamma: block -> coeff for C
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})
    ]
    
    # Inner Terms for 2x2 x 2x3 multiplication
    # u: 4 coeffs for a00, a01, a10, a11
    # v: 6 coeffs for b00, b01, b10, b11, b02, b12
    # w: 6 coeffs for m00, m01, m02, m10, m11, m12
    
    inner_terms = [
        # Strassen 2x2x2 (cols 0,1)
        # T1
        ([1, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0]),
        # T2
        ([0, 0, 1, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0]),
        # T3
        ([1, 0, 0, 0], [0, 1, 0, -1, 0, 0], [0, 1, 0, 0, 1, 0]),
        # T4
        ([0, 0, 0, 1], [-1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]),
        # T5
        ([1, 1, 0, 0], [0, 0, 0, 1, 0, 0], [-1, 1, 0, 0, 0, 0]),
        # T6
        ([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
        # T7
        ([0, 1, 0, -1], [0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]),
        # Naive 2x2x1 (col 2)
        # N1
        ([1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]),
        # N2
        ([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]),
        # N3
        ([0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]),
        # N4
        ([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1])
    ]
    
    def get_global_u(block, i, j):
        if block == '00': return i * m + j
        elif block == '01': return i * m + (j + 2)
        elif block == '10': return (i + 2) * m + j
        elif block == '11': return (i + 2) * m + (j + 2)
        else: raise ValueError(block)

    def get_global_v(block, j, k):
        if block == '00': return j * p + k
        elif block == '01': return j * p + (k + 3)
        elif block == '10': return (j + 2) * p + k
        elif block == '11': return (j + 2) * p + (k + 3)
        else: raise ValueError(block)

    def get_global_w(block, i, k):
        # W[k*n + i]
        if block == '00': return k * n + i
        elif block == '01': return (k + 3) * n + i
        elif block == '10': return k * n + (i + 2)
        elif block == '11': return (k + 3) * n + (i + 2)
        else: raise ValueError(block)

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        for u, v, w in inner_terms:
            # Fill U
            for i in range(2):
                for j in range(2):
                    coeff = u[i * 2 + j]
                    if coeff != 0:
                        for b, val in alpha.items():
                            r = get_global_u(b, i, j)
                            U[r][col] += coeff * val
            # Fill V
            for j in range(2):
                for k in range(3):
                    coeff = v[j * 3 + k]
                    if coeff != 0:
                        for b, val in beta.items():
                            r = get_global_v(b, j, k)
                            V[r][col] += coeff * val
            # Fill W
            for i in range(2):
                for k in range(3):
                    coeff = w[i * 3 + k]
                    if coeff != 0:
                        for b, val in gamma.items():
                            r = get_global_w(b, i, k)
                            W[r][col] += coeff * val
            col += 1
            
    return {"U": U, "V": V, "W": W}
