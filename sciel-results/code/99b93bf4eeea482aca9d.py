import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 3
    bs = 3
    # Strassen terms
    # a: {(I,J): coeff}, b: {(K,L): coeff}, c: {(P,Q): coeff}
    terms = [
        # M1
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': {(0,0): 1, (1,1): 1}},
        # M2
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': {(1,0): 1}},
        # M3
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': {(0,1): 1}},
        # M4
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': {(0,0): 1, (1,0): 1}},
        # M5
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': {(0,0): -1, (0,1): 1}},
        # M6
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': {(1,1): 1}},
        # M7
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': {(0,0): 1, (1,1): -1}},
    ]
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in terms:
        a_coeffs = t['a']
        b_coeffs = t['b']
        c_coeffs = t['c']
        
        for i_prime in range(bs):
            for j_prime in range(bs):
                for k_prime in range(bs):
                    # U
                    for (I, J), coeff_a in a_coeffs.items():
                        row = I * bs + i_prime
                        col_idx = J * bs + j_prime
                        U[row * m + col_idx][col] = coeff_a
                    # V
                    for (K, L), coeff_b in b_coeffs.items():
                        row = K * bs + j_prime
                        col_idx = L * bs + k_prime
                        V[row * p + col_idx][col] = coeff_b
                    # W
                    for (P, Q), coeff_c in c_coeffs.items():
                        row = P * bs + i_prime
                        col_idx = Q * bs + k_prime
                        # W index is k*n + i -> col_idx * n + row
                        W[col_idx * n + row][col] = coeff_c
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
