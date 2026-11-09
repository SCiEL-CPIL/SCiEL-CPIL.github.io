import time
import numpy as np

def construct():
    """
    Rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    
    This decomposition splits the 4-dimensional middle index into two 
    2-dimensional parts. This results in two independent subproblems 
    of size 2x2 by 2x8.
    
    Each 2x2 by 2x8 subproblem is solved by splitting the 8-dimensional 
    output into four 2-dimensional blocks, resulting in four independent 
    2x2 by 2x2 matrix multiplications.
    
    Each 2x2 by 2x2 multiplication is solved using Strassen's algorithm 
    with rank 7.
    
    Total rank = 2 (middle splits) * 4 (output splits) * 7 (Strassen) = 56.
    """
    n, m, p = 2, 4, 8
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b10 - b11)
    # m4 = a11 * (b01 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b10)
    # m7 = (a01 - a11) * (b01 + b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    X_forms = [
        [1, 0, 0, 1],  # m1: a00+a11
        [0, 0, 1, 1],  # m2: a10+a11
        [1, 0, 0, 0],  # m3: a00
        [0, 0, 0, 1],  # m4: a11
        [1, 1, 0, 0],  # m5: a00+a01
        [-1, 0, 1, 0], # m6: a10-a00
        [0, 1, 0, -1], # m7: a01-a11
    ]
    
    Y_forms = [
        [1, 0, 0, 1],  # m1: b00+b11
        [1, 0, 0, 0],  # m2: b00
        [0, 1, 0, -1], # m3: b10-b11
        [-1, 0, 1, 0], # m4: b01-b00
        [0, 0, 0, 1],  # m5: b11
        [1, 1, 0, 0],  # m6: b00+b10
        [0, 0, 1, 1],  # m7: b01+b11
    ]
    
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0],  # c11
    ]
    
    # Dimensions
    num_middle_splits = 2  # Split m=4 into two 2s
    num_output_splits = 4  # Split p=8 into four 2s
    num_strassen = 7
    
    rank = num_middle_splits * num_output_splits * num_strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for mid_split in range(num_middle_splits):
        for out_split in range(num_output_splits):
            for s in range(num_strassen):
                # U coefficients (A part)
                # A is 2x4. mid_split selects columns 0-1 or 2-3.
                # s selects the Strassen combination.
                for i in range(n):
                    for j in range(2):
                        a_idx = i * m + mid_split * 2 + j
                        U[a_idx][r] = X_forms[s][i * 2 + j]
                
                # V coefficients (B part)
                # B is 4x8. mid_split selects rows 0-1 or 2-3.
                # out_split selects columns 0-1, 2-3, 4-5, 6-7.
                for j in range(2):
                    for k in range(2):
                        b_idx = (mid_split * 2 + j) * p + (out_split * 2 + k)
                        V[b_idx][r] = Y_forms[s][j * 2 + k]
                
                # W coefficients (C part)
                # C is 2x8. out_split selects columns 0-1, 2-3, 4-5, 6-7.
                for i in range(n):
                    for k in range(2):
                        c_idx = (out_split * 2 + k) * n + i
                        W[c_idx][r] = Z_combos[i * 2 + k][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    n, m, p = 2, 4, 8
    assert len(U) == n*m
    assert len(U[0]) == len(V[0])
    assert len(V) == m*p
    assert len(W) == p*n
    assert len(W[0]) == len(U[0])
    print("Constructed successfully with rank", len(U[0]))
