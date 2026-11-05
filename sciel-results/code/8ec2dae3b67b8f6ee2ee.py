import numpy as np

def construct():
    """
    Returns the factor matrices U, V, W for the rank-189 decomposition 
    of the 6x6 matrix multiplication tensor using Strassen's algorithm 
    on 2x2 blocks within a 3x3 block structure.
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition details
    # M_s = (linear comb of X) * (linear comb of Y)
    # Z = linear comb of M_s
    
    # X indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Y indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Z indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U_local vectors for each M_s (size 4)
    # V_local vectors for each M_s (size 4)
    # W_local vectors for each M_s (size 4)
    
    U_loc = [
        [1, 0, 0, 1], # M1: X00 + X11
        [0, 0, 1, 1], # M2: X10 + X11
        [1, 0, 0, 0], # M3: X00
        [0, 0, 0, 1], # M4: X11
        [1, 0, 1, 0], # M5: X00 + X10
        [-1, 0, 0, 1],# M6: X11 - X00
        [0, 1, -1, 0] # M7: X01 - X11
    ]
    
    V_loc = [
        [1, 0, 0, 1], # M1: Y00 + Y11
        [1, 0, 0, 0], # M2: Y00
        [0, -1, 1, 0],# M3: Y10 - Y11
        [-1, 1, 0, 0],# M4: Y01 - Y00
        [0, 0, 0, 1], # M5: Y11
        [1, 1, 0, 0], # M6: Y00 + Y01
        [0, 0, 1, 1]  # M7: Y10 + Y11
    ]
    
    W_loc = [
        [1, 0, 0, 1], # M1: Z00 + Z11
        [0, 0, 1, -1],# M2: Z10 - Z11
        [0, 1, 0, 1], # M3: Z01 + Z11
        [1, 0, 1, 0], # M4: Z00 + Z10
        [-1, 1, 0, 0],# M5: -Z00 + Z01
        [0, 0, 0, 1]  # M6: Z11
    ]
    # Note: M7 is missing in W_loc above? 
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    # Let's re-verify W_loc for M7.
    # M7 contributes +1 to Z00.
    # So W_loc[6] should be [1, 0, 0, 0].
    
    W_loc = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    # Iterate over block indices I, J, K
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # U vector
                    for l in range(4):
                        if U_loc[s][l] != 0:
                            # l maps to local (i', j')
                            # l=0:(0,0), l=1:(0,1), l=2:(1,0), l=3:(1,1)
                            i_prime = l // 2
                            j_prime = l % 2
                            u_idx = (I * 2 + i_prime) * 6 + (J * 2 + j_prime)
                            U[u_idx][col] = U_loc[s][l]
                            
                    # V vector
                    for l in range(4):
                        if V_loc[s][l] != 0:
                            i_prime = l // 2
                            j_prime = l % 2
                            v_idx = (J * 2 + i_prime) * 6 + (K * 2 + j_prime)
                            V[v_idx][col] = V_loc[s][l]
                            
                    # W vector
                    for l in range(4):
                        if W_loc[s][l] != 0:
                            i_prime = l // 2
                            j_prime = l % 2
                            w_idx = (K * 2 + i_prime) * 6 + (I * 2 + j_prime)
                            W[w_idx][col] = W_loc[s][l]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
