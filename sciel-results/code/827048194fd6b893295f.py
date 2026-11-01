"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56 decomposition using 8 Strassen blocks (rank 7 each)
    rank = 56
    
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M_q = A_comb_q * B_comb_q
    # A_comb_q is a vector of 4 coeffs for A entries
    # B_comb_q is a vector of 4 coeffs for B entries
    # C_comb is a 4x7 matrix where C_comb[c_idx, q] is the coeff of M_q in C_c_idx
    
    A_coeffs = [
        [1, 0, 1, 1], # M0: a+d
        [0, 0, 1, 1], # M1: c+d
        [1, 0, 0, 0], # M2: a
        [0, 0, 0, 1], # M3: d
        [1, 1, 0, 0], # M4: a+b
        [-1, 0, 1, 0],# M5: c-a
        [0, 1, 0, -1] # M6: b-d
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M0: e+h
        [1, 0, 0, 0], # M1: e
        [0, 1, 0, -1],# M2: f-h
        [-1, 0, 1, 0],# M3: g-e
        [0, 0, 0, 1], # M4: h
        [1, 1, 0, 0], # M5: e+f
        [0, 0, 1, 1]  # M6: g+h
    ]
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00: M0 + M3 - M4 + M6
        [0, 0, 1, 0, 1, 0, 0],   # C01: M2 + M4
        [0, 1, 0, 1, 0, 0, 0],   # C10: M1 + M3
        [1, -1, 1, 0, 0, 1, 0]   # C11: M0 - M1 + M2 + M5
    ]
    
    col = 0
    # 8 blocks: 2 j-blocks x 4 k-blocks
    for j_block in range(2):
        j0 = j_block * 2
        j1 = j_block * 2 + 1
        for k_block in range(4):
            k0 = k_block * 2
            k1 = k_block * 2 + 1
            
            for q in range(7):
                # A part
                # A indices in block: 
                # local (0,0) -> global i=0, j=j0
                # local (0,1) -> global i=0, j=j1
                # local (1,0) -> global i=1, j=j0
                # local (1,1) -> global i=1, j=j1
                for li in range(2):
                    for lj in range(2):
                        idx_A = li * 2 + lj
                        val = A_coeffs[q][idx_A]
                        if val != 0:
                            gi = li
                            gj = j0 + lj
                            U[gi * m + gj][col] = val
                
                # B part
                # B indices in block:
                # local (0,0) -> global j=j0, k=k0
                # local (0,1) -> global j=j0, k=k1
                # local (1,0) -> global j=j1, k=k0
                # local (1,1) -> global j=j1, k=k1
                for lj in range(2):
                    for lk in range(2):
                        idx_B = lj * 2 + lk
                        val = B_coeffs[q][idx_B]
                        if val != 0:
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk][col] = val
                
                # C part
                # C indices in block:
                # local (0,0) -> global i=0, k=k0
                # local (0,1) -> global i=0, k=k1
                # local (1,0) -> global i=1, k=k0
                # local (1,1) -> global i=1, k=k1
                for li in range(2):
                    for lk in range(2):
                        idx_C = li * 2 + lk
                        val = C_coeffs[idx_C][q]
                        if val != 0:
                            gi = li
                            gk = k0 + lk
                            W[gk * n + gi][col] = val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
