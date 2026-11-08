import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2_factors():
    """
    Return (U, V, W) for 2x2x2 Strassen.
    U: (4, 7), V: (4, 7), W: (4, 7)
    where U[i,j] corresponds to A[i//2, i%2], etc.
    """
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Strassen:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11) * (b00 + b11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (a10 + a11) * b00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3: a00 * (b01 - b11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4: a11 * (b10 - b00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (a00 + a01) * b11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: (a10 - a00) * (b00 + b01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (a01 - a11) * (b10 + b11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition using:
    # Part 1: Row 0 of A (1x4) times B (4x8) -> 32 rank terms
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) -> use Strassen on 2x2 blocks
    
    # Total rank = 32 + 8 * 7 = 32 + 56 = 88
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: Row 0 (1x4) x (4x8)
    # For each j in 0..3 and k in 0..7:
    # U[0*m + j][col] = 1
    # V[j*p + k][col] = 1
    # W[k*n + 0][col] = 1
    
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: Rows 1-2 (2x4) x (4x8)
    # Split into 2x2 blocks
    # j_blocks: 0, 1 (columns 0-1, 2-3 of A)
    # k_blocks: 0, 1, 2, 3 (columns 0-1, 2-3, 4-5, 6-7 of B)
    
    sU, sV, sW = _strassen_2x2x2_factors()
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                for l in range(4):
                    coeff = sU[t, l]
                    if coeff != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j, col] += coeff
                        
                for l in range(4):
                    coeff = sV[t, l]
                    if coeff != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k, col] += coeff
                        
                for l in range(4):
                    coeff = sW[t, l]
                    if coeff != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i, col] += coeff
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
