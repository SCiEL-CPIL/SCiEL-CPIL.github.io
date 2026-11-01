"""Matrix multiplication 4x4x4 with rank-49 Strassen-based decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 matrices (7 factors)
    # U2[m][u][v] is coeff of X[u][v] in m-th factor (0-indexed m)
    U2 = np.array([
        [[1, 0], [0, 1]],   # m=0: X00+X11
        [[0, 0], [1, 1]],   # m=1: X10+X11
        [[1, 0], [0, 0]],   # m=2: X00
        [[0, 0], [0, 1]],   # m=3: X11
        [[1, 1], [0, 0]],   # m=4: X00+X01
        [[-1, 0], [1, 0]],  # m=5: X10-X00
        [[0, 1], [0, -1]]   # m=6: X01-X11
    ], dtype=float)
    
    V2 = np.array([
        [[1, 0], [0, 1]],   # m=0: Y00+Y11
        [[1, 0], [0, 0]],   # m=1: Y00
        [[0, 1], [0, -1]],  # m=2: Y01-Y11
        [[-1, 0], [1, 0]],  # m=3: Y10-Y00
        [[0, 0], [0, 1]],   # m=4: Y11
        [[1, 1], [0, 0]],   # m=5: Y00+Y01
        [[0, 0], [1, 1]]    # m=6: Y10+Y11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],   # m=0: Z00+=1, Z11+=1
        [0, 0, 1, -1],  # m=1: Z10+=1, Z11-=1
        [0, 1, 0, 1],   # m=2: Z01+=1, Z11+=1
        [1, 0, 1, 0],   # m=3: Z00+=1, Z10+=1
        [-1, 1, 0, 0],  # m=4: Z00-=1, Z01+=1
        [0, 0, 0, 1],   # m=5: Z11+=1
        [1, 0, 0, 0]    # m=6: Z00+=1
    ], dtype=float)
    
    # Build 4x4 factors using tensor product (Kronecker product) of 2x2 factors
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # For each pair of 2x2 factors (k, l) where k is outer, l is inner
    for k in range(7):
        for l in range(7):
            r = k * 7 + l  # rank index 0..48
            
            # U: A is 4x4 = (2x2 blocks)
            # A[i][j] where i=2*p+u, j=2*q+v
            # U2[k][p][q] * U2[l][u][v]
            for p in range(2):
                for q in range(2):
                    for u in range(2):
                        for v in range(2):
                            coeff = U2[k, p, q] * U2[l, u, v]
                            if coeff != 0:
                                row = p * 4 + q * 2 + u  # Wait, need correct indexing
                                # A is 4x4, row i = 2*p + u, col j = 2*q + v
                                # Linear index = i * 4 + j
                                i = 2 * p + u
                                j = 2 * q + v
                                idx = i * 4 + j
                                U[idx, r] += coeff
            
            # V: B is 4x4
            # B[j][k_idx] where j=2*r+u, k_idx=2*s+v
            # V2[k][r][s] * V2[l][u][v]
            for r_idx in range(2):
                for s_idx in range(2):
                    for u in range(2):
                        for v in range(2):
                            coeff = V2[k, r_idx, s_idx] * V2[l, u, v]
                            if coeff != 0:
                                j = 2 * r_idx + u
                                k_idx = 2 * s_idx + v
                                idx = j * 4 + k_idx
                                V[idx, r] += coeff
            
            # W: C is 4x4, C[k_idx][i]
            # C[k_idx][i] where k_idx=2*q+v, i=2*p+u
            # W2[k][q][p] * W2[l][v][u]
            # W2 is indexed as [m][row][col] where row*2+col
            for q in range(2):
                for p in range(2):
                    for v in range(2):
                        for u in range(2):
                            coeff = W2[k, q * 2 + p] * W2[l, v * 2 + u]
                            if coeff != 0:
                                k_idx = 2 * q + v
                                i = 2 * p + u
                                idx = k_idx * 4 + i
                                W[idx, r] += coeff
    
    # Convert to lists of integers (or half-integers, but these are all integers)
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
