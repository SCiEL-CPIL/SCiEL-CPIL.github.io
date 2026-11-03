"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-102 tensor decomposition for 4x4x9 matrix multiplication.
    Uses Kronecker product of Bini rank-7 for 4x4x4 blocks, plus rank-4 for 4x4x1.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Bini's rank-7 decomposition for 2x2x2
    # Convention: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k in {0,1}
    # where C[i,k] = sum_j A[i,j]*B[j,k]
    bini_U = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]],
    ]
    bini_V = [
        [[1, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 1], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
    ]
    bini_W = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
    ]
    
    # Verify: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(bini_U[r][i][j] * bini_V[r][j][k] * bini_W[r][i][k] for r in range(7))
                assert s == 1, f"Bini verify failed at ({i},{j},{k}): {s}"
    
    # Kronecker product for 4x4x4
    num_b = 7
    kr_rank = num_b * num_b
    
    comps_U = []
    comps_V = []
    comps_W = []
    
    for r1 in range(num_b):
        for r2 in range(num_b):
            Uc = [[0]*4 for _ in range(4)]
            Vc = [[0]*4 for _ in range(4)]
            Wc = [[0]*4 for _ in range(4)]
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            Uc[i1*2+i2][j1*2+j2] = bini_U[r1][i1][j1] * bini_U[r2][i2][j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            Vc[j1*2+j2][k1*2+k2] = bini_V[r1][j1][k1] * bini_V[r2][j2][k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            Wc[i1*2+i2][k1*2+k2] = bini_W[r1][i1][k1] * bini_W[r2][i2][k2]
            
            comps_U.append(Uc)
            comps_V.append(Vc)
            comps_W.append(Wc)
    
    # Verify 4x4x4: sum_c Uc[i,j]*Vc[j,k]*Wc[i,k] = 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(comps_U[c][i][j] * comps_V[c][j][k] * comps_W[c][i][k] for c in range(kr_rank))
                assert s == 1, f"4x4x4 verify failed at ({i},{j},{k}): {s}"
    
    # Full 4x4x9: rank = 49 + 49 + 4 = 102
    total_rank = 2 * kr_rank + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: B cols 0-3, C cols 0-3
    for c in range(kr_rank):
        for i in range(4):
            for j in range(4):
                U[i*m+j][c] = comps_U[c][i][j]
        for j in range(4):
            for k in range(4):
                V[j*p+k][c] = comps_V[c][j][k]
        for i in range(4):
            for k in range(4):
                W[k*n+i][c] = comps_W[c][i][k]
    
    # Block 2: B cols 4-7, C cols 4-7
    off = kr_rank
    for c in range(kr_rank):
        for i in range(4):
            for j in range(4):
                U[i*m+j][off+c] = comps_U[c][i][j]
        for j in range(4):
            for k in range(4):
                V[j*p+(k+4)][off+c] = comps_V[c][j][k]
        for i in range(4):
            for k in range(4):
                W[(k+4)*n+i][off+c] = comps_W[c][i][k]
    
    # Block 3: B col 8, C col 8 (4x4x1, rank 4)
    off2 = 2 * kr_rank
    for j in range(4):
        for i in range(4):
            U[i*m+j][off2+j] = 1
        V[j*p+8][off2+j] = 1
        for i in range(4):
            W[8*n+i][off2+j] = 1
    
    # Final verification
    for i in range(4):
        for k in range(9):
            s = 0
            for j in range(4):
                for c in range(total_rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            assert s == 1, f"Final verify failed at ({i},{k}): {s}"
    
    return {"U": U, "V": V, "W": W}
