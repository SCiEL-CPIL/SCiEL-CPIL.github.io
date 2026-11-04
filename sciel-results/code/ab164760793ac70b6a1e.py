import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is 56
    # 2 j-blocks * 28 rank per j-block
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a 2x2x2 Strassen block
    # a_i, a_j are row indices in A
    # b_j, b_k are col indices in B (which are row indices in B)
    # c_k, c_i are col indices in C (which are row indices in C for W)
    # Note: W index is k*n + i.
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col_idx
        # A indices: i1*m + j1, i1*m + j2, i2*m + j1, i2*m + j2
        # B indices: j1*p + k1, j1*p + k2, j2*p + k1, j2*p + k2
        # C indices: k1*n + i1, k1*n + i2, k2*n + i1, k2*n + i2
        
        # m1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        U[i1*m + j1][col_idx] += 1
        U[i2*m + j2][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        V[j2*p + k2][col_idx] += 1
        # C00 += m1, C11 += m1
        W[k1*n + i1][col_idx] += 1
        W[k2*n + i2][col_idx] += 1
        col_idx += 1
        
        # m2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        U[i2*m + j1][col_idx] += 1
        U[i2*m + j2][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        # C10 += m2
        W[k1*n + i2][col_idx] += 1
        col_idx += 1
        
        # m3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        U[i1*m + j1][col_idx] += 1
        V[j1*p + k2][col_idx] += 1
        V[j2*p + k2][col_idx] -= 1
        # C01 += m3
        W[k2*n + i1][col_idx] += 1
        col_idx += 1
        
        # m4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        U[i2*m + j2][col_idx] += 1
        V[j2*p + k1][col_idx] += 1
        V[j1*p + k1][col_idx] -= 1
        # C00 += m4, C10 += m4
        W[k1*n + i1][col_idx] += 1
        W[k1*n + i2][col_idx] += 1
        col_idx += 1
        
        # m5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        U[i1*m + j1][col_idx] += 1
        U[i1*m + j2][col_idx] += 1
        V[j2*p + k2][col_idx] += 1
        # C00 -= m5, C01 += m5
        W[k1*n + i1][col_idx] -= 1
        W[k2*n + i1][col_idx] += 1
        col_idx += 1
        
        # m6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        U[i2*m + j1][col_idx] += 1
        U[i1*m + j1][col_idx] -= 1
        V[j1*p + k1][col_idx] += 1
        V[j1*p + k2][col_idx] += 1
        # C11 += m6
        W[k2*n + i2][col_idx] += 1
        col_idx += 1
        
        # m7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        U[i1*m + j2][col_idx] += 1
        U[i2*m + j2][col_idx] -= 1
        V[j2*p + k1][col_idx] += 1
        V[j2*p + k2][col_idx] += 1
        # C00 += m7, C11 -= m7
        W[k1*n + i1][col_idx] += 1
        W[k2*n + i2][col_idx] -= 1
        col_idx += 1

    # Helper to add a 2x2x1 block (schoolbook)
    def add_221_block(i1, i2, j1, j2, k1):
        nonlocal col_idx
        # C[i1,k1] += A[i1,j1]B[j1,k1] + A[i1,j2]B[j2,k1]
        # C[i2,k1] += A[i2,j1]B[j1,k1] + A[i2,j2]B[j2,k1]
        
        # m1 = A[i1,j1] * B[j1,k1]
        U[i1*m + j1][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1
        
        # m2 = A[i1,j2] * B[j2,k1]
        U[i1*m + j2][col_idx] += 1
        V[j2*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1
        
        # m3 = A[i2,j1] * B[j1,k1]
        U[i2*m + j1][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        W[k1*n + i2][col_idx] += 1
        col_idx += 1
        
        # m4 = A[i2,j2] * B[j2,k1]
        U[i2*m + j2][col_idx] += 1
        V[j2*p + k1][col_idx] += 1
        W[k1*n + i2][col_idx] += 1
        col_idx += 1

    # Helper to add a 1x2x2 block (schoolbook)
    def add_122_block(i1, j1, j2, k1, k2):
        nonlocal col_idx
        # C[i1,k1] += A[i1,j1]B[j1,k1] + A[i1,j2]B[j2,k1]
        # C[i1,k2] += A[i1,j1]B[j1,k2] + A[i1,j2]B[j2,k2]
        
        # m1 = A[i1,j1] * B[j1,k1]
        U[i1*m + j1][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1
        
        # m2 = A[i1,j1] * B[j1,k2]
        U[i1*m + j1][col_idx] += 1
        V[j1*p + k2][col_idx] += 1
        W[k2*n + i1][col_idx] += 1
        col_idx += 1
        
        # m3 = A[i1,j2] * B[j2,k1]
        U[i1*m + j2][col_idx] += 1
        V[j2*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1
        
        # m4 = A[i1,j2] * B[j2,k2]
        U[i1*m + j2][col_idx] += 1
        V[j2*p + k2][col_idx] += 1
        W[k2*n + i1][col_idx] += 1
        col_idx += 1

    # Helper to add a 1x2x1 block (schoolbook)
    def add_121_block(i1, j1, j2, k1):
        nonlocal col_idx
        # C[i1,k1] += A[i1,j1]B[j1,k1] + A[i1,j2]B[j2,k1]
        
        # m1 = A[i1,j1] * B[j1,k1]
        U[i1*m + j1][col_idx] += 1
        V[j1*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1
        
        # m2 = A[i1,j2] * B[j2,k1]
        U[i1*m + j2][col_idx] += 1
        V[j2*p + k1][col_idx] += 1
        W[k1*n + i1][col_idx] += 1
        col_idx += 1

    # Process J block 0 (j=0,1)
    # I1 = 0,1 ; K1 = 0,1
    add_strassen_block(0, 1, 0, 1, 0, 1)
    # I1 = 0,1 ; K2 = 2,3
    add_strassen_block(0, 1, 0, 1, 2, 3)
    # I1 = 0,1 ; K3 = 4
    add_221_block(0, 1, 0, 1, 4)
    # I2 = 2 ; K1 = 0,1
    add_122_block(2, 0, 1, 0, 1)
    # I2 = 2 ; K2 = 2,3
    add_122_block(2, 0, 1, 2, 3)
    # I2 = 2 ; K3 = 4
    add_121_block(2, 0, 1, 4)

    # Process J block 1 (j=2,3)
    # I1 = 0,1 ; K1 = 0,1
    add_strassen_block(0, 1, 2, 3, 0, 1)
    # I1 = 0,1 ; K2 = 2,3
    add_strassen_block(0, 1, 2, 3, 2, 3)
    # I1 = 0,1 ; K3 = 4
    add_221_block(0, 1, 2, 3, 4)
    # I2 = 2 ; K1 = 0,1
    add_122_block(2, 2, 3, 0, 1)
    # I2 = 2 ; K2 = 2,3
    add_122_block(2, 2, 3, 2, 3)
    # I2 = 2 ; K3 = 4
    add_121_block(2, 2, 3, 4)

    return {"U": U, "V": V, "W": W}
