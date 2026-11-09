"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Construct a rank-32 decomposition for 4x4*4x4 using 2x2 block Strassen
    # Then use two copies for 4x4*4x8, giving rank 64.
    
    # For 4x4*4x4, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    # where Cij are 2x2 blocks.
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2*2x2 product can be done with 7 mults using Strassen.
    # But the products are not independent, so we can't simply use 4*7=28.
    
    # For simplicity, let me use schoolbook for each 2x2*2x2: 8 mults each.
    # Total: 4*8 = 32 mults.
    
    # This gives a rank-32 decomposition for 4x4*4x4.
    
    n2, m2, p2 = 4, 4, 4
    rank_4x4x4 = 32
    U_4x4x4 = [[0]*rank_4x4x4 for _ in range(n2*m2)]
    V_4x4x4 = [[0]*rank_4x4x4 for _ in range(m2*p2)]
    W_4x4x4 = [[0]*rank_4x4x4 for _ in range(p2*n2)]
    
    col = 0
    # For each 2x2 block of C, compute it using schoolbook 2x2*2x2
    for ci in range(2):  # block row of C
        for cj in range(2):  # block col of C
            # C[ci,cj] = sum over bi,bj of A[ci,bi]*B[bi,cj]
            for bi in range(2):
                for bj in range(2):
                    # A block (ci,bi), B block (bi,cj)
                    # Each is 2x2, so 4 scalar mults
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                # A row: 2*ci + ii, col: 2*bi + jj
                                # B row: 2*bi + kk, col: 2*cj + jj... wait
                                # B is 4x4, block (bi,cj) has rows 2*bi..2*bi+1, cols 2*cj..2*cj+1
                                # So B row: 2*bi + kk, col: 2*cj + jj
                                # C row: 2*ci + ii, col: 2*cj + jj
                                a_row = 2*ci + ii
                                a_col = 2*bi + jj
                                b_row = 2*bi + kk
                                b_col = 2*cj + jj
                                c_row = 2*ci + ii
                                c_col = 2*cj + jj
                                
                                U_4x4x4[a_row*m2 + a_col][col] = 1
                                V_4x4x4[b_row*p2 + b_col][col] = 1
                                W_4x4x4[c_row*n2 + c_col][col] = 1
                                col += 1
    
    # Now for 4x4*4x8, use two copies
    rank = 2 * rank_4x4x4  # 64
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    offset = rank_4x4x4
    
    # First half: B cols 0-3, C cols 0-3
    for r in range(rank_4x4x4):
        for i in range(n*m):
            U[i][r] = U_4x4x4[i][r]
        for j in range(m):
            for k in range(4):
                V[j*p + k][r] = V_4x4x4[j*4 + k][r]
        for k_out in range(4):
            for i_out in range(n):
                W[k_out*n + i_out][r] = W_4x4x4[k_out*4 + i_out][r]
    
    # Second half: B cols 4-7, C cols 4-7
    for r in range(rank_4x4x4):
        for i in range(n*m):
            U[i][offset+r] = U_4x4x4[i][r]
        for j in range(m):
            for k in range(4, 8):
                local_k = k - 4
                V[j*p + k][offset+r] = V_4x4x4[j*4 + local_k][r]
        for k_out in range(4, 8):
            local_k_out = k_out - 4
            for i_out in range(n):
                W[k_out*n + i_out][offset+r] = W_4x4x4[local_k_out*4 + i_out][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
