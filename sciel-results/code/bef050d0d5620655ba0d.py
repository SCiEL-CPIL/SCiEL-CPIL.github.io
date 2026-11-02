"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Construct rank 81 solution: 49 (for 4x4x4) + 32 (for 4x4x2)
    
    # Strassen coefficients for 2x2
    # A_coeffs[k][u][v]
    A_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    B_coeffs = [
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, 1]]
    ]
    # C_coeffs[i][j][k]
    C_coeffs = [
        [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]
    ]

    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442
    
    U_new = [[0] * total_rank for _ in range(16)]
    V_new = [[0] * total_rank for _ in range(24)]
    W_new = [[0] * total_rank for _ in range(24)]
    
    # Part 1: 4x4x4 (Rank 49)
    # A is 4x4, B_left is 4x4, C_left is 4x4
    # B_left corresponds to B[:, 0:4]
    # C_left corresponds to C[:, 0:4]
    
    for k_out in range(7):
        # A_k is a 2x2 matrix of linear forms in A
        # B_k is a 2x2 matrix of linear forms in B
        # We need to compute A_k * B_k using inner Strassen
        # This yields 7 terms
        for k_in in range(7):
            col_idx = k_out * 7 + k_in
            
            # Determine U, V, W for this term
            # U: coefficients for A
            # V: coefficients for B (only first 4 cols)
            # W: coefficients for C (only first 4 cols)
            
            # Inner Strassen linear forms for A_k
            # L_A = sum_{u,v} A_inner[k_in][u][v] * A_k[u][v]
            # A_k[u][v] = sum_{i,j} A_outer[k_out][u][v][i][j] * A_block(i,j)
            # A_block(i,j) is 2x2 matrix of A entries
            
            # So L_A is a linear form in A entries.
            # U[i*4+j] = coeff of A[i,j] in L_A
            
            # Similarly for V.
            
            # W: The term contributes to C_left.
            # C_left = sum_{k_out} C_outer[i][j][k_out] * (A_k * B_k)_{i,j}
            # (A_k * B_k)_{i,j} = sum_{k_in} C_inner[i][j][k_in] * M_k_in
            # M_k_in is the bilinear form.
            
            # So W[k*4+i] = C_outer[i][j][k_out] * C_inner[i][j][k_in]
            # for the appropriate i,j,k.
            
            # We need to sum over i,j for W?
            # No, W is a vector.
            # The bilinear form M_k_in is added to C_left entries.
            # The coefficient for C_left[i_c, k_c] is:
            # sum_{i,j} C_outer[i][j][k_out] * C_inner[i][j][k_in] * delta(i_c, i*2+r) * delta(k_c, j*2+c)
            # where r,c are the entries of the 2x2 matrix M_k_in.
            
            # Actually, M_k_in is a 2x2 matrix of bilinear forms.
            # Each entry (M_k_in)_{r,c} is a bilinear form.
            # We represent the term as U_col, V_col, W_col.
            # U_col and V_col are fixed for this (k_out, k_in).
            # W_col has non-zero entries at multiple positions.
            
            # Calculate U_col
            for i in range(4):
                for j in range(4):
                    # A[i,j] is in block (i//2, j//2) at pos (i%2, j%2)
                    bi, bj = i // 2, j // 2
                    ri, rj = i % 2, j % 2
                    
                    # A_k[u][v] contains A_block(i,j) with coeff A_outer[k_out][u][v][i//2][j//2]
                    # L_A = sum_{u,v} A_inner[k_in][u][v] * A_k[u][v]
                    # So coeff of A[i,j] in L_A is:
                    # sum_{u,v} A_inner[k_in][u][v] * A_outer[k_out][u][v][bi][bj]
                    # Note: A_outer[u][v] is a 2x2 matrix of coeffs for the 4 blocks.
                    # A_outer[k_out][u][v] is a 2x2 matrix.
                    # Entry (bi, bj) is the coeff for block (bi, bj).
                    
                    coeff = 0
                    for u in range(2):
                        for v in range(2):
                            coeff += A_coeffs[k_in][u][v] * A_coeffs[k_out][u][v][bi][bj]
                    
                    U_new[i*4+j][col_idx] = coeff
            
            # Calculate V_col
            for j in range(4):
                for k in range(4):
                    # B[j,k] is in block (j//2, k//2) at pos (j%2, k%2)
                    bj, bk = j // 2, k // 2
                    rj, rk = j % 2, k % 2
                    
                    coeff = 0
                    for u in range(2):
                        for v in range(2):
                            coeff += B_coeffs[k_in][u][v] * B_coeffs[k_out][u][v][bj][bk]
                    
                    V_new[j*6+k][col_idx] = coeff
            
            # Calculate W_col
            # C_left[i_c, k_c] gets contribution from:
            # sum_{i,j} C_outer[i][j][k_out] * C_inner[i][j][k_in] * (M_k_in)_{r,c}
            # where i_c = i*2+r, k_c = j*2+c
            # (M_k_in)_{r,c} is the bilinear form.
            # In the rank-1 decomposition, the "value" is 1 (implicitly).
            # So W[k_c*4 + i_c] = sum_{i,j} C_outer[i][j][k_out] * C_inner[i][j][k_in]
            # for the specific i_c, k_c?
            # No, the bilinear form is the SAME for all i,j?
            # No, the bilinear form M_k_in is fixed for (k_out, k_in).
            # It is added to C_left[i_c, k_c] with weight C_outer[i][j][k_out] * C_inner[i][j][k_in].
            # So for a fixed (k_out, k_in), the term contributes to ALL C_left entries?
            # No, the bilinear form is a single object.
            # It is added to C_left[i_c, k_c] with a specific coefficient.
            # So W[k_c*4 + i_c] = C_outer[i][j][k_out] * C_inner[i][j][k_in]
            # where i = i_c // 2, j = k_c // 2.
            
            for i_c in range(4):
                for k_c in range(4):
                    i = i_c // 2
                    j = k_c // 2
                    W_new[k_c*4 + i_c][col_idx] = C_coeffs[i][j][k_out] * C_coeffs[i][j][k_in]
    
    # Part 2: 4x4x4x2 (Rank 32)
    # B_right is B[:, 4:6]
    # C_right is C[:, 4:6]
    # Schoolbook
    col_offset = rank_444
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_new[i*4+j][col_offset] = 1
                V_new[j*6 + (4+k)][col_offset] = 1
                W_new[(4+k)*4 + i][col_offset] = 1
                col_offset += 1
    
    # Verify
    if verify(U_new, V_new, W_new, total_rank, n, m, p):
        return {'U': U_new, 'V': V_new, 'W': W_new}
    else:
        return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
