import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Rank 50 solution based on Strassen blocks for 2x2x2 and schoolbook for 2x2x1.
    # Structure:
    # Split m=4 into two blocks of size 2.
    # For each block, split p=7 into three blocks of size 2 and one of size 1.
    # Each 2x2x2 block uses rank 7.
    # Each 2x2x1 block uses rank 4.
    # Total per m-block: 3*7 + 4 = 25.
    # Total rank: 2 * 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_block(i_offset, j_offset, k_offset, block_type):
        nonlocal col
        # Indices in the global matrices
        # A is n x m. Index i * m + j.
        # B is m x p. Index j * p + k.
        # C is n x p. Index k * n + i (Trace convention W[k*n+i, r])
        
        if block_type == 'strassen':
            # 2x2x2 block
            # A submatrix rows i_offset, i_offset+1; cols j_offset, j_offset+1
            # B submatrix rows j_offset, j_offset+1; cols k_offset, k_offset+1
            # C submatrix rows i_offset, i_offset+1; cols k_offset, k_offset+1
            
            a = i_offset * m + j_offset
            b = i_offset * m + (j_offset + 1)
            c = (i_offset + 1) * m + j_offset
            d = (i_offset + 1) * m + (j_offset + 1)
            
            e = j_offset * p + k_offset
            f = j_offset * p + (k_offset + 1)
            g = (j_offset + 1) * p + k_offset
            h = (j_offset + 1) * p + (k_offset + 1)
            
            c00 = k_offset * n + i_offset
            c01 = (k_offset + 1) * n + i_offset
            c10 = k_offset * n + (i_offset + 1)
            c11 = (k_offset + 1) * n + (i_offset + 1)
            
            # M1 = (a+d)(e+h)
            U[a][col] += 1; U[d][col] += 1
            V[e][col] += 1; V[h][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[c][col] += 1; U[d][col] += 1
            V[e][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[a][col] += 1
            V[f][col] += 1; V[h][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[d][col] += 1
            V[g][col] += 1; V[e][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[a][col] += 1; U[b][col] += 1
            V[h][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[c][col] += 1; U[a][col] -= 1
            V[e][col] += 1; V[f][col] += 1
            W[c11][col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[b][col] += 1; U[d][col] -= 1
            V[g][col] += 1; V[h][col] += 1
            W[c00][col] += 1
            col += 1
            
        elif block_type == 'schoolbook_221':
            # 2x2x1 block
            # A submatrix 2x2, B submatrix 2x1, C submatrix 2x1
            
            a = i_offset * m + j_offset
            b = i_offset * m + (j_offset + 1)
            c = (i_offset + 1) * m + j_offset
            d = (i_offset + 1) * m + (j_offset + 1)
            
            e = j_offset * p + k_offset
            g = (j_offset + 1) * p + k_offset
            
            c00 = k_offset * n + i_offset
            c10 = k_offset * n + (i_offset + 1)
            
            # a*e
            U[a][col] += 1
            V[e][col] += 1
            W[c00][col] += 1
            col += 1
            
            # b*g
            U[b][col] += 1
            V[g][col] += 1
            W[c00][col] += 1
            col += 1
            
            # c*e
            U[c][col] += 1
            V[e][col] += 1
            W[c10][col] += 1
            col += 1
            
            # d*g
            U[d][col] += 1
            V[g][col] += 1
            W[c10][col] += 1
            col += 1

    # Iterate over the two m-blocks (j=0..1 and j=2..3)
    for j_start in [0, 2]:
        # Iterate over the p-blocks (k=0..1, 2..3, 4..5, 6..6)
        for k_start in [0, 2, 4]:
            add_block(0, j_start, k_start, 'strassen')
        add_block(0, j_start, 6, 'schoolbook_221')
        
    return {"U": U, "V": V, "W": W}
