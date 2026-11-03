import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Total rank
    # Part 1: M_{2,1,6} for j=0. Rank 12.
    # Part 2: M_{2,2,6} for j=1,2. Rank 21 (3 * 7).
    total_rank = 12 + 21
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # Part 1: M_{2,1,6}
    # A indices: i in {0,1}, j=0. U index: i*3 + 0 = i.
    # B indices: j=0, k in {0..5}. V index: 0*6 + k = k.
    # C indices: k in {0..5}, i in {0,1}. W index: k*2 + i.
    for i in range(n):
        for k in range(p):
            U[i][col] = 1
            V[k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Part 2: M_{2,2,6}
    # A indices: i in {0,1}, j in {1,2}.
    # Sub-problem j' in {0,1} maps to original j = j' + 1.
    # U index in original: i*3 + (j' + 1).
    # B indices: j in {1,2}, k in {0..5}.
    # Sub-problem j' in {0,1} maps to original j = j' + 1.
    # V index in original: (j' + 1)*6 + k.
    # C indices: k in {0..5}, i in {0,1}.
    # W index in original: k*2 + i.
    
    # Split k into 3 groups of 2: k in {0,1}, {2,3}, {4,5}
    # For each group, use Strassen M_{2,2,2}
    
    def add_strassen_block(k_offset):
        nonlocal col
        # Strassen coefficients
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # A entries in sub-problem:
        # a = A[0, j'=0] -> orig U[0*3 + 1] = U[1]
        # b = A[0, j'=1] -> orig U[0*3 + 2] = U[2]
        # c = A[1, j'=0] -> orig U[1*3 + 1] = U[4]
        # d = A[1, j'=1] -> orig U[1*3 + 2] = U[5]
        
        # B entries in sub-problem:
        # e = B[j'=0, k=k_offset+0] -> orig V[1*6 + k_offset] = V[6+k_offset]
        # f = B[j'=0, k=k_offset+1] -> orig V[1*6 + k_offset+1] = V[7+k_offset]
        # g = B[j'=1, k=k_offset+0] -> orig V[2*6 + k_offset] = V[12+k_offset]
        # h = B[j'=1, k=k_offset+1] -> orig V[2*6 + k_offset+1] = V[13+k_offset]
        
        # C entries in sub-problem (and original):
        # C[0, k_offset+0] -> orig W[(k_offset+0)*2 + 0] = W[2*k_offset]
        # C[0, k_offset+1] -> orig W[(k_offset+1)*2 + 0] = W[2*k_offset+2]
        # C[1, k_offset+0] -> orig W[(k_offset+0)*2 + 1] = W[2*k_offset+1]
        # C[1, k_offset+1] -> orig W[(k_offset+1)*2 + 1] = W[2*k_offset+3]
        
        u_a = 1; u_b = 2; u_c = 4; u_d = 5
        v_e = 6 + k_offset; v_f = 7 + k_offset
        v_g = 12 + k_offset; v_h = 13 + k_offset
        
        w_00 = 2 * k_offset
        w_01 = 2 * k_offset + 2
        w_10 = 2 * k_offset + 1
        w_11 = 2 * k_offset + 3
        
        # M1
        U[u_a][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1; V[v_h][col] = 1
        W[w_00][col] = 1; W[w_11][col] = 1
        col += 1
        
        # M2
        U[u_c][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1
        W[w_10][col] = 1; W[w_11][col] = -1
        col += 1
        
        # M3
        U[u_a][col] = 1
        V[v_f][col] = 1; V[v_h][col] = -1
        W[w_01][col] = 1; W[w_11][col] = 1
        col += 1
        
        # M4
        U[u_d][col] = 1
        V[v_g][col] = 1; V[v_e][col] = -1
        W[w_00][col] = 1; W[w_10][col] = 1
        col += 1
        
        # M5
        U[u_a][col] = 1; U[u_b][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = -1; W[w_01][col] = 1
        col += 1
        
        # M6
        U[u_c][col] = 1; U[u_a][col] = -1
        V[v_e][col] = 1; V[v_f][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M7
        U[u_b][col] = 1; U[u_d][col] = -1
        V[v_g][col] = 1; V[v_h][col] = 1
        W[w_00][col] = 1
        col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    return {"U": U, "V": V, "W": W}
