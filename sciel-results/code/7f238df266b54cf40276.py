"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-minimal decomposition using Strassen for 2x2 blocks
    # 7 blocks of 2x2 (14 columns) + 1 schoolbook column = 7*7 + 4 = 53 rank
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # Strassen 2x2 decomposition vectors
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 - M2 + M4 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 - M6
    
    # U vectors for A entries (a=0, b=1, c=2, d=3)
    strassen_U = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 1, 0, 0],  # M3: a (for f-h term, but U is just a)
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [1, 0, -1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    # W vectors for C entries (C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    # But W index is k*n+i, so for a 2x2 block at columns 2b, 2b+1:
    # C[k, 2b+0] -> index (2b+0)*2 + i, C[k, 2b+1] -> index (2b+1)*2 + i
    # Wait, the output is C (n x p) = A (n x m) * B (m x p)
    # C has shape (n, p) = (2, 15)
    # W has shape (p*n, rank) = (30, rank)
    # W[k*n + i, c] corresponds to C[i, k]
    
    # For block b (columns 2b, 2b+1 of B):
    # B entries: B[0, 2b]=e, B[0, 2b+1]=f, B[1, 2b]=g, B[1, 2b+1]=h
    # V indices: j*p+k
    #   e: 0*15 + 2b = 2b
    #   f: 0*15 + (2b+1) = 2b+1
    #   g: 1*15 + 2b = 15+2b
    #   h: 1*15 + (2b+1) = 15+2b+1
    
    # C entries from this block:
    # C[0, 2b] = M1 - M2 + M4 + M7
    # C[0, 2b+1] = M3 + M5
    # C[1, 2b] = M2 + M4
    # C[1, 2b+1] = M1 - M3 + M5 - M6
    
    # W indices:
    # C[0, 2b] -> W[(2b)*2 + 0] = 4b
    # C[0, 2b+1] -> W[(2b+1)*2 + 0] = 4b+2
    # C[1, 2b] -> W[(2b)*2 + 1] = 4b+1
    # C[1, 2b+1] -> W[(2b+1)*2 + 1] = 4b+3
    
    # W vectors (for each of the 7 Strassen terms, what to add to each C entry):
    # M1: +1 to C[0,2b], +1 to C[1,2b+1]
    # M2: -1 to C[0,2b], +1 to C[1,2b]
    # M3: +1 to C[0,2b+1], -1 to C[1,2b+1]
    # M4: +1 to C[0,2b], +1 to C[1,2b]
    # M5: +1 to C[0,2b+1], -1 to C[1,2b+1]
    # M6: -1 to C[1,2b+1]
    # M7: +1 to C[0,2b]
    
    strassen_W_coeffs = [
        # (C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1])
        [1, 0, 0, 1],   # M1
        [-1, 0, 1, 0],  # M2
        [0, 1, 0, -1],  # M3
        [1, 0, 1, 0],   # M4
        [0, 1, 0, -1],  # M5
        [0, 0, 0, -1],  # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # V vectors for each Strassen term (for a block at columns 2b, 2b+1):
    # M1: (e+h) -> e + h
    # M2: (c+d)*e -> e
    # M3: a*(f-h) -> f - h
    # M4: d*(g-e) -> g - e
    # M5: (a+b)*h -> h
    # M6: (c-a)*(e+f) -> e + f
    # M7: (b-d)*(g+h) -> g + h
    
    def set_V(block_b, term_idx, col_offset):
        """Set V entries for a Strassen term in block_b."""
        # V indices: j*15 + k
        e_idx = 0 * 15 + (2 * block_b)
        f_idx = 0 * 15 + (2 * block_b + 1)
        g_idx = 1 * 15 + (2 * block_b)
        h_idx = 1 * 15 + (2 * block_b + 1)
        
        V_entries = {
            0: [(e_idx, 1), (h_idx, 1)],       # e + h
            1: [(e_idx, 1)],                    # e
            2: [(f_idx, 1), (h_idx, -1)],      # f - h
            3: [(g_idx, 1), (e_idx, -1)],      # g - e
            4: [(h_idx, 1)],                    # h
            5: [(e_idx, 1), (f_idx, 1)],       # e + f
            6: [(g_idx, 1), (h_idx, 1)],       # g + h
        }
        
        for idx, val in V_entries[term_idx]:
            V[idx][col_offset] += val
    
    col = 0
    # 7 blocks of 2x2
    for block_b in range(7):
        for t in range(7):
            # U
            for i in range(4):
                U[i][col] += strassen_U[t][i]
            
            # V
            set_V(block_b, t, col)
            
            # W
            # C[0,2b] -> 4*block_b
            # C[0,2b+1] -> 4*block_b + 2
            # C[1,2b] -> 4*block_b + 1
            # C[1,2b+1] -> 4*block_b + 3
            w_base = 4 * block_b
            W[w_base + 0][col] += strassen_W_coeffs[t][0]   # C[0,2b]
            W[w_base + 2][col] += strassen_W_coeffs[t][1]   # C[0,2b+1]
            W[w_base + 1][col] += strassen_W_coeffs[t][2]   # C[1,2b]
            W[w_base + 3][col] += strassen_W_coeffs[t][3]   # C[1,2b+1]
            
            col += 1
    
    # Last column (column 14) - schoolbook
    # A = [[a, b], [c, d]], B[:, 14] = [x, y]^T
    # C[0, 14] = a*x + b*y
    # C[1, 14] = c*x + d*y
    # 4 terms:
    # T1: a*x -> C[0,14]
    # T2: b*y -> C[0,14]
    # T3: c*x -> C[1,14]
    # T4: d*y -> C[1,14]
    
    x_idx = 0 * 15 + 14
    y_idx = 1 * 15 + 14
    c0_14 = 14 * 2 + 0  # C[0,14] -> W index
    c1_14 = 14 * 2 + 1  # C[1,14] -> W index
    
    schoolbook_terms = [
        # (U_a, U_b, U_c, U_d, V_x, V_y, W_c0, W_c1)
        [1, 0, 0, 0, 1, 0, 1, 0],  # a*x
        [0, 1, 0, 0, 0, 1, 1, 0],  # b*y
        [0, 0, 1, 0, 1, 0, 0, 1],  # c*x
        [0, 0, 0, 1, 0, 1, 0, 1],  # d*y
    ]
    
    for term in schoolbook_terms:
        U[0][col] += term[0]
        U[1][col] += term[1]
        U[2][col] += term[2]
        U[3][col] += term[3]
        V[x_idx][col] += term[4]
        V[y_idx][col] += term[5]
        W[c0_14][col] += term[6]
        W[c1_14][col] += term[7]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
