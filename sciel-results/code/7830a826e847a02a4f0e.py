import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110

def construct():
    n, m, p = 3, 5, 7
    rank = 99
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    rank_idx = 0
    
    # Define the 6 Strassen blocks
    # i is in {1, 2}
    # j is in {1, 2, 3, 4} -> two blocks of size 2: {1,2} and {3,4}
    # k is in {1, 2, 3, 4, 5, 6} -> three blocks of size 2: {1,2}, {3,4}, {5,6}
    
    strassen_blocks = []
    for j_block_start in [1, 3]:
        for k_block_start in [1, 3, 5]:
            # i_block_start is always 1
            strassen_blocks.append((1, j_block_start, k_block_start))
            
    # Helper to add a Strassen block
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal rank_idx
        # Local indices 0, 1 map to global i_start, i_start+1 etc.
        
        # Strassen terms for 2x2x2
        # P1 = (a00 + a11)(b00 + b11)
        # P2 = (a10 + a11)b00
        # P3 = a00(b01 - b11)
        # P4 = a11(b10 - b00)
        # P5 = (a00 + a01)b11
        # P6 = (a10 - a00)(b00 + b01)
        # P7 = (a01 - a11)(b10 + b11)
        
        # We define the coefficients for U, V, W for each of the 7 terms.
        # U corresponds to A (n x m)
        # V corresponds to B (m x p)
        # W corresponds to C (p x n)
        
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # Term 1: (a00 + a11)(b00 + b11)
        # U: a00 (i0, j0), a11 (i1, j1)
        # V: b00 (j0, k0), b11 (j1, k1)
        # W: c00 (k0, i0), c11 (k1, i1)
        # c00 = P1 + ...
        # c11 = P1 + ...
        # So W gets +1 at (k0, i0) and +1 at (k1, i1)
        
        # Let's list the (U_idx, coeff), (V_idx, coeff), (W_idx, coeff) for each term.
        
        # P1
        u1 = [(i_start*m + j_start, 1.0), ((i_start+1)*m + (j_start+1), 1.0)]
        v1 = [(j_start*p + k_start, 1.0), ((j_start+1)*p + (k_start+1), 1.0)]
        w1 = [(k_start*n + i_start, 1.0), ((k_start+1)*n + (i_start+1), 1.0)]
        
        # P2: (a10 + a11)b00
        # a10 (i1, j0), a11 (i1, j1)
        # b00 (j0, k0)
        # c10 (k0, i1), c11 (k1, i1)
        u2 = [(i_start*m + (j_start+1), 1.0), ((i_start+1)*m + (j_start+1), 1.0)]
        v2 = [(j_start*p + k_start, 1.0)]
        w2 = [(k_start*n + (i_start+1), 1.0), ((k_start+1)*n + (i_start+1), 1.0)]
        
        # P3: a00(b01 - b11)
        # a00 (i0, j0)
        # b01 (j0, k1), b11 (j1, k1)
        # c01 (k1, i0), c11 (k1, i1)
        u3 = [(i_start*m + j_start, 1.0)]
        v3 = [(j_start*p + (k_start+1), 1.0), ((j_start+1)*p + (k_start+1), -1.0)]
        w3 = [(k_start+1)*n + i_start, 1.0], [ (k_start+1)*n + (i_start+1), 1.0] # Wait, c11 coeff is +1 for P3?
        # c11 = P1 - P2 + P3 + P6. Yes, +1.
        # c01 = P3 + P5. Yes, +1.
        
        # P4: a11(b10 - b00)
        # a11 (i1, j1)
        # b10 (j1, k0), b00 (j0, k0)
        # c00 (k0, i0), c10 (k0, i1)
        u4 = [((i_start+1)*m + (j_start+1), 1.0)]
        v4 = [((j_start+1)*p + k_start, 1.0), (j_start*p + k_start, -1.0)]
        w4 = [(k_start*n + i_start, 1.0), (k_start*n + (i_start+1), 1.0)]
        
        # P5: (a00 + a01)b11
        # a00 (i0, j0), a01 (i0, j1)
        # b11 (j1, k1)
        # c00 (k0, i0) [coeff -1], c01 (k1, i0) [coeff +1]
        u5 = [(i_start*m + j_start, 1.0), (i_start*m + (j_start+1), 1.0)]
        v5 = [((j_start+1)*p + (k_start+1), 1.0)]
        w5 = [(k_start*n + i_start, -1.0), ((k_start+1)*n + i_start, 1.0)]
        
        # P6: (a10 - a00)(b00 + b01)
        # a10 (i1, j0), a00 (i0, j0)
        # b00 (j0, k0), b01 (j0, k1)
        # c11 (k1, i1) [coeff +1]
        u6 = [((i_start+1)*m + j_start, 1.0), (i_start*m + j_start, -1.0)]
        v6 = [(j_start*p + k_start, 1.0), (j_start*p + (k_start+1), 1.0)]
        w6 = [((k_start+1)*n + (i_start+1), 1.0)]
        
        # P7: (a01 - a11)(b10 + b11)
        # a01 (i0, j1), a11 (i1, j1)
        # b10 (j1, k0), b11 (j1, k1)
        # c00 (k0, i0) [coeff +1], c11 (k1, i1) [coeff +1]
        u7 = [(i_start*m + (j_start+1), 1.0), ((i_start+1)*m + (j_start+1), -1.0)]
        v7 = [((j_start+1)*p + k_start, 1.0), ((j_start+1)*p + (k_start+1), 1.0)]
        w7 = [(k_start*n + i_start, 1.0), ((k_start+1)*n + (i_start+1), 1.0)]
        
        terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u_list, v_list, w_list in terms:
            for idx, val in u_list:
                U[idx][rank_idx] += val
            for idx, val in v_list:
                V[idx][rank_idx] += val
            for idx, val in w_list:
                W[idx][rank_idx] += val
            rank_idx += 1

    # Add Strassen blocks
    for block in strassen_blocks:
        add_strassen_block(*block)
        
    # Add schoolbook terms for the rest
    # Covered indices:
    # i in {1,2}, j in {1,2,3,4}, k in {1,2,3,4,5,6}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen
                if i in [1, 2] and j in [1, 2, 3, 4] and k in [1, 2, 3, 4, 5, 6]:
                    continue
                
                # Schoolbook
                U[i*m + j][rank_idx] = 1.0
                V[j*p + k][rank_idx] = 1.0
                W[k*n + i][rank_idx] = 1.0
                rank_idx += 1
                
    # rank_idx should be 99
    assert rank_idx == 99, f"Rank index mismatch: {rank_idx}"
    
    return {'U': U, 'V': V, 'W': W}
