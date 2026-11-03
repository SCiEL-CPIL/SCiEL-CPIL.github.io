import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def get_strassen(n):
    """
    Returns U, V, W, rank for n x n matrix multiplication using Strassen recursion.
    """
    if n == 1:
        return [[1]], [[1]], [[1]], 1
    
    s = n // 2
    U_s, V_s, W_s, r_s = get_strassen(s)
    
    total_rank = 7 * r_s
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    def add_block(M, row_off, col_off, Block, scale):
        for i in range(len(Block)):
            for j in range(len(Block[0])):
                M[row_off + i][col_off + j] += scale * Block[i][j]
    
    # P1 = (A00 + A11)(B00 + B11)
    for r in range(r_s):
        off = r
        add_block(U, 0, off, U_s[r], 1)
        add_block(U, s*s, off, U_s[r], 1)
        add_block(V, 0, off, V_s[r], 1)
        add_block(V, s*s, off, V_s[r], 1)
        add_block(W, 0, off, W_s[r], 1)
        add_block(W, s, off, W_s[r], -1)
        add_block(W, s*s, off, W_s[r], 1)
        add_block(W, s*s+s, off, W_s[r], -1)
        
    # P2 = (A10 + A11)B00
    for r in range(r_s):
        off = r_s + r
        add_block(U, s, off, U_s[r], 1)
        add_block(U, s*s, off, U_s[r], 1)
        add_block(V, 0, off, V_s[r], 1)
        add_block(W, s*s, off, W_s[r], 1)
        
    # P3 = A00(B11 - B10)
    for r in range(r_s):
        off = 2*r_s + r
        add_block(U, 0, off, U_s[r], 1)
        add_block(V, s*s, off, V_s[r], 1)
        add_block(V, s, off, V_s[r], -1)
        add_block(W, s*s+s, off, W_s[r], -1)
        
    # P4 = A11(B00 - B01)
    for r in range(r_s):
        off = 3*r_s + r
        add_block(U, s*s, off, U_s[r], 1)
        add_block(V, 0, off, V_s[r], 1)
        add_block(V, s, off, V_s[r], -1)
        add_block(W, 0, off, W_s[r], -1)
        add_block(W, s*s, off, W_s[r], 1)
        
    # P5 = (A00 + A01)B11
    for r in range(r_s):
        off = 4*r_s + r
        add_block(U, 0, off, U_s[r], 1)
        add_block(U, s, off, U_s[r], 1)
        add_block(V, s*s, off, V_s[r], 1)
        add_block(W, s, off, W_s[r], 1)
        add_block(W, s*s+s, off, W_s[r], 1)
        
    # P6 = (A10 - A00)(B00 + B01)
    for r in range(r_s):
        off = 5*r_s + r
        add_block(U, s, off, U_s[r], 1)
        add_block(U, 0, off, U_s[r], -1)
        add_block(V, 0, off, V_s[r], 1)
        add_block(V, s, off, V_s[r], 1)
        add_block(W, 0, off, W_s[r], 1)
        add_block(W, s*s+s, off, W_s[r], -1)
        
    # P7 = (A01 - A11)(B10 + B11)
    for r in range(r_s):
        off = 6*r_s + r
        add_block(U, s, off, U_s[r], 1)
        add_block(U, s*s, off, U_s[r], -1)
        add_block(V, s, off, V_s[r], 1)
        add_block(V, s*s, off, V_s[r], 1)
        add_block(W, 0, off, W_s[r], -1)
        add_block(W, s, off, W_s[r], 1)
        
    return U, V, W, total_rank

def get_rectangular(a, b, c):
    """
    Schoolbook decomposition for (a x c) * (c x b) -> (a x b).
    """
    rank = a * b * c
    U = [[0] * rank for _ in range(a * c)]
    V = [[0] * rank for _ in range(c * b)]
    W = [[0] * rank for _ in range(a * b)]
    
    for i in range(a):
        for j in range(c):
            for k in range(b):
                col = i * b * c + j * b + k
                U[i * c + j][col] = 1
                V[j * b + k][col] = 1
                W[i * b + k][col] = 1
                
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Block split 4 + 1
    block_sizes = [4, 1]
    num_blocks = 2
    
    row_offsets = [0, 4, 5]
    
    # Get 4x4x4 decomposition (Rank 49)
    U_4, V_4, W_4, r_4 = get_strassen(4)
    
    total_rank = 0
    # Pre-calculate ranks
    ranks = {}
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                a = block_sizes[i]
                b = block_sizes[j]
                c = block_sizes[k]
                if a == 4 and b == 4 and c == 4:
                    ranks[(i,j,k)] = r_4
                else:
                    ranks[(i,j,k)] = a * b * c
                total_rank += ranks[(i,j,k)]
                
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                a = block_sizes[i]
                b = block_sizes[j]
                c = block_sizes[k]
                
                if a == 4 and b == 4 and c == 4:
                    U_block, V_block, W_block = U_4, V_4, W_4
                else:
                    U_block, V_block, W_block, _ = get_rectangular(a, b, c)
                    
                r = ranks[(i,j,k)]
                
                a_row_off = row_offsets[i]
                a_col_off = row_offsets[k]
                b_row_off = row_offsets[k]
                b_col_off = row_offsets[j]
                c_row_off = row_offsets[i]
                c_col_off = row_offsets[j]
                
                for ri in range(a):
                    for ci in range(c):
                        local_row = ri * c + ci
                        global_row = (a_row_off + ri) * m + (a_col_off + ci)
                        for r_idx in range(r):
                            U_full[global_row][col_offset + r_idx] = U_block[local_row][r_idx]
                            
                for ri in range(c):
                    for ci in range(b):
                        local_row = ri * b + ci
                        global_row = (b_row_off + ri) * p + (b_col_off + ci)
                        for r_idx in range(r):
                            V_full[global_row][col_offset + r_idx] = V_block[local_row][r_idx]
                            
                for ri in range(a):
                    for ci in range(b):
                        local_row = ri * b + ci
                        global_row = (c_col_off + ci) * n + (c_row_off + ri)
                        for r_idx in range(r):
                            W_full[global_row][col_offset + r_idx] = W_block[local_row][r_idx]
                            
                col_offset += r
                
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                max_err = max(max_err, abs(s - 1))
                
    if max_err < 1e-10:
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
        return {"U": U_school, "V": V_school, "W": W_school}
