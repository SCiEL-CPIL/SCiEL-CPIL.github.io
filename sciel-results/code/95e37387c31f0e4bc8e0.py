import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _build_decomposition():
    """
    Build a rank-83 decomposition for 4x4x7 matrix multiplication.
    
    Strategy: 
    - Rank-49 Strassen for the first 4 columns (4x4x4)
    - Rank-34 for the remaining 3 columns (4x4x3)
    
    Total rank = 83
    """
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 coefficients
    u_in = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    u_out = u_in[:]
    v_out = v_in[:]
    w_outer = w_out[:]
    
    rank_strassen = 49
    rank_tail = 34
    total_rank = rank_strassen + rank_tail
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x rank
    
    # Part 1: Strassen 4x4x4 for the first 4 columns of the 4x7 output
    col = 0
    for k_outer in range(4):  # only first 4 columns
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        ig = 2 * br + i_inner
                        jg = 2 * bc + j_inner
                        u_vec[ig * 4 + jg] += ci * co
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        jg = 2 * br + i_inner
                        kg = 2 * bc + j_inner
                        # Only contribute to first 4 columns of B (k=0..3)
                        if kg < 4:
                            v_vec[jg * 7 + kg] += ci * co
            
            w_vec = [0] * 28
            for cb in range(4):
                cow = w_outer[cb][k_outer]
                if cow == 0:
                    continue
                for ci2 in range(4):
                    ciw = w_out[ci2][l_inner]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    cbr, cbc = cb // 2, cb % 2
                    cir, cic = ci2 // 2, ci2 % 2
                    ig = 2 * cbr + cir
                    kg = 2 * cbc + cic
                    w_vec[kg * 4 + ig] += tc
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            col += 1
    
    # Part 2: Rank-34 decomposition for the 4x4x3 block (columns 4,5,6 of B)
    # We need to compute C[:, 4:7] = A * B[:, 4:7]
    # A is 4x4, B_tail is 4x3
    # 
    # Construction: Partition into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B_tail = [[B1], [B2]] where B1, B2 are 2x3
    # C_tail = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    # For each 2x2 * 2x3 product, use rank 7 construction:
    # A2 = [[a, b], [c, d]], B2 = [[x1,x2,x3],[y1,y2,y3]]
    # 7 multiplications:
    # p1 = (a+b)*(x1+y1)
    # p2 = (a+b)*(x2+y2)
    # p3 = (a+b)*(x3+y3)
    # p4 = c*x1
    # p5 = c*x2
    # p6 = c*x3
    # p7 = d*(y1+y2+y3) -- this isn't right
    
    # Correct rank-7 for 2x2x3:
    # p1 = (a+d)*(x1+y1)
    # p2 = (a+d)*(x2+y2)
    # p3 = (a+d)*(x3+y3)
    # p4 = (c+d)*(x1+y1)
    # p5 = (c+d)*(x2+y2)
    # p6 = (c+d)*(x3+y3)
    # p7 = a*x1 + b*y1 -- no, must be scalar
    
    # Let me use a known valid rank-7 construction:
    # Actually, let me just use schoolbook for now (rank 6 per 2x2x3)
    # 4 blocks * 6 = 24, but we need 34 total for the 4x4x3
    
    # Simpler approach: use schoolbook for the tail with rank 48
    # Total rank = 49 + 48 = 97
    
    # Rebuild with rank 97
    rank_tail_actual = 4 * 4 * 3  # 48
    total_rank_actual = rank_strassen + rank_tail_actual  # 97
    
    U = [[0] * total_rank_actual for _ in range(n * m)]
    V = [[0] * total_rank_actual for _ in range(m * p)]
    W = [[0] * total_rank_actual for _ in range(p * n)]
    
    col = 0
    for k_outer in range(4):
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        ig = 2 * br + i_inner
                        jg = 2 * bc + j_inner
                        u_vec[ig * 4 + jg] += ci * co
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        jg = 2 * br + i_inner
                        kg = 2 * bc + j_inner
                        if kg < 4:
                            v_vec[jg * 7 + kg] += ci * co
            
            w_vec = [0] * 28
            for cb in range(4):
                cow = w_outer[cb][k_outer]
                if cow == 0:
                    continue
                for ci2 in range(4):
                    ciw = w_out[ci2][l_inner]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    cbr, cbc = cb // 2, cb % 2
                    cir, cic = ci2 // 2, ci2 % 2
                    ig = 2 * cbr + cir
                    kg = 2 * cbc + cic
                    w_vec[kg * 4 + ig] += tc
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            col += 1
    
    # Schoolbook for columns 4,5,6
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

_DECOMP = _build_decomposition()

def construct():
    """
    Construct a rank-97 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    """
    return _DECOMP
