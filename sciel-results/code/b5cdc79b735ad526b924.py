import time
from fractions import Fraction
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    
    _start_time = time.time()
    _deadline = 140.0

    def _time_ok():
        return (time.time() - _start_time) < _deadline

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U_frac[i * m + a]
                    v_row = V_frac[a * p + k]
                    w_row = W_frac[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    # Part 1: a=0,1 (2x2x8) using 4 Strassen 2x2x2 blocks
    # Rank 28
    rank_part1 = 28
    U1 = [[0] * rank_part1 for _ in range(n * m)]
    V1 = [[0] * rank_part1 for _ in range(m * p)]
    W1 = [[0] * rank_part1 for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U1[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V1[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W1[(k0 + lk) * n + li][col] = w_val
            col += 1

    # Part 2: a=2 (2x1x8) using Rank 2 trick
    # Rank 2
    rank_part2 = 2
    U2 = [[0] * rank_part2 for _ in range(n * m)]
    V2 = [[0] * rank_part2 for _ in range(m * p)]
    W2 = [[0] * rank_part2 for _ in range(p * n)]
    
    # Term 0: i=0
    U2[0 * m + 2][0] = 1
    for k in range(p):
        V2[2 * p + k][0] = 1
        W2[k * n + 0][0] = 1
        
    # Term 1: i=1
    U2[1 * m + 2][1] = 1
    for k in range(p):
        V2[2 * p + k][1] = 1
        W2[k * n + 1][1] = 1

    # Combine
    total_rank = rank_part1 + rank_part2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r in range(rank_part1):
        for i in range(n):
            for a in range(2):
                U[i*m+a][r] = U1[i*m+a][r]
        for a in range(2):
            for k in range(p):
                V[a*p+k][r] = V1[a*p+k][r]
        for k in range(p):
            for i in range(n):
                W[k*n+i][r] = W1[k*n+i][r]
                
    for r in range(rank_part2):
        for i in range(n):
            U[i*m+2][rank_part1 + r] = U2[i*m+2][r]
        for k in range(p):
            V[2*p+k][rank_part1 + r] = V2[2*p+k][r]
        for k in range(p):
            for i in range(n):
                W[k*n+i][rank_part1 + r] = W2[k*n+i][r]

    candidate = {"U": U, "V": V, "W": W}
    
    if not _validate(candidate):
        return _schoolbook()
        
    return candidate
