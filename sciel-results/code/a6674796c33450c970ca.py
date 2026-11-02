"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition correctness."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return U, V, W for 2x2 @ 2x2 matrix multiplication with Strassen's 7 terms.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[1][0] = 1.0; W[2][0] = 1.0; W[3][0] = 1.0
    
    # M2 = (c+d)e
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3 = a(f-h)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0
    
    # M4 = d(g-e)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[2][3] = 1.0
    
    # M5 = (a+b)h
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[1][4] = 1.0
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W


def _strassen_4x4_4x4():
    """
    Return U, V, W for 4x4 @ 4x4 using nested Strassen (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    rank = 49
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(16)]
    W = [[0.0]*rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block-level Strassen coefficients
    A_blocks = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    C_contrib = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    for t_b in range(7):
        for t_s in range(7):
            t = t_b * 7 + t_s
            
            for (bi, bj, ca) in A_blocks[t_b]:
                for i_l in range(2):
                    for j_l in range(2):
                        us = U2[i_l*2+j_l][t_s]
                        if us != 0:
                            U[bi*2+i_l][bj*2+j_l][t] += ca * us
            
            for (bj, bk, cb) in B_blocks[t_b]:
                for j_l in range(2):
                    for k_l in range(2):
                        vs = V2[j_l*2+k_l][t_s]
                        if vs != 0:
                            V[bj*2+j_l][bk*2+k_l][t] += cb * vs
            
            for (bi, bk) in C_contrib:
                for (tb_c, cc) in C_contrib[(bi, bk)]:
                    if tb_c == t_b:
                        for i_l in range(2):
                            for k_l in range(2):
                                ws = W2[k_l*2+i_l][t_s]
                                if ws != 0:
                                    W[bk*2+k_l][bi*2+i_l][t] += cc * ws
    
    return U, V, W


def _construct_4x4_5():
    """
    Construct 4x4 @ 4x5 decomposition.
    Use rank-49 for 4x4@4x4 part, plus 16 terms for 4x4@4x1 part.
    Total rank: 65.
    """
    n, m, p = 4, 4, 5
    U44, V44, W44 = _strassen_4x4_4x4()
    
    if not _verify_decomposition(U44, V44, W44, 4, 4, 4):
        return _schoolbook(n, m, p)
    
    rank = 49 + 16
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(20)]
    W = [[0.0]*rank for _ in range(20)]
    
    for i in range(4):
        for j in range(4):
            for t in range(49):
                U[i*4+j][t] = U44[i*4+j][t]
    for j in range(4):
        for k in range(4):
            for t in range(49):
                V[j*5+k][t] = V44[j*4+k][t]
    for k in range(4):
        for i in range(4):
            for t in range(49):
                W[k*4+i][t] = W44[k*4+i][t]
    
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1.0
            V[j*5+4][col] = 1.0
            W[4*4+i][col] = 1.0
            col += 1
    
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_iters=2000, time_budget=60):
    start_time = time.time()
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = len(U[0])
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        if best_rank <= 20:
            break
        
        t_remove = random.randint(0, best_rank - 1)
        
        is_zero = True
        for row in best_U:
            if row[t_remove] != 0:
                is_zero = False
                break
        if is_zero:
            for row in best_V:
                if row[t_remove] != 0:
                    is_zero = False
                    break
        if is_zero:
            for row in best_W:
                if row[t_remove] != 0:
                    is_zero = False
                    break
        if is_zero:
            best_U = [row[:t_remove] + row[t_remove+1:] for row in best_U]
            best_V = [row[:t_remove] + row[t_remove+1:] for row in best_V]
            best_W = [row[:t_remove] + row[t_remove+1:] for row in best_W]
            best_rank -= 1
            continue
        
        if best_rank > 1:
            t1 = random.randint(0, best_rank - 1)
            t2 = random.randint(0, best_rank - 1)
            if t1 == t2:
                continue
            
            a, b, c = None, None, None
            valid = True
            for idx in range(len(best_U)):
                if best_U[idx][t1] != 0:
                    a = best_U[idx][t2] / best_U[idx][t1]
                    break
            if a is None:
                for idx in range(len(best_U)):
                    if best_U[idx][t2] != 0:
                        valid = False
                        break
            if valid and a is not None:
                for idx in range(len(best_V)):
                    if best_V[idx][t1] != 0:
                        b = best_V[idx][t2] / best_V[idx][t1]
                        break
                if b is None:
                    valid = False
            if valid and b is not None:
                for idx in range(len(best_W)):
                    if best_W[idx][t1] != 0:
                        c = best_W[idx][t2] / best_W[idx][t1]
                        break
                if c is None:
                    valid = False
            if valid and a is not None and b is not None and c is not None:
                if abs(a*b*c - 1.0) < 1e-9:
                    best_U = [row[:t2] + row[t2+1:] for row in best_U]
                    best_V = [row[:t2] + row[t2+1:] for row in best_V]
                    best_W = [row[:t2] + row[t2+1:] for row in best_W]
                    best_rank -= 1
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    U, V, W = _construct_4x4_5()
    
    if _verify_decomposition(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
    else:
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
    
    if time.time() - start_time < deadline - 5:
        remaining = min(50, deadline - (time.time() - start_time) - 5)
        U_r, V_r, W_r, rank_r = _try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                                   max_iters=5000, time_budget=remaining)
        if rank_r < best_rank and _verify_decomposition(U_r, V_r, W_r, n, m, p):
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r
    
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END

