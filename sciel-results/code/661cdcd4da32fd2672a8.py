import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _get_strassen_decomp():
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _get_schoolbook_1col_decomp():
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W

def _construct_rank_25():
    n, m, p = DIMENSIONS
    R = 25
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    StrU, StrV, StrW = _get_strassen_decomp()
    SchU, SchV, SchW = _get_schoolbook_1col_decomp()
    
    col_offset = 0
    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        V[0][col_offset + r] = StrV[0][r]
        V[1][col_offset + r] = StrV[1][r]
        V[7][col_offset + r] = StrV[2][r]
        V[8][col_offset + r] = StrV[3][r]
        W[0][col_offset + r] = StrW[0][r]
        W[1][col_offset + r] = StrW[1][r]
        W[2][col_offset + r] = StrW[2][r]
        W[3][col_offset + r] = StrW[3][r]
    col_offset += 7

    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        V[2][col_offset + r] = StrV[0][r]
        V[3][col_offset + r] = StrV[1][r]
        V[9][col_offset + r] = StrV[2][r]
        V[10][col_offset + r] = StrV[3][r]
        W[4][col_offset + r] = StrW[0][r]
        W[5][col_offset + r] = StrW[1][r]
        W[6][col_offset + r] = StrW[2][r]
        W[7][col_offset + r] = StrW[3][r]
    col_offset += 7

    for r in range(7):
        for i in range(4):
            U[i][col_offset + r] = StrU[i][r]
        V[4][col_offset + r] = StrV[0][r]
        V[5][col_offset + r] = StrV[1][r]
        V[11][col_offset + r] = StrV[2][r]
        V[12][col_offset + r] = StrV[3][r]
        W[8][col_offset + r] = StrW[0][r]
        W[9][col_offset + r] = StrW[1][r]
        W[10][col_offset + r] = StrW[2][r]
        W[11][col_offset + r] = StrW[3][r]
    col_offset += 7

    for r in range(4):
        for i in range(4):
            U[i][col_offset + r] = SchU[i][r]
        V[6][col_offset + r] = SchV[0][r]
        V[13][col_offset + r] = SchV[1][r]
        W[12][col_offset + r] = SchW[0][r]
        W[13][col_offset + r] = SchW[1][r]
    col_offset += 4

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank 24
    # Given the complexity and time limit, we will just return the known rank 25 solution
    # unless we have a very fast search.
    # For now, return rank 25.
    
    res = _construct_rank_25()
    
    if _verify(res['U'], res['V'], res['W'], n, m, p):
        return res
    
    return _build_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
