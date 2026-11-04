import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            lhs = 0
                            for c in range(rank):
                                lhs += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            rhs = (1 if j == j2 and k == k2 else 0)
                            if abs(lhs - rhs) > 1e-10:
                                return False
    return True

def _strassen_2x2_2x2():
    U = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, -1, 0, 0, 1, 1],
        [1, 0, 0, -1, 1, 0, 0]
    ]
    W = [
        [1, 0, 1, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return {"U": U, "V": V, "W": W}

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    U = [[0] * (rank1 + rank2) for _ in range(len(U1))]
    V = [[0] * (rank1 + rank2) for _ in range(len(V1))]
    W = [[0] * (rank1 + rank2) for _ in range(len(W1))]
    for r in range(rank1):
        for i in range(len(U1)):
            U[i][r] = U1[i][r]
        for i in range(len(V1)):
            V[i][r] = V1[i][r]
        for i in range(len(W1)):
            W[i][r] = W1[i][r]
    for r in range(rank2):
        for i in range(len(U2)):
            U[i][rank1 + r] = U2[i][r]
        for i in range(len(V2)):
            V[i][rank1 + r] = V2[i][r]
        for i in range(len(W2)):
            W[i][rank1 + r] = W2[i][r]
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    if time.time() > deadline:
        return best
    
    # Try to construct a better decomposition using Strassen for 4x4x4 part and schoolbook for 4x4x3 part
    # A (4x4) * B (4x7) = A * [B_left (4x4), B_right (4x3)]
    # We can compute A * B_left using Strassen (rank 49) and A * B_right using schoolbook (rank 48)
    # Total rank = 49 + 48 = 97
    
    # Strassen 4x4x4:
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B_left into 2x2 blocks: B_left = [[B00, B01], [B10, B11]]
    # C_left = A * B_left = [[C00, C01], [C10, C11]]
    # Each block product is 2x2 * 2x2 -> 2x2, using Strassen rank 7
    # Total rank for 4x4x4 is 7*7 = 49
    
    # Schoolbook 4x4x3:
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B_right into 2x2 blocks: B_right = [[B00, B01, B02], [B10, B11, B12]] where B02, B12 are 2x1
    # Actually, let's just use schoolbook for the whole 4x4x3 part: rank 4*4*3 = 48
    
    # Let's construct the combined decomposition
    # U_total: (16, 97), V_total: (28, 97), W_total: (28, 97)
    
    # For the Strassen part (4x4x4):
    # U_strassen: (16, 49), V_strassen: (16, 49), W_strassen: (16, 49)
    # For the schoolbook part (4x4x3):
    # U_school: (16, 48), V_school: (12, 48), W_school: (12, 48)
    
    # But we need to align the dimensions:
    # U_total[i, c] = U_strassen[i, c] for c < 49, U_school[i, c-49] for c >= 49
    # V_total[j, c] = V_strassen[j, c] for c < 49, V_school[j, c-49] for c >= 49 (only for j < 16)
    # W_total[k, c] = W_strassen[k, c] for c < 49, W_school[k, c-49] for c >= 49 (only for k < 12)
    
    # Let's construct Strassen 4x4x4
    def _strassen_4x4_4x4():
        U2 = _strassen_2x2_2x2()["U"]
        V2 = _strassen_2x2_2x2()["V"]
        W2 = _strassen_2x2_2x2()["W"]
        
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        
        for i_outer in range(4):
            for i_inner in range(4):
                i = i_outer * 4 + i_inner
                for c_outer in range(7):
                    for c_inner in range(7):
                        c = c_outer * 7 + c_inner
                        U4[i][c] = U2[i_outer][c_outer] * U2[i_inner][c_inner]
                        V4[i][c] = V2[i_outer][c_outer] * V2[i_inner][c_inner]
                        W4[i][c] = W2[i_outer][c_outer] * W2[i_inner][c_inner]
        
        return {"U": U4, "V": V4, "W": W4}
    
    strassen_444 = _strassen_4x4_4x4()
    
    # Schoolbook 4x4x3
    def _schoolbook_4x4_3():
        U = [[0] * 48 for _ in range(16)]
        V = [[0] * 48 for _ in range(12)]
        W = [[0] * 48 for _ in range(12)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    school_443 = _schoolbook_4x4_3()
    
    # Combine
    U_total = [[0] * 97 for _ in range(16)]
    V_total = [[0] * 97 for _ in range(28)]
    W_total = [[0] * 97 for _ in range(28)]
    
    for i in range(16):
        for c in range(49):
            U_total[i][c] = strassen_444["U"][i][c]
            V_total[i][c] = strassen_444["V"][i][c]
            W_total[i][c] = strassen_444["W"][i][c]
    
    for i in range(16):
        for c in range(48):
            U_total[i][49 + c] = school_443["U"][i][c]
    
    for j in range(16):
        for c in range(48):
            V_total[j][49 + c] = school_443["V"][j][c]
    
    for k in range(12):
        for c in range(48):
            W_total[k][49 + c] = school_443["W"][k][c]
    
    # Verify the combined decomposition
    combined = {"U": U_total, "V": V_total, "W": W_total}
    
    if time.time() > deadline:
        return best
    
    if _verify(U_total, V_total, W_total):
        return combined
    
    return best

