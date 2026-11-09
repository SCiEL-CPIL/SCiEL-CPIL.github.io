import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        if u_idx is not None:
            U[u_idx][col] += val
        if v_idx is not None:
            V[v_idx][col] += val
        if w_idx is not None:
            W[w_idx][col] += val
        col += 1

    # Block 1: A[:, 0] * B[0, :]
    # A[i, 0] * B[0, k] -> C[i, k]
    for i in range(2):
        for k in range(3):
            u_idx = i * 3 + 0
            v_idx = 0 * 3 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # Block 2a: A[:, 1:3] * B[1:3, 0]
    # A[i, j] * B[j, 0] -> C[i, 0] for j in 1, 2
    for i in range(2):
        for j in range(1, 3):
            u_idx = i * 3 + j
            v_idx = j * 3 + 0
            w_idx = 0 * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # Block 2b: Strassen for A[:, 1:3] * B[1:3, 1:3]
    # A_sub: [[A01, A02], [A11, A12]]
    # B_sub: [[B11, B12], [B21, B22]]
    # C_sub: [[C01, C02], [C11, C12]]
    
    # Indices for A_sub
    a00_u = 0 * 3 + 1
    a01_u = 0 * 3 + 2
    a10_u = 1 * 3 + 1
    a11_u = 1 * 3 + 2
    
    # Indices for B_sub
    b00_v = 1 * 3 + 1
    b01_v = 1 * 3 + 2
    b10_v = 2 * 3 + 1
    b11_v = 2 * 3 + 2
    
    # Indices for C_sub
    c00_w = 1 * 2 + 0
    c01_w = 2 * 2 + 0
    c10_w = 1 * 2 + 1
    c11_w = 2 * 2 + 1
    
    # m1 = (a00 + a10) * (b00 + b11)
    # C00 += 1, C11 += 1
    U[a00_u][col] = 1
    U[a10_u][col] = 1
    V[b00_v][col] = 1
    V[b11_v][col] = 1
    W[c00_w][col] = 1
    W[c11_w][col] = 1
    col += 1
    
    # m2 = (a10 + a11) * b00
    # C10 += 1, C11 -= 1
    U[a10_u][col] = 1
    U[a11_u][col] = 1
    V[b00_v][col] = 1
    W[c10_w][col] = 1
    W[c11_w][col] = -1
    col += 1
    
    # m3 = a00 * (b10 - b00)
    # C01 += 1
    U[a00_u][col] = 1
    V[b10_v][col] = 1
    V[b00_v][col] = -1
    W[c01_w][col] = 1
    col += 1
    
    # m4 = a11 * (b01 - b10)
    # C00 += 1
    U[a11_u][col] = 1
    V[b01_v][col] = 1
    V[b10_v][col] = -1
    W[c00_w][col] = 1
    col += 1
    
    # m5 = (a00 + a01) * b11
    # C00 -= 1, C01 += 1
    U[a00_u][col] = 1
    U[a01_u][col] = 1
    V[b11_v][col] = 1
    W[c00_w][col] = -1
    W[c01_w][col] = 1
    col += 1
    
    # m6 = (a01 - a10) * (b00 + b10)
    # C11 += 1
    U[a01_u][col] = 1
    U[a10_u][col] = -1
    V[b00_v][col] = 1
    V[b10_v][col] = 1
    W[c11_w][col] = 1
    col += 1
    
    # m7 = (a11 - a00) * (b01 + b11)
    # C00 -= 1, C10 += 1
    U[a11_u][col] = 1
    U[a00_u][col] = -1
    V[b01_v][col] = 1
    V[b11_v][col] = 1
    W[c00_w][col] = -1
    W[c10_w][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
