"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen(I, J, K):
        nonlocal col
        # Strassen coefficients for A (U), B (V), C (W)
        # Local indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1)
        
        # m1 = (a+d)(e+h)
        u_vals = [1, 0, 0, 1]
        v_vals = [1, 0, 0, 1]
        w_vals = [1, 0, 0, 1] # C11=1, C22=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m2 = (c+d)e
        u_vals = [0, 0, 1, 1]
        v_vals = [1, 0, 0, 0]
        w_vals = [0, 0, 1, -1] # C21=1, C22=-1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m3 = a(f-h)
        u_vals = [1, 0, 0, 0]
        v_vals = [0, 1, 0, -1]
        w_vals = [0, 1, 0, 1] # C12=1, C22=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m4 = d(g-e)
        u_vals = [0, 0, 0, 1]
        v_vals = [-1, 0, 1, 0]
        w_vals = [1, 0, 1, 0] # C11=1, C21=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m5 = (a+b)h
        u_vals = [1, 1, 0, 0]
        v_vals = [0, 0, 0, 1]
        w_vals = [-1, 1, 0, 0] # C11=-1, C12=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m6 = (c-a)(e+f)
        u_vals = [-1, 0, 1, 0]
        v_vals = [1, 1, 0, 0]
        w_vals = [0, 0, 0, 1] # C22=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1
        
        # m7 = (b-d)(g+h)
        u_vals = [0, 1, 0, -1]
        v_vals = [0, 0, 1, 1]
        w_vals = [1, 0, 0, 0] # C11=1
        for u_l, v_l, w_l in zip(range(4), range(4), range(4)):
            if u_vals[u_l] != 0:
                i_l = u_l // 2
                j_l = u_l % 2
                u_g = (I + i_l) * m + (J + j_l)
                U[u_g][col] += u_vals[u_l]
            if v_vals[v_l] != 0:
                j_l = v_l // 2
                k_l = v_l % 2
                v_g = (J + j_l) * p + (K + k_l)
                V[v_g][col] += v_vals[v_l]
            if w_vals[w_l] != 0:
                k_l = w_l // 2
                i_l = w_l % 2
                w_g = (K + k_l) * n + (I + i_l)
                W[w_g][col] += w_vals[w_l]
        col += 1

    # 1. Block n=1 (i=0), m=4, p=5. Rank 20.
    for j in range(4):
        for k in range(5):
            add_schoolbook(0, j, k)

    # 2. Block n=2 (i=1,2), m=4, p=5.
    # Split m into 0,1 and 2,3.
    for J in [0, 2]:
        # Sub-block M_2,2,5 at offset I=1, J=J
        # Split p into 0,1 (Strassen), 2,3 (Strassen), 4 (Schoolbook)
        
        # p=0,1
        add_strassen(1, J, 0)
        
        # p=2,3
        add_strassen(1, J, 2)
        
        # p=4
        for i_l in range(2):
            for j_l in range(2):
                i_g = 1 + i_l
                j_g = J + j_l
                k_g = 4
                add_schoolbook(i_g, j_g, k_g)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
