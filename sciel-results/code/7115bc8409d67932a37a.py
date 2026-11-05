import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _rank7_2x2x2():
    """
    Build a rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    Returns (U7, V7, W7) each of shape (4, 7) in the 2x2 local index convention:
      A indices: 0=A00, 1=A01, 2=A10, 3=A11
      B indices: 0=B00, 1=B01, 2=B10, 3=B11
      C indices: 0=C00, 1=C01, 2=C10, 3=C11
    """
    U7 = np.zeros((4, 7))
    V7 = np.zeros((4, 7))
    W7 = np.zeros((4, 7))
    
    # Product 1: (A00+A11)(B00+B11) -> C00 + C11
    U7[0, 0] = 1; U7[3, 0] = 1
    V7[0, 0] = 1; V7[3, 0] = 1
    W7[0, 0] = 1; W7[3, 0] = 1
    
    # Product 2: (A01+A11)(B00+B10) -> C00 + C10
    U7[1, 1] = 1; U7[3, 1] = 1
    V7[0, 1] = 1; V7[2, 1] = 1
    W7[0, 1] = 1; W7[2, 1] = 1
    
    # Product 3: (A00+A10)(B01+B11) -> C01 + C11
    U7[0, 2] = 1; U7[2, 2] = 1
    V7[1, 2] = 1; V7[3, 2] = 1
    W7[1, 2] = 1; W7[3, 2] = 1
    
    # Product 4: (A00+A01)(B00+B01) -> C00 + C01
    U7[0, 3] = 1; U7[1, 3] = 1
    V7[0, 3] = 1; V7[1, 3] = 1
    W7[0, 3] = 1; W7[1, 3] = 1
    
    # Product 5: (A10+A11)(B00+B01) -> C10 + C11
    U7[2, 4] = 1; U7[3, 4] = 1
    V7[0, 4] = 1; V7[1, 4] = 1
    W7[2, 4] = 1; W7[3, 4] = 1
    
    # Product 6: (A01+A10)(B00+B11) -> C00 + C11
    U7[1, 5] = 1; U7[2, 5] = 1
    V7[0, 5] = 1; V7[3, 5] = 1
    W7[0, 5] = 1; W7[3, 5] = 1
    
    # Product 7: (A00+A11)(B01+B10) -> C01 + C10
    U7[0, 6] = 1; U7[3, 6] = 1
    V7[1, 6] = 1; V7[2, 6] = 1
    W7[1, 6] = 1; W7[2, 6] = 1
    
    # Verify all 64 entries
    for a in range(4):
        for b in range(4):
            for c in range(4):
                val = sum(U7[a, r] * V7[b, r] * W7[c, r] for r in range(7))
                i_a = a // 2
                j_a = a % 2
                j_b = b // 2
                k_b = b % 2
                i_c = c // 2
                k_c = c % 2
                expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                if abs(val - expected) > 1e-9:
                    return None
    return U7, V7, W7

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Try rank-27 decomposition using rank-7 for 2x2x2 blocks
    rank7 = _rank7_2x2x2()
    if rank7 is not None:
        U7, V7, W7 = rank7
        # Total rank: Part1 (2x5 direct) = 10, Part2 (2x1 direct) = 2, Part3 (2x2 rank7) = 7, Part4 (2x2 rank7) = 7
        rank = 10 + 2 + 7 + 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        
        # Part 3: 2x2 block (j=0,1, k=0,1) using rank-7
        A_map3 = [0, 1, 3, 4]
        B_map3 = [0, 1, 5, 6]
        C_map3 = [0, 1, 2, 3]
        for rr in range(7):
            for a_local in range(4):
                if U7[a_local, rr] != 0:
                    U[A_map3[a_local]][r] += U7[a_local, rr]
            for b_local in range(4):
                if V7[b_local, rr] != 0:
                    V[B_map3[b_local]][r] += V7[b_local, rr]
            for c_local in range(4):
                if W7[c_local, rr] != 0:
                    W[C_map3[c_local]][r] += W7[c_local, rr]
            r += 1
        
        # Part 4: 2x2 block (j=0,1, k=2,3) using rank-7
        A_map4 = [0, 1, 3, 4]
        B_map4 = [2, 3, 7, 8]
        C_map4 = [4, 5, 6, 7]
        for rr in range(7):
            for a_local in range(4):
                if U7[a_local, rr] != 0:
                    U[A_map4[a_local]][r] += U7[a_local, rr]
            for b_local in range(4):
                if V7[b_local, rr] != 0:
                    V[B_map4[b_local]][r] += V7[b_local, rr]
            for c_local in range(4):
                if W7[c_local, rr] != 0:
                    W[C_map4[c_local]][r] += W7[c_local, rr]
            r += 1
        
        # Verify full decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = sum(U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx] for r_idx in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-28 Strassen-based decomposition
    return _strassen_rank28(n, m, p)

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": U, "V": V, "W": W}
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
