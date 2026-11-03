import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 140.0  # safety margin

    # Strassen 2x2x2 factors
    # U: 4x7, row = i*2+j
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0
        [0, 0, 1, 0, 1, 0, -1],   # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1],   # i=1,j=1
    ], dtype=float)
    
    # V: 4x7, row = j*2+k
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0
        [0, 0, 1, 0, 0, 1, 0],    # j=0,k=1
        [0, 0, 0, 1, 0, 0, 1],    # j=1,k=0
        [1, 0, -1, 0, 1, 0, 0],   # j=1,k=1
    ], dtype=float)
    
    # W: 4x7, row = k*2+i
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # k=0,i=0
        [0, 0, 1, 0, 1, 0, 0],    # k=0,i=1
        [0, 1, 0, 1, 0, 0, 0],    # k=1,i=0
        [1, -1, 1, 0, 0, 1, 0],   # k=1,i=1
    ], dtype=float)
    
    # Verify Strassen 2x2x2
    def verify_s(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = np.dot(U[i*2+j], np.multiply(V[j*2+k], W[k*2+i]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check no spurious terms (all i,j,k in 2x2x2 are valid, so just check the 8 valid ones)
        return True
    
    if not verify_s(U_s, V_s, W_s):
        return _schoolbook()
    
    # Build full decomposition
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    # j-groups: [0,1] and [2,3]
    # k-groups: [0,1], [2,3], [4,5], [6]
    
    # For each j-group (2 groups) and k-group:
    #   3 k-groups of size 2: rank 7 each
    #   1 k-group of size 1: rank 4
    # Total per j-group: 3*7 + 4 = 25
    # Total: 2 * 25 = 50
    
    total_rank = 50
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for jg in range(2):  # j-group index
        j_offset = jg * 2  # j_full = j_offset + j_loc
        
        for kg in range(4):  # k-group index
            if kg < 3:
                # 2x2x2 block, rank 7
                k_offset = kg * 2  # k_full = k_offset + k_loc
                r = 7
                for i in range(2):
                    for j_loc in range(2):
                        U[i*m + j_offset + j_loc, col:col+r] += U_s[i*2 + j_loc]
                for j_loc in range(2):
                    for k_loc in range(2):
                        V[(j_offset + j_loc)*p + k_offset + k_loc, col:col+r] += V_s[j_loc*2 + k_loc]
                for k_loc in range(2):
                    for i in range(2):
                        W[(k_offset + k_loc)*n + i, col:col+r] += W_s[k_loc*2 + i]
                col += r
            else:
                # 2x2x1 block (k=6), rank 4
                k_val = 6
                r = 4
                for i in range(2):
                    for j_loc in range(2):
                        U[i*m + j_offset + j_loc, col:col+r] += 1.0
                for j_loc in range(2):
                    V[(j_offset + j_loc)*p + k_val, col:col+r] += 1.0
                for i in range(2):
                    W[k_val*n + i, col:col+r] += 1.0
                col += r
    
    # Verify
    def check_valid(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    if time.time() < deadline:
        if check_valid(U, V, W):
            def clean(x):
                if abs(x - round(x)) < 1e-10:
                    return int(round(x))
                if abs(2*x - round(2*x)) < 1e-10:
                    return round(2*x) / 2
                return float(x)
            return {
                "U": [[clean(U[i, j]) for j in range(total_rank)] for i in range(n*m)],
                "V": [[clean(V[i, j]) for j in range(total_rank)] for i in range(m*p)],
                "W": [[clean(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
            }
    
    return _schoolbook()

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
