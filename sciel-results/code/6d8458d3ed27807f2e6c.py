"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
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

    # Build candidate with rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # Split B1 cols into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: rows indexed by (i, a) where i in {0,1}, a in {0,1}
            su = S_U[r]
            for li in range(2):
                for la in range(2):
                    u_val = su[li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            
            # V: rows indexed by (a, k) where a in {0,1}, k in {k0, k0+1}
            sv = S_V[r]
            for la in range(2):
                for lk in range(2):
                    v_val = sv[la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            
            # W: rows indexed by (k, i) where k in {k0, k0+1}, i in {0,1}
            sw = S_W[r]
            for lk in range(2):
                for li in range(2):
                    w_val = sw[lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8) - schoolbook for column 2
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Validate using integer arithmetic (exact for integer entries)
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c]
                        if uv == 0:
                            continue
                        vv = v_row[c]
                        if vv == 0:
                            continue
                        ww = w_row[c]
                        if ww == 0:
                            continue
                        acc += uv * vv * ww
                    if acc != 1:
                        return False
        return True

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
