import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

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

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    total_rank = 26
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    # n_block=2, m_block=2, p_block=2
    # i_off, a_off, k_off are offsets for the global indices
    def add_strassen(i_off, a_off, k_off):
        nonlocal col
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = (i_off + li) * m + (a_off + la)
                    U[u_idx][col] += u_val
            
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = (a_off + la) * p + (k_off + lk)
                    V[v_idx][col] += v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k_off + lk) * n + (i_off + li)
                    W[w_idx][col] += w_val
            
            col += 1
            if not _elapsed_ok():
                return False
        return True

    # Helper to add Rank 1 "Outer Product" like term
    # M_{2,1,K}: n=2, m=1, p=K
    # U[i, a, r], V[a, k, r], W[k, i, r]
    # All 1s
    def add_rank1_21K(i_off, a_off, k_off, p_block):
        nonlocal col
        for li in range(2):
            for lk in range(p_block):
                # U[i, a] = 1
                u_idx = (i_off + li) * m + (a_off) # a is single index
                U[u_idx][col] += 1
                
                # V[a, k] = 1
                v_idx = (a_off) * p + (k_off + lk)
                V[v_idx][col] += 1
                
                # W[k, i] = 1
                w_idx = (k_off + lk) * n + (i_off + li)
                W[w_idx][col] += 1
        col += 1
        return True

    # Helper to add Rank 1 "Dot Product" like term
    # M_{2,2,1}: n=2, m=2, p=1
    # U[i, a, r], V[a, k, r], W[k, i, r]
    # All 1s
    def add_rank1_221(i_off, a_off, k_off):
        nonlocal col
        for li in range(2):
            for la in range(2):
                # U[i, a] = 1
                u_idx = (i_off + li) * m + (a_off + la)
                U[u_idx][col] += 1
                
                # V[a, k] = 1
                v_idx = (a_off + la) * p + (k_off)
                V[v_idx][col] += 1
                
                # W[k, i] = 1
                w_idx = (k_off) * n + (i_off + li)
                W[w_idx][col] += 1
        col += 1
        return True

    # Block 1: 2x3x3 (Cols 0-2 of B)
    # A_L (2x2) @ B_LL (2x2) -> Strassen
    if not add_strassen(0, 0, 0): return _schoolbook()
    # A_L (2x2) @ B_LR (2x1) -> Rank 1 221
    if not add_rank1_221(0, 0, 2): return _schoolbook()
    # A_R (2x1) @ B_R (1x3) -> Rank 1 213
    if not add_rank1_21K(0, 2, 0, 3): return _schoolbook()
    
    # Block 2: 2x3x3 (Cols 3-5 of B)
    # A_L (2x2) @ B_LL (2x2) -> Strassen
    if not add_strassen(0, 0, 3): return _schoolbook()
    # A_L (2x2) @ B_LR (2x1) -> Rank 1 221
    if not add_rank1_221(0, 0, 5): return _schoolbook()
    # A_R (2x1) @ B_R (1x3) -> Rank 1 213
    if not add_rank1_21K(0, 2, 3, 3): return _schoolbook()

    # Block 3: 2x3x2 (Cols 6-7 of B)
    # A_L (2x2) @ B_LL (2x2) -> Strassen
    if not add_strassen(0, 0, 6): return _schoolbook()
    # A_R (2x1) @ B_R (1x2) -> Rank 1 212
    if not add_rank1_21K(0, 2, 6, 2): return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
