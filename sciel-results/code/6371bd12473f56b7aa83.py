import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # Schoolbook fallback (rank 48)
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

    # Exact validation
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
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
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build rank-44 base candidate
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # Search for rank-10 decomposition of 2x3x2
    def _find_rank_10_2x3x2():
        n2, m2, p2 = 2, 3, 2
        rank = 10
        # U: 6x10, V: 6x10, W: 4x10
        
        def _calc_err(U, V, W):
            err = 0.0
            for i in range(n2):
                for a in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m2+a][r] * V[a*p2+k][r] * W[k*n2+i][r]
                        err += (s - 1.0) ** 2
            return err

        def _random_init():
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n2*m2)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m2*p2)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p2*n2)]
            return U, V, W

        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(200):
            if not _elapsed_ok():
                return None
            
            U, V, W = _random_init()
            err = _calc_err(U, V, W)
            
            improved = True
            while improved and _elapsed_ok():
                improved = False
                var_type = random.choice(['U', 'V', 'W'])
                if var_type == 'U':
                    idx = random.randint(0, n2*m2-1)
                    r = random.randint(0, rank-1)
                    old_val = U[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        U[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                            if err == 0:
                                return U, V, W
                        else:
                            U[idx][r] = old_val
                elif var_type == 'V':
                    idx = random.randint(0, m2*p2-1)
                    r = random.randint(0, rank-1)
                    old_val = V[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        V[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                            if err == 0:
                                return U, V, W
                        else:
                            V[idx][r] = old_val
                else:
                    idx = random.randint(0, p2*n2-1)
                    r = random.randint(0, rank-1)
                    old_val = W[idx][r]
                    for new_val in [-2, -1, 0, 1, 2]:
                        if new_val == old_val: continue
                        W[idx][r] = new_val
                        new_err = _calc_err(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                            if err == 0:
                                return U, V, W
                        else:
                            W[idx][r] = old_val
                
                if err < best_err:
                    best_err = err
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]

        if best_err == 0:
            return best_U, best_V, best_W
        return None

    def _build_rank_40():
        res = _find_rank_10_2x3x2()
        if not res:
            return None
        
        U_block, V_block, W_block = res
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block_idx in range(4):
            k0 = block_idx * 2
            offset = block_idx * 10
            
            for i in range(n):
                for a in range(m):
                    for r in range(10):
                        U[i*m+a][offset + r] = U_block[i*m+a][r]
            
            for a in range(m):
                for k_block in range(2):
                    for r in range(10):
                        v_idx_block = a*2 + k_block
                        v_idx_global = a*p + (k0 + k_block)
                        V[v_idx_global][offset + r] = V_block[v_idx_block][r]
            
            for k_block in range(2):
                for i in range(n):
                    for r in range(10):
                        w_idx_block = k_block*n + i
                        w_idx_global = (k0 + k_block)*n + i
                        W[w_idx_global][offset + r] = W_block[w_idx_block][r]
        
        return {"U": U, "V": V, "W": W}

    if _elapsed_ok():
        cand_40 = _build_rank_40()
        if cand_40:
            if _validate(cand_40):
                best_candidate = cand_40
                best_rank = 40

    return best_candidate
