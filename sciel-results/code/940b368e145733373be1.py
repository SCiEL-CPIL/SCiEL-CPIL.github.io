"""Model-and-optimize approach: parameterize rank-minimal decomposition and verify."""
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
        from fractions import Fraction
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

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
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

    # Build the rank-44 solution first (Strassen for 2x2x8 + schoolbook for 2x1x8)
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen blocks
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

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
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if not _validate(candidate):
        return _schoolbook()

    # Attempt to find a rank 40 decomposition for 2x3x8 by finding a rank 10 decomposition for 2x3x2
    # and tiling it 4 times.
    
    def _find_rank10_2x3x2():
        """
        Try to find a rank 10 decomposition for 2x3x2.
        Uses structured initialization and local search.
        """
        import random
        random.seed(42)
        
        nn, mm, pp = 2, 3, 2
        r = 10
        
        def _is_valid(UU, VV, WW):
            for li in range(nn):
                for la in range(mm):
                    for lk in range(pp):
                        acc = 0
                        for c in range(r):
                            acc += UU[li*mm+la][c] * VV[la*pp+lk][c] * WW[lk*nn+li][c]
                        if acc != 1:
                            return False
            return True
        
        # Strategy 1: Start from rank-11 solution and try to eliminate one column
        def _build_2x3x2_rank11():
            r11 = 11
            UU11 = [[0]*r11 for _ in range(nn*mm)]
            VV11 = [[0]*r11 for _ in range(mm*pp)]
            WW11 = [[0]*r11 for _ in range(pp*nn)]
            
            col = 0
            for rr in range(7):
                for li in range(2):
                    for la in range(2):
                        uv = S_U[rr][li*2+la]
                        if uv != 0:
                            UU11[li*mm+la][col] = uv
                for la in range(2):
                    for lk in range(2):
                        vv = S_V[rr][la*2+lk]
                        if vv != 0:
                            VV11[la*pp+lk][col] = vv
                for lk in range(2):
                    for li in range(2):
                        wv = S_W[rr][lk*2+li]
                        if wv != 0:
                            WW11[lk*nn+li][col] = wv
                col += 1
            
            for li in range(2):
                for lk in range(2):
                    UU11[li*mm+2][col] = 1
                    VV11[2*pp+lk][col] = 1
                    WW11[lk*nn+li][col] = 1
                    col += 1
            
            return UU11, VV11, WW11
        
        UU11, VV11, WW11 = _build_2x3x2_rank11()
        
        # Try to merge/eliminate columns by finding linear dependencies
        for elim_col in range(11):
            # Create a rank-10 candidate by zeroing out column elim_col
            UU10 = [[UU11[i][c] for c in range(11) if c != elim_col] for i in range(nn*mm)]
            VV10 = [[VV11[i][c] for c in range(11) if c != elim_col] for i in range(mm*pp)]
            WW10 = [[WW11[i][c] for c in range(11) if c != elim_col] for i in range(pp*nn)]
            
            if _is_valid(UU10, VV10, WW10):
                return UU10, VV10, WW10
        
        # Strategy 2: Random search with small integer entries
        for trial in range(200):
            if not _elapsed_ok():
                break
            UU = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(nn*mm)]
            VV = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(mm*pp)]
            WW = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(r)] for _ in range(pp*nn)]
            
            if _is_valid(UU, VV, WW):
                return UU, VV, WW
        
        # Strategy 3: Structured patterns - try to find a pattern that works
        # Use the fact that for 2x3x2, we need 12 constraints satisfied by 10 rank-1 terms
        # Try patterns where each rank-1 term has at most 3 non-zero entries per matrix
        
        for trial in range(100):
            if not _elapsed_ok():
                break
            UU = [[0]*r for _ in range(nn*mm)]
            VV = [[0]*r for _ in range(mm*pp)]
            WW = [[0]*r for _ in range(pp*nn)]
            
            for c in range(r):
                # Each term: pick which (i,a) pair, which (a,k) pair, which (k,i) pair
                ia = random.randint(0, nn*mm-1)
                ak = random.randint(0, mm*pp-1)
                ki = random.randint(0, pp*nn-1)
                
                u_val = random.choice([-1, 1])
                v_val = random.choice([-1, 1])
                w_val = random.choice([-1, 1])
                
                UU[ia][c] = u_val
                VV[ak][c] = v_val
                WW[ki][c] = w_val
            
            if _is_valid(UU, VV, WW):
                return UU, VV, WW
        
        return None

    # Try to find rank 10
    res_10 = _find_rank10_2x3x2()
    
    if res_10:
        UU2, VV2, WW2 = res_10
        rank_2x3x2 = 10
    else:
        # Fallback to rank 11
        def _build_2x3x2_rank11():
            nn, mm, pp = 2, 3, 2
            r = 11
            UU = [[0]*r for _ in range(nn*mm)]
            VV = [[0]*r for _ in range(mm*pp)]
            WW = [[0]*r for _ in range(pp*nn)]
            
            col = 0
            for rr in range(7):
                for li in range(2):
                    for la in range(2):
                        uv = S_U[rr][li*2+la]
                        if uv != 0:
                            UU[li*mm+la][col] = uv
                for la in range(2):
                    for lk in range(2):
                        vv = S_V[rr][la*2+lk]
                        if vv != 0:
                            VV[la*pp+lk][col] = vv
                for lk in range(2):
                    for li in range(2):
                        wv = S_W[rr][lk*2+li]
                        if wv != 0:
                            WW[lk*nn+li][col] = wv
                col += 1
            
            for li in range(2):
                for lk in range(2):
                    UU[li*mm+2][col] = 1
                    VV[2*pp+lk][col] = 1
                    WW[lk*nn+li][col] = 1
                    col += 1
            
            return UU, VV, WW

        UU2, VV2, WW2 = _build_2x3x2_rank11()
        rank_2x3x2 = 11

    # Build 2x3x8 by tiling 2x3x2 four times
    if not _elapsed_ok():
        return _schoolbook()
    
    rank_total = rank_2x3x2 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    for block in range(4):
        k0 = block * 2
        for c in range(rank_2x3x2):
            col = block * rank_2x3x2 + c
            for li in range(n):
                for la in range(m):
                    u_val = UU2[li*m + la][c]
                    if u_val != 0:
                        U[li*m + la][col] = u_val

            for la in range(m):
                for lk in range(2):
                    v_val = VV2[la*2 + lk][c]
                    if v_val != 0:
                        V[la*p + (k0 + lk)][col] = v_val

            for lk in range(2):
                for li in range(n):
                    w_val = WW2[lk*2 + li][c]
                    if w_val != 0:
                        W[(k0 + lk)*n + li][col] = w_val

        if not _elapsed_ok():
            return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if not _elapsed_ok():
        return _schoolbook()
    
    if not _validate(candidate):
        return _schoolbook()
        
    return candidate

# EVOLVE-BLOCK-END
