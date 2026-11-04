"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
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

    # Build 2x3x2 block decomposition (rank 11): Strassen for 2x2x2 + schoolbook for 2x1x2
    def _build_2x3x2_block():
        """Return U, V, W rows for a single 2x3x2 block, indexed by local (i,a,k)."""
        nn, mm, pp = 2, 3, 2
        r = 11
        UU = [[0]*r for _ in range(nn*mm)]
        VV = [[0]*r for _ in range(mm*pp)]
        WW = [[0]*r for _ in range(pp*nn)]
        
        col = 0
        # Strassen for 2x2x2 (i in {0,1}, a in {0,1}, k in {0,1})
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
        
        # Schoolbook for 2x1x2 (i in {0,1}, a=2, k in {0,1})
        for li in range(2):
            for lk in range(2):
                UU[li*mm+2][col] = 1
                VV[2*pp+lk][col] = 1
                WW[lk*nn+li][col] = 1
                col += 1
        
        return UU, VV, WW

    UU_block, VV_block, WW_block = _build_2x3x2_block()

    # Assemble 4 blocks for 2x3x8: each block handles 2 k-columns
    rank = 44  # 4 * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for block_idx in range(4):
        k0 = block_idx * 2
        base_col = block_idx * 11
        for c in range(11):
            col = base_col + c
            # U: local (i, a) -> global (i, a)
            for li in range(2):
                for la in range(3):
                    U[li * m + la][col] = UU_block[li * 3 + la][c]
            # V: local (a, lk) -> global (a, k0+lk)
            for la in range(3):
                for lk in range(2):
                    V[la * p + (k0 + lk)][col] = VV_block[la * 2 + lk][c]
            # W: local (lk, li) -> global (k0+lk, li)
            for lk in range(2):
                for li in range(2):
                    W[(k0 + lk) * n + li][col] = WW_block[lk * 2 + li][c]

    if not _elapsed_ok():
        return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if not _validate(candidate):
        return _schoolbook()

    # Try to reduce rank via linear dependency elimination
    # The 4 blocks share the same structure, so there may be linear dependencies
    # across blocks that allow rank reduction
    
    # Strategy: identify columns (terms) that are linearly dependent
    # across the U, V, W matrices simultaneously
    
    # For each pair of terms, check if one can be expressed as a linear combination
    # of the other in all three matrices simultaneously
    
    # Simpler: try removing each term individually and see if we can compensate
    # by adjusting remaining terms (this is hard)
    
    # Alternative: use the fact that the 4 blocks are identical in structure
    # If we can find a non-trivial linear combination of the 4 block patterns
    # that gives zero in all three matrices, we can reduce rank
    
    # Let's try: for each set of terms, check if there's a linear relation
    # that allows us to drop one term
    
    # Actually, the simplest structural improvement: 
    # The 2x3x2 block uses rank 11, but maybe we can do rank 10
    # Let's try a direct construction for rank 10 2x3x2
    
    # For 2x3x2 with rank 10:
    # We need 10 terms such that for all (i,a,k) in {0,1}x{0,1,2}x{0,1}:
    # sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1
    
    # Try: 7 Strassen terms for (a in {0,1}, k in {0,1}) + 3 terms for a=2
    # For a=2, we need: sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1 for all i,k
    # With 3 terms, let's try:
    # Term 7: U[0,2]=1, U[1,2]=0, V[2,0]=1, V[2,1]=0, W[0,0]=1, W[1,0]=0, W[0,1]=0, W[1,1]=0
    #   Gives: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=0
    # Term 8: U[0,2]=0, U[1,2]=1, V[2,0]=0, V[2,1]=1, W[0,0]=0, W[1,0]=0, W[0,1]=0, W[1,1]=1
    #   Gives: (0,0)=0, (0,1)=0, (1,0)=0, (1,1)=1
    # Term 9: U[0,2]=1, U[1,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=0, W[1,0]=1, W[0,1]=1, W[1,1]=0
    #   Gives: (0,0)=1*1*0=0, (0,1)=1*1*1=1, (1,0)=1*1*1=1, (1,1)=1*1*0=0
    # Total for a=2: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1. 
    
    # Let's build this rank-10 2x3x2 block
    def _build_2x3x2_rank10():
        nn, mm, pp = 2, 3, 2
        r = 10
        UU = [[0]*r for _ in range(nn*mm)]
        VV = [[0]*r for _ in range(mm*pp)]
        WW = [[0]*r for _ in range(pp*nn)]
        
        col = 0
        # Strassen for 2x2x2 (i in {0,1}, a in {0,1}, k in {0,1})
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
        
        # Term 7: handles (i=0, k=0) for a=2
        UU[0*mm+2][col] = 1  # U[0,2]
        UU[1*mm+2][col] = 0  # U[1,2]
        VV[2*pp+0][col] = 1  # V[2,0]
        VV[2*pp+1][col] = 0  # V[2,1]
        WW[0*nn+0][col] = 1  # W[0,0]
        WW[1*nn+0][col] = 0  # W[1,0]
        WW[0*nn+1][col] = 0  # W[0,1]
        WW[1*nn+1][col] = 0  # W[1,1]
        col += 1
        
        # Term 8: handles (i=1, k=1) for a=2
        UU[0*mm+2][col] = 0
        UU[1*mm+2][col] = 1
        VV[2*pp+0][col] = 0
        VV[2*pp+1][col] = 1
        WW[0*nn+0][col] = 0
        WW[1*nn+0][col] = 0
        WW[0*nn+1][col] = 0
        WW[1*nn+1][col] = 1
        col += 1
        
        # Term 9: handles (i=0, k=1) and (i=1, k=0) for a=2
        UU[0*mm+2][col] = 1
        UU[1*mm+2][col] = 1
        VV[2*pp+0][col] = 1
        VV[2*pp+1][col] = 1
        WW[0*nn+0][col] = 0
        WW[1*nn+0][col] = 1
        WW[0*nn+1][col] = 1
        WW[1*nn+1][col] = 0
        col += 1
        
        return UU, VV, WW
    
    # Validate rank-10 2x3x2
    UU10, VV10, WW10 = _build_2x3x2_rank10()
    valid_r10 = True
    from fractions import Fraction
    for li in range(2):
        for la in range(3):
            for lk in range(2):
                acc = Fraction(0)
                for c in range(10):
                    acc += Fraction(UU10[li*3+la][c]) * Fraction(VV10[la*2+lk][c]) * Fraction(WW10[lk*2+li][c])
                if acc != 1:
                    valid_r10 = False
                    break
            if not valid_r10:
                break
        if not valid_r10:
            break
    
    if valid_r10 and _elapsed_ok():
        # Build 2x3x8 with rank 40 (4 * 10)
        rank40 = 40
        U40 = [[0] * rank40 for _ in range(n * m)]
        V40 = [[0] * rank40 for _ in range(m * p)]
        W40 = [[0] * rank40 for _ in range(p * n)]
        
        for block_idx in range(4):
            k0 = block_idx * 2
            base_col = block_idx * 10
            for c in range(10):
                col = base_col + c
                for li in range(2):
                    for la in range(3):
                        U40[li * m + la][col] = UU10[li * 3 + la][c]
                for la in range(3):
                    for lk in range(2):
                        V40[la * p + (k0 + lk)][col] = VV10[la * 2 + lk][c]
                for lk in range(2):
                    for li in range(2):
                        W40[(k0 + lk) * n + li][col] = WW10[lk * 2 + li][c]
        
        candidate40 = {"U": U40, "V": V40, "W": W40}
        if _validate(candidate40):
            return candidate40
    
    return candidate

# EVOLVE-BLOCK-END
