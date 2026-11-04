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

    # Now try to reduce rank by removing redundant terms
    # A term c is redundant if setting all its entries to 0 still leaves the decomposition valid
    # AND the term has some structure that allows removal
    
    # Better approach: try to find linear dependencies among the rank-1 tensors
    # If two terms can be combined or one is redundant, we can reduce rank
    
    # Try: for each term, check if we can zero it out and compensate by adjusting others
    # This is complex, so instead try a simpler heuristic:
    # Look for terms that only affect a single (i,a,k) entry and see if they can be merged
    
    # Actually, let's try a different strategy: use a 2x3x3 base decomposition
    # and tile it across the 8 columns
    
    # The rank of 2x3x3 matrix multiplication: let's try to find it
    # Lower bound: max(6, 9, 6) = 9? Not necessarily.
    # Let's try to construct a low-rank 2x3x3 decomposition
    
    # Try: use Strassen on a 2x2 subblock and add terms for the 3rd column
    # 2x2x3: rank <= 7 + 2*3 = 13? Or better?
    
    # Let's try a direct construction for 2x3x3 with small rank
    # T[i,a,k] = 1 for i in {0,1}, a in {0,1,2}, k in {0,1,2}
    
    # Try rank 12: use Strassen for 2x2x2 (7 terms) + 2x1x2 (4 terms) + 2x3x1 (6 terms) = 17? No.
    # Better: 2x2x2 (7) + 2x1x2 (4) + 2x2x1 (4) + 2x1x1 (2) = 17
    
    # Let me try a smarter approach: find the minimal rank for 2x3x3 by search
    
    # For now, let's try to see if we can do better than 44 for 2x3x8
    # by using a different block structure
    
    # Alternative: split 8 into 3+3+2
    # 2x3x3 + 2x3x3 + 2x3x2
    # If rank(2x3x3) = r and rank(2x3x2) = s, total = 2r + s
    
    # Let's try to find rank(2x3x2) and rank(2x3x3)
    
    # For 2x3x2: T[i,a,k] = 1 for i in {0,1}, a in {0,1,2}, k in {0,1}
    # This is equivalent to multiplying 2x3 by 3x2
    # Rank <= 2*3*2 = 12 (schoolbook)
    # Can we do better?
    
    # Try: use Strassen on 2x2x2 part (a in {0,1}, k in {0,1}) = 7 terms
    # Plus 2x1x2 part (a=2, k in {0,1}) = 4 terms
    # Total: 11
    
    # Let's verify this works for 2x3x2
    def _build_2x3x2():
        """Build rank-11 decomposition for 2x3x2."""
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
    
    UU2, VV2, WW2 = _build_2x3x2()
    
    # Validate 2x3x2
    from fractions import Fraction
    valid_2x3x2 = True
    for li in range(2):
        for la in range(3):
            for lk in range(2):
                acc = Fraction(0)
                for c in range(11):
                    acc += Fraction(UU2[li*3+la][c]) * Fraction(VV2[la*2+lk][c]) * Fraction(WW2[lk*2+li][c])
                if acc != 1:
                    valid_2x3x2 = False
                    break
            if not valid_2x3x2:
                break
        if not valid_2x3x2:
            break
    
    if valid_2x3x2:
        # Now try 2x3x3
        # Rank <= 7 (Strassen 2x2x2) + 4 (2x1x2) + 6 (2x3x1) = 17? 
        # Or: 7 + 4 + 2*3 = 17
        # Better: try to share terms
        
        # Actually, for 2x3x3:
        # Use Strassen for 2x2x2 (7 terms) covering i in {0,1}, a in {0,1}, k in {0,1}
        # Then need: i in {0,1}, a in {0,1}, k=2: 4 entries
        #           i in {0,1}, a=2, k in {0,1,2}: 6 entries
        # Total additional: 4 + 6 = 10, total rank = 17
        
        # But maybe we can do better by mixing
        # Let's just use the tiling approach: 2x3x8 = 2x3x3 + 2x3x3 + 2x3x2
        # Rank = 2*17 + 11 = 45? That's worse than 44.
        
        # Alternative: 2x3x8 = 4 * 2x3x2
        # Rank = 4 * 11 = 44. Same as current.
        
        # So the current 44 is already using the optimal tiling of 2x3x2 blocks.
        
        # Can we do 2x3x2 with rank < 11?
        # Lower bound for 2x3x2: the tensor has 12 entries, each must equal 1.
        # A rank-1 tensor covers at most... well, it's a product of vectors.
        # The rank is at least the max of the ranks of the slices.
        # Slice along i: two 3x2 matrices, each rank <= 2, so rank >= 2? No, that's not a lower bound.
        
        # Let's try to find if rank 10 works for 2x3x2
        # We need 12 equations. With 10 terms, we have 10*(6+6+4) = 160 variables, 12 equations.
        # Should be feasible, but the equations are nonlinear.
        
        # For now, let's just return the rank-44 solution since it's valid
        pass
    
    return candidate

# EVOLVE-BLOCK-END
