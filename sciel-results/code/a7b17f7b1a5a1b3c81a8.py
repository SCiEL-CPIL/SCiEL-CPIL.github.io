"""
Strategy:
1. Base: 28-term decomposition using 4 Strassen blocks for 2x2x8.
2. Structural change: Try to absorb the 2x1x8 schoolbook terms (16 terms) into the
   28 Strassen terms by allowing non-zero entries in U[:,2,r] and V[2,:,r].
   This would reduce rank from 44 to 28 if successful.
3. If full absorption fails, try partial absorption or other structural optimizations.
4. Use exact Fraction arithmetic for validation.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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

    # Verify Strassen decomposition internally
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

    # Build 28-term base: 4 Strassen blocks for 2x2x8
    rank_28 = 28
    U_28 = [[0] * rank_28 for _ in range(nm)]
    V_28 = [[0] * rank_28 for _ in range(mp)]
    W_28 = [[0] * rank_28 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_28[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_28[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_28[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Try to absorb the 2x1x8 part into the 28 Strassen terms
    # We need to find u_extra[i,r], v_extra[k,r] for i in {0,1}, k in {0..7}, r in {0..27}
    # such that for all i in {0,1}, k in {0..7}:
    # sum_r u_extra[i,r] * v_extra[k,r] * W_28[k*n+i, r] = 1
    #
    # This is a system of 16 equations in 28*2 = 56 unknowns (u_extra and v_extra)
    # But it's bilinear, so we can't solve directly. Instead, we can fix one set and solve for the other.
    
    # Strategy: Try small integer/half-integer values for u_extra and v_extra
    # and check if the equations are satisfied.
    
    def _try_absorb(U_base, V_base, W_base):
        """Try to absorb the 2x1x8 part into the given 28-term base."""
        if not _elapsed_ok():
            return None
        
        # For each of the 28 terms, we can add u_extra[i,r] to U[i*3+2, r]
        # and v_extra[k,r] to V[2*8+k, r]
        # The constraint is: for each (i,k) in {0,1}x{0..7}:
        # sum_r u_extra[i,r] * v_extra[k,r] * W_base[k*2+i, r] = 1
        
        # Try small values: u_extra[i,r] and v_extra[k,r] in {-1, -1/2, 0, 1/2, 1}
        # This is 5^56 which is too large, so we need a smarter approach.
        
        # Alternative: Fix u_extra to be simple (e.g., all 1s or 0s) and solve for v_extra
        # Or use a greedy approach: for each (i,k), try to satisfy the equation
        # by adjusting one term at a time.
        
        # Let's try: set u_extra[i,r] = 1 for all i,r, and solve for v_extra
        # Then: sum_r v_extra[k,r] * W_base[k*2+i, r] = 1 for each i,k
        # This is 16 linear equations in 28 unknowns (v_extra[k,r] for k=0..7, r=0..27)
        # But it's actually 2 separate systems (one for i=0, one for i=1)
        # Each has 8 equations in 28 unknowns.
        
        # Actually, let's be more systematic. We have:
        # For i=0: sum_r u_extra[0,r] * v_extra[k,r] * W_base[k*2+0, r] = 1 for k=0..7
        # For i=1: sum_r u_extra[1,r] * v_extra[k,r] * W_base[k*2+1, r] = 1 for k=0..7
        
        # Let's try u_extra[i,r] = 1 for all i,r. Then:
        # For i=0: sum_r v_extra[k,r] * W_base[k*2+0, r] = 1 for k=0..7
        # For i=1: sum_r v_extra[k,r] * W_base[k*2+1, r] = 1 for k=0..7
        
        # This is 16 linear equations. Let's solve it.
        # Variables: v_extra[k,r] for k=0..7, r=0..27 (228 variables)
        # But we only need to satisfy 16 equations, so there are many solutions.
        # Let's find a simple solution.
        
        # For each k, we need:
        # sum_r v_extra[k,r] * W_base[k*2+0, r] = 1  (for i=0)
        # sum_r v_extra[k,r] * W_base[k*2+1, r] = 1  (for i=1)
        
        # Let's try v_extra[k,r] = 1 for all k,r. Then:
        # sum_r W_base[k*2+0, r] = 1? and sum_r W_base[k*2+1, r] = 1?
        # Let's check.
        
        for k in range(p):
            sum_w0 = sum(W_base[k * n + 0][r] for r in range(rank_28))
            sum_w1 = sum(W_base[k * n + 1][r] for r in range(rank_28))
            if sum_w0 != 1 or sum_w1 != 1:
                # v_extra[k,r] = 1 doesn't work for this k
                pass
        
        # Let's try a different approach: for each k, set v_extra[k,r] such that
        # the two equations are satisfied. Since we have 2 equations and 28 unknowns
        # for each k, we can set most to 0 and solve for 2.
        
        # For each k, pick two terms r1, r2 such that W_base[k*2+0, r1] and W_base[k*2+0, r2]
        # are linearly independent, and solve for v_extra[k,r1] and v_extra[k,r2].
        
        v_extra = [[0] * rank_28 for _ in range(p)]
        
        for k in range(p):
            # Find two terms r1, r2 with non-zero W values
            r1 = None
            r2 = None
            for r in range(rank_28):
                if W_base[k * n + 0][r] != 0 or W_base[k * n + 1][r] != 0:
                    if r1 is None:
                        r1 = r
                    else:
                        r2 = r
                        break
            
            if r1 is None or r2 is None:
                return None  # Can't satisfy for this k
            
            w0_r1 = Fraction(W_base[k * n + 0][r1])
            w1_r1 = Fraction(W_base[k * n + 1][r1])
            w0_r2 = Fraction(W_base[k * n + 0][r2])
            w1_r2 = Fraction(W_base[k * n + 1][r2])
            
            # Solve:
            # v_extra[k,r1] * w0_r1 + v_extra[k,r2] * w0_r2 = 1
            # v_extra[k,r1] * w1_r1 + v_extra[k,r2] * w1_r2 = 1
            
            det = w0_r1 * w1_r2 - w0_r2 * w1_r1
            if det == 0:
                # Try another pair
                found = False
                for r2_alt in range(rank_28):
                    if r2_alt == r1:
                        continue
                    w0_r2a = Fraction(W_base[k * n + 0][r2_alt])
                    w1_r2a = Fraction(W_base[k * n + 1][r2_alt])
                    det_alt = w0_r1 * w1_r2a - w0_r2a * w1_r1
                    if det_alt != 0:
                        r2 = r2_alt
                        w0_r2 = w0_r2a
                        w1_r2 = w1_r2a
                        det = det_alt
                        found = True
                        break
                if not found:
                    return None
            
            v_r1 = (1 * w1_r2 - 1 * w1_r1) / det  # Wait, this is wrong
            # Let me redo:
            # v1 * w0_r1 + v2 * w0_r2 = 1
            # v1 * w1_r1 + v2 * w1_r2 = 1
            # Multiply first by w1_r2, second by w1_r1:
            # v1 * w0_r1 * w1_r2 + v2 * w0_r2 * w1_r2 = w1_r2
            # v1 * w1_r1 * w1_r2 + v2 * w1_r2 * w1_r1 = w1_r1
            # Subtract:
            # v2 * (w0_r2 * w1_r2 - w1_r2 * w1_r1) = w1_r2 - w1_r1
            # That's not right either. Let me use Cramer's rule.
            
            # v1 = (1 * w1_r2 - 1 * w1_r1) / (w0_r1 * w1_r2 - w0_r2 * w1_r1)
            # v2 = (w0_r1 * 1 - w1_r1 * 1) / (w0_r1 * w1_r2 - w0_r2 * w1_r1)
            
            v_r1 = (Fraction(1) * w1_r2 - Fraction(1) * w1_r1) / det
            v_r2 = (w0_r1 * Fraction(1) - w1_r1 * Fraction(1)) / det
            
            # Check if v_r1 and v_r2 are half-integers
            if v_r1.denominator > 2 or v_r2.denominator > 2:
                return None
            
            v_extra[k][r1] = v_r1
            v_extra[k][r2] = v_r2
        
        # Now check if the full system is satisfied
        # For each (i,k): sum_r u_extra[i,r] * v_extra[k,r] * W_base[k*2+i, r] = 1
        # We set u_extra[i,r] = 1 for all i,r
        for i in range(n):
            for k in range(p):
                acc = Fraction(0)
                for r in range(rank_28):
                    if v_extra[k][r] != 0:
                        acc += Fraction(1) * v_extra[k][r] * Fraction(W_base[k * n + i][r])
                if acc != 1:
                    return None
        
        # Build the full 28-term decomposition
        U_full = [row[:] for row in U_base]
        V_full = [row[:] for row in V_base]
        W_full = [row[:] for row in W_base]
        
        for i in range(n):
            for r in range(rank_28):
                U_full[i * m + 2][r] = 1  # u_extra[i,r] = 1
        
        for k in range(p):
            for r in range(rank_28):
                if v_extra[k][r] != 0:
                    V_full[2 * p + k][r] = v_extra[k][r]
        
        candidate = {"U": U_full, "V": V_full, "W": W_full}
        if _validate(candidate):
            return candidate
        return None

    # Try to absorb
    absorbed = _try_absorb(U_28, V_28, W_28)
    if absorbed is not None:
        return absorbed
    
    # If absorption fails, fall back to 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if _validate(candidate_44):
        return candidate_44
    
    return _schoolbook()

# EVOLVE-BLOCK-END
