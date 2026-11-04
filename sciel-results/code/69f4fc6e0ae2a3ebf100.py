"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
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
        # Target tensor T[i, a, k] = 1 for all i, a, k
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
    # Vectors for U (i,a), V (a,k), W (k,i)
    # Indices: U: 00,01,10,11 -> 0,1,2,3
    #          V: 00,01,10,11 -> 0,1,2,3
    #          W: 00,01,10,11 -> 0,1,2,3 (k*n+i, n=2)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
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

    # Build candidate with Strassen for 2x2 blocks
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0,1. Cols 0..7.
    # Split B1 cols into 4 blocks of 2: (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            # U indices: i in 0,1; a in 0,1 (global a = local a)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            # V indices: a in 0,1 (global a = local a); k in k0, k1 (global k = k0 + lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            # W indices: k in k0, k1 (global k = k0 + lk); i in 0,1 (global i = li)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 uses col 2 of A (global a=2).
    # B2 uses row 2 of B (global a=2).
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

    # Now apply linear algebraic compression to reduce rank
    # We look for linear dependencies among the rank-1 terms
    # A term c is a triple (u_c, v_c, w_c) where u_c is a vector in R^(n*m), etc.
    # Two terms c1, c2 are linearly dependent if there exists alpha != 0 such that
    # u_{c1} = alpha * u_{c2}, v_{c1} = alpha * v_{c2}, w_{c1} = alpha * w_{c2}
    # More generally, we can try to find a linear combination of terms that cancels out
    
    def _compress(U, V, W, max_time=10.0):
        """Try to reduce rank by finding and eliminating linearly dependent terms."""
        from fractions import Fraction
        rank = len(U[0])
        n_rows_u = len(U)
        n_rows_v = len(V)
        n_rows_w = len(W)
        
        # Convert to Fraction for exact arithmetic
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        # Try to find terms that can be eliminated
        # A term c can be eliminated if it's a linear combination of other terms
        # in a way that preserves the tensor decomposition
        
        # Simpler approach: check if any two terms are scalar multiples
        # If term c1 = alpha * term c2 for some alpha, we can merge them
        # but this doesn't reduce rank unless alpha=1 and we can cancel
        
        # Better: look for pairs of terms where one is negative of the other
        # or where we can do a change of variables
        
        # For now, let's just check for exact duplicates or negatives
        # that can be combined
        used = [False] * rank
        new_U = []
        new_V = []
        new_W = []
        
        for c in range(rank):
            if used[c]:
                continue
            # Check if this term is a scalar multiple of any previous term
            merged = False
            for c2 in range(len(new_U)):
                # Check if term c is scalar multiple of term c2
                # Find alpha such that U_frac[c] = alpha * new_U[c2]
                alpha = None
                valid = True
                for i in range(n_rows_u):
                    if new_U[c2][i] == 0:
                        if U_frac[c][i] != 0:
                            valid = False
                            break
                    else:
                        a_candidate = U_frac[c][i] / new_U[c2][i]
                        if alpha is None:
                            alpha = a_candidate
                        elif alpha != a_candidate:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Verify alpha works for V and W
                for i in range(n_rows_v):
                    if new_V[c2][i] != 0:
                        if V_frac[c][i] != alpha * new_V[c2][i]:
                            valid = False
                            break
                if not valid:
                    continue
                for i in range(n_rows_w):
                    if new_W[c2][i] != 0:
                        if W_frac[c][i] != alpha * new_W[c2][i]:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Term c is alpha * term c2
                # We can't simply merge them in a rank-1 decomposition
                # unless we're doing a more sophisticated transformation
                # For now, skip this optimization as it's complex
                break
            
            new_U.append(U_frac[c])
            new_V.append(V_frac[c])
            new_W.append(W_frac[c])
            used[c] = True
        
        if len(new_U) < rank:
            # Convert back to int/float
            new_U_int = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_U]
            new_V_int = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_V]
            new_W_int = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in new_W]
            return new_U_int, new_V_int, new_W_int
        
        return U, V, W
    
    # Try compression
    if _elapsed_ok():
        U_c, V_c, W_c = _compress(U, V, W)
        candidate_c = {"U": U_c, "V": V_c, "W": W_c}
        if _validate(candidate_c):
            if len(U_c[0]) < len(U[0]):
                return candidate_c
    
    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END

