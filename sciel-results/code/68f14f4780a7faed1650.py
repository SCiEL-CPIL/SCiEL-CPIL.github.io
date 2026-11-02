"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS
    
    # Try rank 18 first
    if time.time() < deadline:
        result = _find_decomposition(n, m, p, 18)
        if result is not None:
            return result
    
    # Try rank 19
    if time.time() < deadline:
        result = _find_decomposition(n, m, p, 19)
        if result is not None:
            return result
    
    # Fallback to schoolbook (rank 20)
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

def _find_decomposition(n, m, p, rank):
    import numpy as np
    from fractions import Fraction
    import random
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
    
    # Try multiple random initializations
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        
        W = np.zeros((pn, rank))
        success = True
        
        for c in range(pn):
            M = np.zeros((nm * mp, rank))
            for a in range(nm):
                for b in range(mp):
                    for r in range(rank):
                        M[a * mp + b, r] = U[a, r] * V[b, r]
            
            t = np.zeros(nm * mp)
            for (a, b, cc) in valid:
                if cc == c:
                    t[a * mp + b] = 1
            
            Wc, residuals, rank_c, sv = np.linalg.lstsq(M, t, rcond=None)
            W[c, :] = Wc
            
            if not np.allclose(M @ Wc, t, atol=1e-8):
                success = False
                break
        
        if not success:
            continue
        
        # Round to rationals
        U_frac = [[Fraction(U[a, r]).limit_denominator(1000) for r in range(rank)] for a in range(nm)]
        V_frac = [[Fraction(V[b, r]).limit_denominator(1000) for r in range(rank)] for b in range(mp)]
        W_frac = [[Fraction(W[c, r]).limit_denominator(1000) for r in range(rank)] for c in range(pn)]
        
        # Verify
        ok = True
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    expected = 1 if (a, b, c) in valid else 0
                    actual = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(rank))
                    if abs(float(actual - expected)) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None
# EVOLVE-BLOCK-END
