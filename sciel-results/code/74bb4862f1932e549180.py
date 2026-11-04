"""Matrix multiplication 4x4 * 4x9: rank-minimal exact tensor decomposition search."""
import time
import random
from fractions import Fraction as F
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 120.0

def _validate_numpy(U, V, W):
    """Fast validation using numpy. U, V, W are lists of lists of int/Fraction.
    Returns True if the decomposition is valid."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    # Determine scale factor (power of 2) to convert to integers
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False  # non-dyadic
                    if d > maxden:
                        maxden = d
    s = 1
    while maxden % s != 0:
        s *= 2
    
    def to_int(mat):
        return np.array([[int(v * s) if isinstance(v, F) else int(v) * s for v in row] for row in mat], dtype=np.int64)
    
    Us = to_int(U)
    Vs = to_int(V)
    Ws = to_int(W)
    s3 = s * s * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i * m + j]
            for k in range(p):
                Vk = Vs[j * p + k]
                Wk = Ws[k * n + i]
                if np.dot(Ui, Vk * Wk) != s3:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # 1) Schoolbook (rank 144) - guaranteed valid
    r, U, V, W = schoolbook()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 2) Random search with structured sampling for better rank
    # Use a seeded RNG for reproducibility within the time budget
    rng = random.Random(42)
    
    # Strategy: Try small ranks first, then increase
    # For 4x4x9, lower bound is max(4*4, 4*9, 4*9) = 36 (but actual lower bound is higher)
    # Strassen-style blocking might get us to ~100-120
    # Let's try random search with sparse matrices
    
    def rand_entry_sparse(density=0.3):
        if rng.random() < density:
            return rng.choice([-2, -1, 1, 2])
        return 0
    
    def rand_entry_frac(density=0.3):
        if rng.random() < density:
            val = rng.choice([-2, -1, 1, 2])
            if rng.random() < 0.3:
                return F(val, 2)
            return val
        return 0
    
    # Phase 1: Search for rank 80-120 with sparse integer entries
    max_tries = 500
    for trial in range(max_tries):
        if time.time() > DEADLINE - 30:
            break
        
        # Vary rank: try different ranges
        if trial < 100:
            r = rng.randint(80, 100)
        elif trial < 250:
            r = rng.randint(100, 120)
        else:
            r = rng.randint(120, 144)
        
        # Vary density
        density = rng.choice([0.15, 0.2, 0.25, 0.3, 0.35])
        
        if trial % 3 == 0:
            gen = lambda: rand_entry_sparse(density)
        elif trial % 3 == 1:
            gen = lambda: rand_entry_frac(density)
        else:
            # Mix: mostly integers, some fractions
            def mixed():
                if rng.random() < 0.7:
                    return rand_entry_sparse(density)
                else:
                    return rand_entry_frac(density * 0.5)
            gen = mixed
        
        U = [[gen() for _ in range(r)] for _ in range(n * m)]
        V = [[gen() for _ in range(r)] for _ in range(m * p)]
        W = [[gen() for _ in range(r)] for _ in range(p * n)]
        
        try:
            if _validate_numpy(U, V, W):
                consider(r, U, V, W)
                # Found a valid one, try to find even better
        except Exception:
            pass
    
    # Phase 2: Local improvement - if we found something, try to reduce rank
    if best is not None:
        best_r, best_U, best_V, best_W = best
        # Try small perturbations
        for _ in range(50):
            if time.time() > DEADLINE - 15:
                break
            r_new = max(60, best_r - rng.randint(1, 5))
            if r_new >= best_r:
                continue
            
            # Copy and zero out some columns
            U_new = [row[:] for row in best_U]
            V_new = [row[:] for row in best_V]
            W_new = [row[:] for row in best_W]
            
            # Zero out random columns from the end
            for c in range(r_new, best_r):
                for i in range(len(U_new)):
                    U_new[i][c] = 0
                for i in range(len(V_new)):
                    V_new[i][c] = 0
                for i in range(len(W_new)):
                    W_new[i][c] = 0
            
            # Add small random noise to remaining entries
            for mat in (U_new, V_new, W_new):
                for row in mat:
                    for idx in range(len(row)):
                        if rng.random() < 0.05:
                            if isinstance(row[idx], F):
                                row[idx] = row[idx] + rng.choice([-F(1,2), F(1,2)])
                            else:
                                row[idx] = row[idx] + rng.choice([-1, 1])
            
            try:
                if _validate_numpy(U_new, V_new, W_new):
                    consider(r_new, U_new, V_new, W_new)
            except Exception:
                pass
    
    if best is None:
        r, U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
