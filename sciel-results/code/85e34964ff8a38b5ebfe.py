"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] == delta_{j,l} for all i,k,j,l."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _solve_w_system(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1
    
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _search_rank_base(n, m, p, rank, deadline, rng, max_iter):
    """Search for rank decomposition of M_{n,m,p}."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        U = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)]
        
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
            
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
            
    return None

def _extend_333_to_334(U_base, V_base, W_base, rank_base):
    """Extend a 3x3x3 decomposition to 3x3x4 by adding 3 columns."""
    n, m, p = 3, 3, 4
    rank = rank_base + 3
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Copy base
    for i in range(nm):
        for c in range(rank_base):
            U[i][c] = U_base[i][c]
    for j in range(m):
        for k in range(3):
            for c in range(rank_base):
                V[j * p + k][c] = V_base[j * 3 + k][c]
    for k in range(3):
        for i in range(n):
            for c in range(rank_base):
                W[k * n + i][c] = W_base[k * n + i][c]
                
    # Add extension
    for j in range(m):
        c = rank_base + j
        for i in range(n):
            U[i * m + j][c] = 1
        for l in range(m):
            V[l * p + 3][c] = 1 if l == j else 0
        for i in range(n):
            W[3 * n + i][c] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Try to find a rank-24 decomposition for 3x3x3, then extend to rank 27
    rng = random.Random(42)
    base_result = _search_rank_base(3, 3, 3, 24, deadline, rng, max_iter=5000)
    if base_result is not None:
        U_base, V_base, W_base = base_result
        U, V, W = _extend_333_to_334(U_base, V_base, W_base, 24)
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Search for rank 27 directly on 3x3x4
    rng2 = random.Random(123)
    result = _search_rank_base(n, m, p, 27, deadline, rng2, max_iter=5000)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
