"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
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

def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                    return False
    return True

def _solve_w_system(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    # Build system: for each (j,l) in [0,m)x[0,m): sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] = delta_{j,l}
    # Note: l ranges over [0,m) not [0,p) because delta_{j,l} only makes sense for j,l in [0,m)
    # Wait, re-reading the constraint: sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l}
    # Here j in [0,m), l in [0,m) (the inner dimension). But V is indexed by [j*p+k] where j in [0,m), k in [0,p)
    # So V[l*p+k] where l in [0,m), k in [0,p). The constraint is over j,l both in [0,m).
    
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)  # m*m equations
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
    
    # Check consistency
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    # Particular solution: free variables = 0
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _fast_consistency_check(U, V, i, k, n, m, p, rank):
    """Fast numpy-based consistency check for the W system."""
    # Build coefficient matrix as float
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
    
    # RHS
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0
    
    # Check consistency: rank([M|b]) == rank(M)
    M_aug = np.column_stack([M, b])
    rank_M = np.linalg.matrix_rank(M)
    rank_aug = np.linalg.matrix_rank(M_aug)
    if rank_M != rank_aug:
        return False
    return True

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        # Random small integer U,V
        U = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: ensure no row of U or V is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Fast float check first
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
                # Exact solve
                Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        # Check half-integer constraint
        if not _check_half_integer(W, rank):
            continue
        
        # Full exact verification
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
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

def _try_structured_search(n, m, p, deadline):
    """Try structured patterns that might yield lower rank."""
    # Try using Kronecker-like structures
    for rank in range(27, 36):
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=5000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    
    # Try structured/random search for lower rank
    try:
        result = _try_structured_search(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # If we still have time, try more random searches at lower ranks
    remaining = deadline - time.time()
    if remaining > 5:
        for rank in range(25, 27):
            if time.time() > deadline:
                break
            rng = random.Random(rank * 7777 + 123)
            try:
                result = _search_rank(rank, n, m, p, deadline, rng, max_iter=10000)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    return best_result
