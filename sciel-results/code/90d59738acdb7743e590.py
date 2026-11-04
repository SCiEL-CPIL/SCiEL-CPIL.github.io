"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np
import sys

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

def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
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

def _fast_consistency_check(U, V, i, k, n, m, p, rank):
    """Fast exact consistency check using rational arithmetic."""
    # Build the system M @ w = b where M[j*m+l, c] = U[i*m+j][c] * V[l*p+k][c]
    # and b[j*m+l] = 1 if j==l else 0
    # We check if b is in the column space of M
    # Use Gaussian elimination with exact fractions for small systems
    M = [[Frac(0)] * rank for _ in range(m * m)]
    b = [Frac(0)] * (m * m)
    for j in range(m):
        for l in range(m):
            b[j * m + l] = Frac(1) if j == l else Frac(0)
            for c in range(rank):
                M[j * m + l][c] = Frac(U[i * m + j][c]) * Frac(V[l * p + k][c])
    
    # Gaussian elimination to check consistency
    R = m * m
    C = rank
    aug = [M[r][:] + [b[r]] for r in range(R)]
    
    col = 0
    for r in range(min(R, C)):
        if col >= C:
            break
        # Find pivot
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
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b_val for a, b_val in zip(aug[rr], aug[r])]
        col += 1
    
    # Check for inconsistency: any row with all zeros in M but nonzero in b
    for rr in range(R):
        if all(aug[rr][c] == 0 for c in range(C)) and aug[rr][C] != 0:
            return False
    return True

def _fast_consistency_check_numpy(U, V, i, k, n, m, p, rank):
    """Fast float consistency check using numpy."""
    M = np.zeros((m*m, rank), dtype=float)
    b = np.zeros(m*m, dtype=float)
    for j in range(m):
        for l in range(m):
            b[j*m+l] = 1.0 if j == l else 0.0
            for c in range(rank):
                M[j*m+l, c] = float(U[i*m+j][c]) * float(V[l*p+k][c])
    try:
        # Check if b is in the column space of M
        # Use matrix rank with a tight tolerance
        r_M = np.linalg.matrix_rank(M, tol=1e-9)
        r_aug = np.linalg.matrix_rank(np.hstack([M, b.reshape(-1, 1)]), tol=1e-9)
        if r_M == r_aug:
            return True
    except:
        pass
    return False

def _search_rank_structured(rank, n, m, p, deadline, rng, max_iter):
    """Try structured patterns for rank-`rank` decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Strategy 1: Random small integers {-1, 0, 1}
        if it % 2 == 0:
            U = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(mp)]
        # Strategy 2: Sparse random
        else:
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            for idx in range(nm):
                if rng.random() < 0.7:
                    col = rng.randint(0, rank - 1)
                    U[idx][col] = rng.choice([-1, 1])
            for idx in range(mp):
                if rng.random() < 0.7:
                    col = rng.randint(0, rank - 1)
                    V[idx][col] = rng.choice([-1, 1])
        
        # Quick check: ensure no row is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency using numpy first
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check_numpy(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
            if not ok:
                break
        
        if not ok:
            continue
            
        # If float check passes, do exact check
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _search_rank_with_completion(rank, n, m, p, deadline, rng, max_iter):
    """Search with completion: find U,V that work for most (i,k), then try to fix the rest."""
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
        failed_ik = []
        ok = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    failed_ik.append((i, k))
                    continue
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    failed_ik.append((i, k))
                    continue
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
        
        if len(failed_ik) > 0:
            # Try to adjust U or V to fix the failed cases
            # This is a heuristic - just skip for now
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
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

def _try_known_patterns(n, m, p, deadline):
    """Try known structured decompositions that might give lower rank."""
    # Prioritize lower ranks as they are the goal
    # Focus more budget on lower ranks where improvements are most likely
    ranks_to_try = list(range(16, 32))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Determine max_iter based on rank
        # With numpy speedup, we can try many more iterations
        if rank <= 20:
            max_iter_structured = 100000
        elif rank <= 24:
            max_iter_structured = 50000
        else:
            max_iter_structured = 20000
            
        # Try structured search
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank_structured(rank, n, m, p, deadline, rng, max_iter=max_iter_structured)
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
        result = _try_known_patterns(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # If we still have time, try more aggressive searches at lower ranks
    remaining = deadline - time.time()
    if remaining > 10:
        for rank in range(16, 24):
            if time.time() > deadline:
                break
            rng = random.Random(rank * 9999 + 555)
            try:
                result = _search_rank_structured(rank, n, m, p, deadline, rng, max_iter=100000)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    break  # Found a good one, stop
            except Exception:
                pass
    
    return best_result
