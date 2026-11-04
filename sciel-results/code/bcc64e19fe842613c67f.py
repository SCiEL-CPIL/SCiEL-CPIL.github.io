"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _schoolbook_decomp(n, m, p):
    """Return (U, V, W) for schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _perturb_decomp(U, V, W, n, m, p, rank, rng, perturbation_scale=2):
    """Perturb a valid decomposition and try to reduce rank by zeroing some columns."""
    nm, mp, pn = n * m, m * p, p * n
    orig_rank = len(U[0])
    
    # Create perturbed version with fewer columns
    new_U = [[0] * rank for _ in range(nm)]
    new_V = [[0] * rank for _ in range(mp)]
    new_W = [[0] * rank for _ in range(pn)]
    
    # Map original columns to new columns, some may be merged
    col_map = {}
    for c in range(orig_rank):
        if rng.random() < 0.7:  # Keep 70% of columns
            target_col = rng.randint(0, rank - 1)
            col_map[c] = target_col
    
    for c in range(orig_rank):
        if c in col_map:
            tc = col_map[c]
            for i in range(nm):
                new_U[i][tc] += U[i][c]
            for j in range(mp):
                new_V[j][tc] += V[j][c]
            for k in range(pn):
                new_W[k][tc] += W[k][c]
    
    # Add some random perturbation
    for _ in range(rng.randint(0, 3)):
        row_u = rng.randint(0, nm - 1)
        col_u = rng.randint(0, rank - 1)
        new_U[row_u][col_u] += rng.randint(-perturbation_scale, perturbation_scale)
        
        row_v = rng.randint(0, mp - 1)
        col_v = rng.randint(0, rank - 1)
        new_V[row_v][col_v] += rng.randint(-perturbation_scale, perturbation_scale)
        
        row_w = rng.randint(0, pn - 1)
        col_w = rng.randint(0, rank - 1)
        new_W[row_w][col_w] += rng.randint(-perturbation_scale, perturbation_scale)
    
    return new_U, new_V, new_W

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
    
    # Get schoolbook as starting point for structured search
    U_school, V_school, W_school = _schoolbook_decomp(n, m, p)
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        if it % 3 == 0:
            # Every 3rd iteration: perturb schoolbook
            U, V, W_init = _perturb_decomp(U_school, V_school, W_school, n, m, p, rank, rng, perturbation_scale=1)
            # Try to fix W
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
            if ok and _check_half_integer(W, rank) and _validate(U, V, W, n, m, p, rank):
                return U, V, W
        elif it % 3 == 1:
            # Random small integer U,V with wider range
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
                    if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                        ok = False
                        break
                    Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
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
        else:
            # Structured pattern: try circulant-like structures
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        U[i * m + j][c] = (1 if (i + j + c) % 3 == 0 else 0)
            
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        V[j * p + k][c] = (1 if (j + k + c) % 3 == 0 else 0)
            
            if any(all(x == 0 for x in row) for row in U):
                continue
            if any(all(x == 0 for x in row) for row in V):
                continue
            
            W = [[0] * rank for _ in range(pn)]
            ok = True
            for i in range(n):
                for k in range(p):
                    if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                        ok = False
                        break
                    Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
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
    # Try ranks from 27 downward, with more iterations for lower ranks
    for rank in range(27, 20, -1):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        # More iterations for lower ranks since they're harder
        max_iter = int(2000 * (36 - rank))
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # First try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    
    # Try structured/random search for lower rank
    try:
        result = _try_structured_search(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # If we still have time, try more aggressive searches at lower ranks
    remaining = deadline - time.time()
    if remaining > 10:
        for rank in range(24, 27):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 5:
                break
            max_iter = int(remaining * 500)  # Use remaining time
            rng = random.Random(rank * 7777 + 123)
            try:
                result = _search_rank(rank, n, m, p, deadline, rng, max_iter=max_iter)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    return best_result
