"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np
from itertools import product

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
    """Fast numpy-based consistency check for the W system."""
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(Frac(U[i * m + j][c])) * float(Frac(V[l * p + k][c]))
    
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0
    
    M_aug = np.column_stack([M, b])
    rank_M = np.linalg.matrix_rank(M)
    rank_aug = np.linalg.matrix_rank(M_aug)
    if rank_M != rank_aug:
        return False
    return True

def _try_milp_heuristic(n, m, p, rank, deadline, rng):
    """
    Heuristic search inspired by MILP formulation.
    We search over small integer/half-integer values for U, V, W.
    To reduce search space, we fix W to have sparse structure or small values.
    We iterate over candidate W patterns and solve for U, V.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Candidate values for entries
    vals = [-1, -0.5, 0, 0.5, 1]
    
    # Strategy: Try to find a solution where W is very sparse (mostly 0s and 1s)
    # This is a common pattern in low-rank matrix multiplication.
    
    # We will perform a local search / random restart on W, then solve for U and V.
    # Solving for U and V is harder (bilinear), so we stick to solving for W given U, V
    # but we can also try to solve for U given V, W.
    
    # Let's try a different approach: Randomize U and V, solve for W.
    # This is what _search_rank_structured does. 
    # To improve, we can use a more intelligent initialization for U and V.
    
    # For 3x3x4, the rank is likely around 20-25.
    # Let's try to enforce sparsity in U and V more aggressively.
    
    best_score = float('inf')
    best_result = None
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize U and V with sparse random integers
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        
        # Ensure each row has at least one non-zero
        for i in range(nm):
            col = rng.randint(0, rank - 1)
            U[i][col] = rng.choice([-1, 1])
            if rng.random() < 0.2:
                col2 = rng.randint(0, rank - 1)
                if col2 != col:
                    U[i][col2] = rng.choice([-1, 1])
                    
        for j in range(mp):
            col = rng.randint(0, rank - 1)
            V[j][col] = rng.choice([-1, 1])
            if rng.random() < 0.2:
                col2 = rng.randint(0, rank - 1)
                if col2 != col:
                    V[j][col2] = rng.choice([-1, 1])
        
        # Solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
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

def _search_rank_structured(rank, n, m, p, deadline, rng, max_iter):
    """Try structured patterns for rank-`rank` decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Strategy 1: Random small integers
        if it % 3 == 0:
            U = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(nm)]
            V = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(mp)]
        # Strategy 2: Sparse random (more zeros)
        elif it % 3 == 1:
            U = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(mp)]
        # Strategy 3: Structured - use block patterns
        elif it % 5 != 0:
            U = []
            for i in range(n):
                for j in range(m):
                    row = [0] * rank
                    # Assign each (i,j) a primary column
                    primary = (i * m + j) % rank
                    row[primary] = rng.randint(1, 2)
                    # Maybe add a secondary
                    if rank > 1 and rng.random() < 0.3:
                        secondary = (primary + 1) % rank
                        if secondary != primary:
                            row[secondary] = rng.randint(-1, 1)
                    U.append(row)
            
            V = []
            for l in range(m):
                for k in range(p):
                    row = [0] * rank
                    primary = (l * p + k) % rank
                    row[primary] = rng.randint(1, 2)
                    if rank > 1 and rng.random() < 0.3:
                        secondary = (primary + 1) % rank
                        if secondary != primary:
                            row[secondary] = rng.randint(-1, 1)
                    V.append(row)
        # Strategy 4: Very sparse, weight 1
        else:
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            for idx in range(nm):
                col = rng.randint(0, rank - 1)
                U[idx][col] = rng.choice([-1, 1])
            for idx in range(mp):
                col = rng.randint(0, rank - 1)
                V[idx][col] = rng.choice([-1, 1])
        
        # Quick check: ensure no row is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
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
    # We will try ranks from 20 to 35, but give more time to lower ranks.
    
    ranks_to_try = list(range(20, 36))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Determine max_iter based on rank
        # Lower ranks need more exploration
        if rank <= 25:
            max_iter_structured = 5000
            max_iter_milp = 500
        elif rank <= 30:
            max_iter_structured = 3000
            max_iter_milp = 300
        else:
            max_iter_structured = 1000
            max_iter_milp = 100
            
        # Try structured search
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank_structured(rank, n, m, p, deadline, rng, max_iter=max_iter_structured)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try MILP heuristic
        rng = random.Random(rank * 7777 + 123)
        result = _try_milp_heuristic(n, m, p, rank, deadline, rng)
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
        for rank in range(15, 20):
            if time.time() > deadline:
                break
            rng = random.Random(rank * 9999 + 555)
            try:
                result = _search_rank_structured(rank, n, m, p, deadline, rng, max_iter=5000)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    return best_result
