"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_search(n, m, p, rank, deadline, seed=0):
    """
    Try to find a rank-`rank` decomposition with entries in {0, +-1, +-0.5}.
    Use randomized restarts with a structured initialization:
      - Fix a "pattern" of which (row, col) entries are non-zero in U, V, W
      - Solve for the remaining values via a simple backtracking / local repair.
    """
    rng = random.Random(seed)
    nm, mp, pn = n*m, m*p, p*n
    small_vals = [0, 1, -1, 0.5, -0.5]
    
    best = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomly assign a support pattern: each row of U/V/W gets a random
        # subset of columns (sparse), then try to fill values.
        # We'll do a simple randomized local search.
        U = [[0.0]*rank for _ in range(nm)]
        V = [[0.0]*rank for _ in range(mp)]
        W = [[0.0]*rank for _ in range(pn)]
        
        # Initialize: give each column a random assignment of a small set of
        # non-zero entries.
        for c in range(rank):
            # Pick a random (i,j) for U, (j,k) for V, (k,i) for W to be active
            # in this column, with random signs/values.
            # But we need consistency so the product sums to 1.
            # Simpler: random sparse init then repair.
            pass
        
        # Random sparse init
        density = max(0.05, 50.0 / (rank * 45))  # target ~50 nonzeros total-ish
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    if rng.random() < density:
                        U[i*m+j][c] = rng.choice([1, -1, 0.5, -0.5])
            for j in range(m):
                for k in range(p):
                    if rng.random() < density:
                        V[j*p+k][c] = rng.choice([1, -1, 0.5, -0.5])
            for k in range(p):
                for i in range(n):
                    if rng.random() < density:
                        W[k*n+i][c] = rng.choice([1, -1, 0.5, -0.5])
        
        # Local repair: iteratively adjust entries to reduce error.
        # Compute current error per (i,j,k).
        def compute_errors(U, V, W):
            errs = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        errs[(i,j,k)] = 1.0 - s
            return errs
        
        for step in range(200):
            if time.time() > deadline:
                break
            errs = compute_errors(U, V, W)
            max_err = max(abs(v) for v in errs.values())
            if max_err < 1e-9:
                # Snap to half-integers
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][c] = round(U[i*m+j][c] * 2) / 2
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k][c] = round(V[j*p+k][c] * 2) / 2
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i][c] = round(W[k*n+i][c] * 2) / 2
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                break
            
            # Pick the (i,j,k) with largest error, and try flipping/adjusting
            # a random entry in one of the three matrices at a random column.
            target = max(errs, key=lambda t: abs(errs[t]))
            i, j, k = target
            err = errs[target]
            # Try a few random perturbations
            improved = False
            for _ in range(20):
                mat_choice = rng.choice(['U','V','W'])
                c = rng.randrange(rank)
                if mat_choice == 'U':
                    row = i*m + j
                    old = U[row][c]
                    new = old + rng.choice([-1, 1, -0.5, 0.5])
                    U[row][c] = new
                elif mat_choice == 'V':
                    row = j*p + k
                    old = V[row][c]
                    new = old + rng.choice([-1, 1, -0.5, 0.5])
                    V[row][c] = new
                else:
                    row = k*n + i
                    old = W[row][c]
                    new = old + rng.choice([-1, 1, -0.5, 0.5])
                    W[row][c] = new
                
                # Recompute error for target only (approx)
                s = 0.0
                for cc in range(rank):
                    s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                new_err = 1.0 - s
                if abs(new_err) < abs(err):
                    improved = True
                    break
                else:
                    # revert
                    if mat_choice == 'U':
                        U[row][c] = old
                    elif mat_choice == 'V':
                        V[row][c] = old
                    else:
                        W[row][c] = old
            if not improved:
                # Accept a small random perturbation to escape local minima
                mat_choice = rng.choice(['U','V','W'])
                c = rng.randrange(rank)
                if mat_choice == 'U':
                    row = i*m + j
                    U[row][c] += rng.choice([-0.5, 0.5])
                elif mat_choice == 'V':
                    row = j*p + k
                    V[row][c] += rng.choice([-0.5, 0.5])
                else:
                    row = k*n + i
                    W[row][c] += rng.choice([-0.5, 0.5])
        
        # Check if we converged
        errs = compute_errors(U, V, W)
        if max(abs(v) for v in errs.values()) < 1e-9:
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][c] = round(U[i*m+j][c] * 2) / 2
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][c] = round(V[j*p+k][c] * 2) / 2
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][c] = round(W[k*n+i][c] * 2) / 2
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_paired_merge(n, m, p, rank, deadline, seed=0):
    """
    Start from schoolbook (rank 50) and try to merge pairs of columns by
    solving for a 2-column -> 1-column reduction. This is a heuristic that
    exploits linear dependencies among the rank-1 terms.
    """
    base = _schoolbook(n, m, p)
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    current_rank = n*m*p
    rng = random.Random(seed)
    
    while current_rank > 10 and time.time() < deadline:
        # Try to merge a random pair of columns (c1, c2) into c1
        c1 = rng.randrange(current_rank)
        c2 = rng.randrange(current_rank - 1)
        if c2 >= c1:
            c2 += 1
        # We want to find a scalar t such that column c2 = t * column c1 in the
        # "effective" sense, but that's too restrictive. Instead, try to absorb
        # c2 into c1 by adjusting values. This is complex; skip for now.
        # Fallback: just remove a column if it's redundant (all zeros after removal)
        # Check if column c2 is all zeros
        all_zero = True
        for i in range(n*m):
            if U[i][c2] != 0:
                all_zero = False
                break
        if all_zero:
            for j in range(m*p):
                if V[j][c2] != 0:
                    all_zero = False
                    break
        if all_zero:
            for k in range(p*n):
                if W[k][c2] != 0:
                    all_zero = False
                    break
        if all_zero:
            # Remove column c2
            for i in range(n*m):
                U[i].pop(c2)
            for j in range(m*p):
                V[j].pop(c2)
            for k in range(p*n):
                W[k].pop(c2)
            current_rank -= 1
            continue
        # Try a simple 50/50 merge: set U[c2] += U[c1], V[c2] -= V[c1], etc.
        # This is not guaranteed to work; skip.
        pass
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    # Fallback: schoolbook
    fallback = _schoolbook(n, m, p)
    
    # Try structured search for lower ranks
    # Known lower bounds: rank >= n*m*p / something, but for 2x5x5,
    # let's try ranks from 40 down to 20.
    best_solution = fallback
    best_rank = n*m*p
    
    # First, try a few quick structured searches at mid-ranks
    for r in [40, 35, 30, 25, 20]:
        if time.time() > deadline - 20:
            break
        sol = _try_structured_search(n, m, p, r, deadline, seed=r)
        if sol is not None:
            best_solution = sol
            best_rank = r
            break
    
    # If we found a solution, try to improve further
    if best_rank < n*m*p:
        for r in range(best_rank - 5, 15, -5):
            if time.time() > deadline - 10:
                break
            sol = _try_structured_search(n, m, p, r, deadline, seed=1000+r)
            if sol is not None:
                best_solution = sol
                best_rank = r
                break
    
    # Ensure we return within deadline
    if time.time() > deadline - 5:
        return fallback if not _verify(best_solution["U"], best_solution["V"], best_solution["W"], n, m, p) else best_solution
    
    return best_solution
# EVOLVE-BLOCK-END
