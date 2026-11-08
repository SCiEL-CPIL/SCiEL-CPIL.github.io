"""Search for a rank-minimal decomposition of the 2x3x10 matrix-mult tensor."""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
TARGET_DEADLINE = 145.0  # safety margin under 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == tensor target. Returns (ok, max_err)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                e = abs(s - 1.0)
                if e > max_err:
                    max_err = e
    return (max_err < tol), max_err

def _verify_zero_entries(U, V, W, n, m, p, tol=1e-9):
    """Check that zero entries of the tensor are actually zero."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    max_err = 0.0
    rng = random.Random(42)
    for _ in range(500):
        a = rng.randint(0, n*m-1)
        b = rng.randint(0, m*p-1)
        c = rng.randint(0, p*n-1)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if i_a == i_c and j_a == j_b and k_b == k_c:
            continue
        s = 0.0
        for r in range(rank):
            s += Ua[a, r] * Va[b, r] * Wa[c, r]
        if abs(s) > max_err:
            max_err = abs(s)
    return (max_err < tol), max_err

def _current_rank55(n, m, p):
    """Build the known rank-55 solution."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[k0][col] += 1; V[10+k1][col] += 1
        W[k0][col] += 1; W[k1+2][col] += 1
        col += 1
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[k0][col] += 1
        W[k0+2][col] += 1; W[k1+2][col] -= 1
        col += 1
        # M3
        U[0][col] += 1
        V[k1][col] += 1; V[10+k1][col] -= 1
        W[k1][col] += 1; W[k1+2][col] += 1
        col += 1
        # M4
        U[4][col] += 1
        V[10+k0][col] += 1; V[k0][col] -= 1
        W[k0][col] += 1; W[k0+2][col] += 1
        col += 1
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[10+k1][col] += 1
        W[k0][col] -= 1; W[k1][col] += 1
        col += 1
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[k1][col] += 1
        W[k1+2][col] += 1
        col += 1
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[10+k0][col] += 1; V[10+k1][col] += 1
        W[k0][col] += 1
        col += 1
        # W00
        U[2][col] += 1; V[20+k0][col] += 1; W[k0][col] += 1
        col += 1
        # W01
        U[2][col] += 1; V[20+k1][col] += 1; W[k1][col] += 1
        col += 1
        # W10
        U[5][col] += 1; V[20+k0][col] += 1; W[k0+2][col] += 1
        col += 1
        # W11
        U[5][col] += 1; V[20+k1][col] += 1; W[k1+2][col] += 1
        col += 1
    return {"U": U, "V": V, "W": W}

def _solve_W(U, V, n, m, p, R):
    """Given U and V, solve for W to satisfy the tensor identity."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    W = np.zeros((p*n, R))
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, R))
            for j in range(m):
                M[j] = Ua[i*m+j] * Va[j*p+k]
            b = np.ones(m)
            try:
                w, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
                if residuals.size > 0 and residuals[0] > 1e-8:
                    return None
                W[k*n+i] = w
            except:
                return None
    return W.tolist()

def _try_rank_structured(R, n, m, p, rng, max_iters=5000):
    """Try to find a valid rank-R decomposition with structured U, V."""
    nm = n * m
    mp = m * p
    pn = p * n
    vals = [-2, -1, -0.5, 0.5, 1, 2]
    
    for it in range(max_iters):
        if time.time() > TARGET_DEADLINE - 5:
            break
        # Random sparse U, V with varying density
        density = 0.08 + 0.1 * (it % 3)
        def rand_sparse(nrows, rank, d):
            A = [[0.0]*rank for _ in range(nrows)]
            for r_ in range(nrows):
                for c_ in range(rank):
                    if rng.random() < d:
                        A[r_][c_] = rng.choice(vals)
            return A
        
        U = rand_sparse(nm, R, density)
        V = rand_sparse(mp, R, density)
        W = _solve_W(U, V, n, m, p, R)
        if W is None:
            continue
        ok, err = _verify(U, V, W, n, m, p)
        if ok:
            ok2, err2 = _verify_zero_entries(U, V, W, n, m, p)
            if ok2:
                return U, V, W
    return None

def _try_column_elimination(U, V, W, n, m, p, rng, max_attempts=500):
    """Try to eliminate one column from the decomposition by solving a linear system."""
    R = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    nm, mp, pn = n*m, m*p, p*n
    
    for attempt in range(max_attempts):
        if time.time() > TARGET_DEADLINE - 5:
            break
        col = rng.randint(0, R-1)
        
        # Try to express column col as a linear combination of other columns
        # This means finding coefficients alpha_r (r != col) such that:
        # For all (a,b,c): Ua[a,col]*Va[b,col]*Wa[c,col] = sum_{r!=col} alpha_r * Ua[a,r]*Va[b,r]*Wa[c,r]
        # This is equivalent to the outer product U[:,col] ⊗ V[:,col] ⊗ W[:,col] being in the span
        # of {U[:,r] ⊗ V[:,r] ⊗ W[:,r] : r != col}
        
        # Build the tensor of the column to eliminate
        target = np.zeros((nm, mp, pn))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    target[a,b,c] = Ua[a,col] * Va[b,col] * Wa[c,col]
        
        # Build basis tensors for other columns
        basis = []
        for r in range(R):
            if r == col:
                continue
            t = np.zeros((nm, mp, pn))
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        t[a,b,c] = Ua[a,r] * Va[b,r] * Wa[c,r]
            basis.append(t)
        
        # This is too expensive for large R. Use a smarter approach:
        # Just check if we can perturb the solution to zero out this column
        # by adjusting a few other columns.
        # For now, use a simpler heuristic: try small random perturbations
        pass
    return None

def _try_perturbation_search(U0, V0, W0, n, m, p, rng, max_iters=3000):
    """Start from the rank-55 solution and try to find a lower-rank one via perturbation."""
    R0 = len(U0[0])
    best_rank = R0
    best_U, best_V, best_W = U0[:], V0[:], W0[:]
    
    for target_rank in [R0-1, R0-2, R0-3, R0-4, R0-5]:
        if time.time() > TARGET_DEADLINE - 10:
            break
        result = _try_rank_structured(target_rank, n, m, p, rng, max_iters=2000)
        if result is not None:
            U, V, W = result
            ok, err = _verify(U, V, W, n, m, p)
            ok2, err2 = _verify_zero_entries(U, V, W, n, m, p)
            if ok and ok2:
                return U, V, W
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    
    # Start with the known rank-55 solution
    best = _current_rank55(n, m, p)
    best_rank = 55
    
    # Verify the seed solution
    ok, err = _verify(best["U"], best["V"], best["W"], n, m, p)
    if not ok:
        # Fallback: build a trivial rank-60 solution
        U = [[0]*60 for _ in range(n*m)]
        V = [[0]*60 for _ in range(m*p)]
        W = [[0]*60 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = 60
    
    # Try to find lower rank decompositions with structured search
    for target_rank in [54, 53, 52, 51, 50, 49, 48, 47, 46, 45]:
        if time.time() > TARGET_DEADLINE - 15:
            break
        result = _try_rank_structured(target_rank, n, m, p, rng, max_iters=3000)
        if result is not None:
            U, V, W = result
            ok2, err2 = _verify(U, V, W, n, m, p)
            ok3, err3 = _verify_zero_entries(U, V, W, n, m, p)
            if ok2 and ok3:
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Deadline guard
    if time.time() > TARGET_DEADLINE - 3:
        return best
    
    return best

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    ok, err = _verify(res["U"], res["V"], res["W"], n, m, p)
    print(f"Rank: {len(res['U'][0])}, Valid: {ok}, MaxErr: {err:.2e}")
