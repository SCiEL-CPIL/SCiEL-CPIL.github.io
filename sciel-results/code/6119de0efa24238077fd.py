"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import math
from fractions import Fraction

try:
    import sympy as sp
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Return dict mapping (i,j,k) -> 1 for the standard matmul tensor."""
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i, j, k)] = 1
    return target

def _tensor_from_factors(U, V, W, n, m, p, rank):
    """Compute tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                T[(i, j, k)] = s
    return T

def _verify_exact(U, V, W, n, m, p, rank, target):
    """Check if the tensor from factors exactly matches target (integer/half-integer)."""
    if not HAVE_SYMPY:
        # Fallback: check with Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        a = U[i * m + j][c]
                        b = V[j * p + k][c]
                        d = W[k * n + i][c]
                        # Convert to Fraction
                        fa = Fraction(a) if not isinstance(a, Fraction) else a
                        fb = Fraction(b) if not isinstance(b, Fraction) else b
                        fd = Fraction(d) if not isinstance(d, Fraction) else d
                        s += fa * fb * fd
                    if s != Fraction(target[(i, j, k)]):
                        return False
        return True
    else:
        # Use sympy for exact rational arithmetic
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sp.Integer(0)
                    for c in range(rank):
                        a = U[i * m + j][c]
                        b = V[j * p + k][c]
                        d = W[k * n + i][c]
                        fa = sp.Rational(a) if not isinstance(a, (sp.Rational, sp.Integer)) else a
                        fb = sp.Rational(b) if not isinstance(b, (sp.Rational, sp.Integer)) else b
                        fd = sp.Rational(d) if not isinstance(d, (sp.Rational, sp.Integer)) else d
                        s += fa * fb * fd
                    if s != sp.Integer(target[(i, j, k)]):
                        return False
        return True

def _try_exact_search(n, m, p, rank, time_budget, rng):
    """Try to find an exact rank-`rank` decomposition via random init + gradient + exact check."""
    target = _build_target_tensor(n, m, p)
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = nm * rank + mp * rank + pn * rank

    # Use rational random initialization with small values
    # Try several random seeds
    attempts = 0
    max_attempts = 50
    deadline = time.time() + time_budget

    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        # Random rational initialization
        U = [[Fraction(rng.randint(-3, 3)) for _ in range(rank)] for _ in range(nm)]
        V = [[Fraction(rng.randint(-3, 3)) for _ in range(rank)] for _ in range(mp)]
        W = [[Fraction(rng.randint(-3, 3)) for _ in range(rank)] for _ in range(pn)]

        # Compute current tensor
        T = _tensor_from_factors(U, V, W, n, m, p, rank)

        # Compute error
        err_sq = Fraction(0)
        for key in target:
            diff = T[key] - target[key]
            err_sq += diff * diff

        if err_sq == 0:
            return U, V, W

        # Simple gradient step (stochastic)
        # Gradient of err_sq w.r.t. U[i*m+j,c] = 2 * (T[i,j,k] - target[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
        # But we need to sum over all k for each (i,j,c)
        # Let's do a few gradient steps
        for step in range(10):
            if time.time() > deadline:
                break
            # Compute gradient
            grad_U = [[Fraction(0) for _ in range(rank)] for _ in range(nm)]
            grad_V = [[Fraction(0) for _ in range(rank)] for _ in range(mp)]
            grad_W = [[Fraction(0) for _ in range(rank)] for _ in range(pn)]

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = T[(i, j, k)] - target[(i, j, k)]
                        if diff == 0:
                            continue
                        # d(err)/d(U[i*m+j,c]) = 2*diff * V[j*p+k,c] * W[k*n+i,c]
                        # d(err)/d(V[j*p+k,c]) = 2*diff * U[i*m+j,c] * W[k*n+i,c]
                        # d(err)/d(W[k*n+i,c]) = 2*diff * U[i*m+j,c] * V[j*p+k,c]
                        for c in range(rank):
                            u_val = U[i * m + j][c]
                            v_val = V[j * p + k][c]
                            w_val = W[k * n + i][c]
                            grad_U[i * m + j][c] += 2 * diff * v_val * w_val
                            grad_V[j * p + k][c] += 2 * diff * u_val * w_val
                            grad_W[k * n + i][c] += 2 * diff * u_val * v_val

            # Take a small step
            lr = Fraction(1, 100)
            for i in range(nm):
                for c in range(rank):
                    U[i][c] -= lr * grad_U[i][c]
            for j in range(mp):
                for c in range(rank):
                    V[j][c] -= lr * grad_V[j][c]
            for k in range(pn):
                for c in range(rank):
                    W[k][c] -= lr * grad_W[k][c]

            # Recompute tensor
            T = _tensor_from_factors(U, V, W, n, m, p, rank)
            err_sq = Fraction(0)
            for key in target:
                diff = T[key] - target[key]
                err_sq += diff * diff

            if err_sq == 0:
                return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60

    # Start with schoolbook as safe fallback
    rank = schoolbook_rank
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

    # Try to find lower rank decompositions
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin

    rng = random.Random(42)

    # Try ranks from lower to schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also rank >= n+m+p-2 = 2+5+6-2 = 11 (weak)
    # For 2x5x6, known best is likely around 25-35
    for try_rank in range(25, schoolbook_rank):
        if time.time() > deadline:
            break
        time_budget_for_rank = min(10, deadline - time.time())
        if time_budget_for_rank < 2:
            break

        result = _try_exact_search(n, m, p, try_rank, time_budget_for_rank, rng)
        if result is not None:
            U, V, W = result
            rank = try_rank
            # Continue trying even lower
            continue
        else:
            # If we couldn't find for this rank, try next
            continue

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
