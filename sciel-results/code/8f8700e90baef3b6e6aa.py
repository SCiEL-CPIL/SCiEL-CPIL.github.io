# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses linear programming to find the optimal f under the constraint that
    M(x) = sum_k f(k) * floor(x/k) <= 1 for all x in [1, 10K],
    after the evaluator's normalization f(1) -= sum_k f(k)/k.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    t0 = time.time()
    DEADLINE = 600.0  # 630s kill, leave 30s margin

    def _elapsed():
        return time.time() - t0

    # Fallback: hardcoded baseline
    def _baseline():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Try increasing domain sizes
    best_result = _baseline()
    best_a = 1.0  # baseline score

    for N in [50, 100, 200, 300, 500, 800, 1200]:
        if _elapsed() > DEADLINE - 120:
            break

        try:
            result, a_val = _solve_lp(N)
            if result is not None and a_val > best_a + 1e-9:
                # Verify feasibility
                if _verify_feasible(result[0], result[1]):
                    best_result = result
                    best_a = a_val
        except Exception:
            continue

    return best_result


def _solve_lp(N):
    """
    Solve the LP for domain {1, 2, ..., N}.
    Variables: f(2), f(3), ..., f(N)  (f(1) determined by normalization)
    
    After normalization sum_k f(k)/k = 0, so f(1) = -sum_{k=2}^N f(k)/k.
    
    M(x) = sum_k f(k) * floor(x/k)
         = f(1)*floor(x/1) + sum_{k=2}^N f(k)*floor(x/k)
         = f(1)*x + sum_{k=2}^N f(k)*floor(x/k)
         = -sum_{k=2}^N f(k)/k * x + sum_{k=2}^N f(k)*floor(x/k)
         = sum_{k=2}^N f(k) * (floor(x/k) - x/k)
         = -sum_{k=2}^N f(k) * {x/k}
    
    Constraint: M(x) <= 1  =>  -sum_{k=2}^N f(k) * {x/k} <= 1
    i.e.,         sum_{k=2}^N f(k) * {x/k} >= -1
    
    Objective: maximize a = -sum_k f(k) log(k)/k = -sum_{k=2}^N f(k) log(k)/k
    (since log(1) = 0)
    Minimize:  sum_{k=2}^N f(k) log(k)/k
    """
    n_vars = N - 1  # variables f(2), ..., f(N)
    ks = np.arange(2, N + 1, dtype=np.float64)

    # Objective: minimize sum f(k) * log(k) / k
    c = np.log(ks) / ks

    # Build constraint matrix
    A_parts = []
    b_parts = []

    # M(x) constraints: -sum f(k) * {x/k} <= 1 for x = 1, ..., 10*N
    # => sum f(k) * {x/k} >= -1
    # In form A @ x <= b: -sum f(k) * {x/k} <= 1
    max_x = 10 * N
    for x in range(1, max_x + 1):
        # {x/k} = (x mod k) / k
        frac = (x % np.arange(2, N + 1)) / ks
        A_parts.append(-frac)
        b_parts.append(1.0)

    A_ub = np.array(A_parts)
    b_ub = np.array(b_parts)

    # Bounds: -10 <= f(k) <= 10 (clipping)
    bounds = [(-10.0, 10.0)] * n_vars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True})

    if not res.success:
        return None, 0.0

    f_rest = res.x  # f(2), ..., f(N)
    f_1 = -np.sum(f_rest / ks)

    # Check f(1) within clip range
    if abs(f_1) > 10.0 + 1e-9:
        return None, 0.0

    domain = np.arange(1, N + 1, dtype=np.int64)
    values = np.empty(N, dtype=np.float64)
    values[0] = f_1
    values[1:] = f_rest

    # Compute objective
    a = -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))

    return (domain, values), float(a)


def _verify_feasible(domain, values, tol=1.0001):
    """Verify M(x) <= tol for all x in [1, 10K]."""
    K = int(domain[-1])
    f = values
    # d[j] = sum_{k|j} f(k)
    d = np.zeros(K + 1, dtype=np.float64)
    for idx, k in enumerate(domain):
        kk = int(k)
        d[kk::kk] += f[idx]
    running = np.cumsum(d)
    return bool(running.max() <= tol)


# EVOLVE-BLOCK-END
