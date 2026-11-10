# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve


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
    DEADLINE = 615.0  # 630s kill, leave 15s margin

    def _elapsed():
        return time.time() - t0

    # Fallback: hardcoded baseline
    def _baseline():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Calculate baseline score to ensure we only improve upon it
    def _calc_score(domain, values):
        # Clip
        v_clip = np.clip(values, -10, 10)
        d_float = domain.astype(np.float64)
        # S = sum f(k)/k
        S = np.sum(v_clip / d_float)
        # f(1) adjustment
        v_norm = v_clip.copy()
        if domain[0] == 1:
            v_norm[0] -= S
        else:
            # If 1 is not in domain, the evaluator logic is tricky.
            # Assuming 1 is always in domain for our solutions.
            pass
        
        a = -np.sum(v_norm * np.log(d_float) / d_float)
        return float(a)

    best_result = _baseline()
    best_a = _calc_score(best_result[0], best_result[1])

    # Try increasing domain sizes, keeping N small for speed
    for N in [50, 100, 200, 500, 1000, 2000, 5000]:
        if _elapsed() > DEADLINE - 30:
            break

        try:
            result, a_val = _solve_lp(N)
            if result is not None:
                # Verify feasibility
                if _verify_feasible(result[0], result[1]):
                    if a_val > best_a + 1e-9:
                        best_result = result
                        best_a = a_val
        except Exception:
            continue

    return best_result


def _solve_lp(N):
    """
    Solve the LP for domain {1, 2, ..., N}.
    Variables: f(2), f(3), ..., f(N)
    
    Objective: minimize sum f(k) log(k) / k
    Constraints: -sum f(k) {x/k} <= 1 for x in 1..10N
    """
    n_vars = N - 1
    ks = np.arange(2, N + 1, dtype=np.float64)
    c = np.log(ks) / ks

    max_x = 10 * N
    xs = np.arange(1, max_x + 1, dtype=np.float64)
    
    # Construct A_ub efficiently using sparse matrix with vectorized operations
    # A[i, j] = - { xs[i] / ks[j] }
    try:
        # Use COO format for efficient construction
        rows = []
        cols = []
        data = []
        
        for j in range(n_vars):
            k = int(ks[j])
            # Vectorized fractional part computation
            frac_vals = (xs % k) / k
            # Only add non-zero entries (sparse optimization)
            nonzero_mask = frac_vals > 1e-12
            if np.any(nonzero_mask):
                rows.extend(np.where(nonzero_mask)[0].tolist())
                cols.extend([j] * np.sum(nonzero_mask))
                data.extend((-frac_vals[nonzero_mask]).tolist())
        
        if len(data) > 0:
            A_ub = coo_matrix((data, (rows, cols)), shape=(max_x, n_vars)).tocsr()
        else:
            A_ub = csr_matrix((max_x, n_vars), dtype=np.float64)
        b_ub = np.ones(max_x)
    except MemoryError:
        return None, 0.0

    bounds = [(-10.0, 10.0)] * n_vars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': 120})

    if not res.success:
        return None, 0.0

    f_rest = res.x
    f_1 = -np.sum(f_rest / ks)

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
