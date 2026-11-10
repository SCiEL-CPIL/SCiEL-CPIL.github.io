# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses LP optimization over a dense domain to exceed the Möbius-function baseline.
    """
    start_time = time.time()
    deadline = start_time + 550  # Leave 80s margin for safety

    def _elapsed():
        return time.time() - start_time

    # We'll try multiple K values and pick the best LP solution
    # K should be small enough that the constraint matrix (10K x K) is manageable
    # For K=200: matrix is 2000x200 = 400k entries, very fast
    # For K=500: matrix is 5000x500 = 2.5M entries, still fast
    # For K=1000: matrix is 10000x1000 = 10M entries, moderate

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try several K values
    K_candidates = [200, 300, 400, 500, 700, 1000, 1500]

    for K in K_candidates:
        if _elapsed() > deadline - 60:
            break

        k_vals = np.arange(1, K + 1, dtype=np.int64)
        x_max = 10 * K

        # Build constraint matrix A: M(x) = sum_k f(k) * floor(x/k) <= 1 for x in [1, x_max]
        # A has shape (x_max, K), A[i-1, k-1] = floor(i / k)
        # This is a large matrix. For K=1000, x_max=10000, that's 10M entries.
        # Build it efficiently using broadcasting per column.

        A = np.zeros((x_max, K), dtype=np.float64)
        x_range = np.arange(1, x_max + 1, dtype=np.float64)
        for k in range(1, K + 1):
            A[:, k - 1] = x_range // k

        # Objective: maximize -sum f(k) * log(k) / k
        # In linprog (minimize), we minimize sum f(k) * log(k) / k
        log_k = np.log(k_vals.astype(np.float64))
        c_obj = log_k / k_vals.astype(np.float64)  # minimize this dot f

        # Equality constraint: sum f(k) / k = 0
        A_eq = (1.0 / k_vals.astype(np.float64)).reshape(1, -1)
        b_eq = np.array([0.0])

        # Inequality constraints: A @ f <= 1 (all ones)
        b_ub = np.ones(x_max, dtype=np.float64)

        # No explicit bounds on f(k), but clipping to [-10, 10] happens in evaluator
        # To keep LP well-behaved, add loose bounds
        bounds = [(-10.0, 10.0)] * K

        try:
            result = linprog(c_obj, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs',
                             options={'presolve': True, 'time_limit': 60})
            if _elapsed() > deadline - 30:
                break

            if result.success:
                f_vals = result.x
                # Compute score
                score = -np.sum(f_vals * log_k / k_vals.astype(np.float64))
                if score > best_score:
                    best_score = score
                    best_domain = k_vals.copy()
                    best_values = f_vals.copy()
        except Exception:
            continue

    # Fallback: use Möbius function if LP didn't produce anything
    if best_domain is None:
        K_max = 50000
        mu = np.zeros(K_max + 1, dtype=np.int8)
        mu[1] = 1
        is_prime = np.ones(K_max + 1, dtype=bool)
        primes = []
        for i in range(2, K_max + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K_max:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]

        k_vals = np.arange(1, K_max + 1)
        mu_over_k = mu[1:] / k_vals.astype(np.float64)

        best_domain = None
        best_values = None
        best_score = -np.inf

        for K in range(5000, 999, -1000):
            if _elapsed() > deadline - 10:
                break
            S1 = np.sum(mu_over_k[:K])
            if S1 < 0:
                continue
            domain = k_vals[:K].copy()
            values = mu[1:K+1].astype(np.float64).copy()
            values[0] = 1.0 - S1
            x_max = 10 * K
            M_prime = np.zeros(x_max, dtype=np.float64)
            x_range = np.arange(1, x_max + 1)
            for k in range(1, K + 1):
                M_prime += values[k-1] * (x_range // k)
            if np.max(M_prime) <= 1.0001:
                log_k = np.log(k_vals[:K])
                score = -np.sum(values * log_k / k_vals[:K])
                if score > best_score:
                    best_score = score
                    best_domain = domain
                    best_values = values

        if best_domain is None:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        else:
            domain = best_domain
            values = best_values
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
