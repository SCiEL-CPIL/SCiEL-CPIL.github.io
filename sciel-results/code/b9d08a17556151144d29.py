# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f using the Möbius function on square-free integers.
    """
    t0 = time.time()
    deadline = 600.0
    margin = 30.0

    def time_ok():
        return (time.time() - t0) < deadline - margin

    def mobius_sieve(n):
        mu = np.ones(n + 1, dtype=np.int8)
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility(domain, values, K):
        """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
        x_max = 10 * K
        # M(x) = sum_{i} values[i] * floor(x / domain[i])
        # Evaluate at all x in [1, x_max]
        # Use vectorized computation
        x = np.arange(1, x_max + 1, dtype=np.int64)
        M = np.zeros(x_max, dtype=np.float64)
        
        # Process in chunks to avoid memory issues
        chunk_size = max(1, min(len(domain), 500))
        for start in range(0, len(domain), chunk_size):
            end = min(start + chunk_size, len(domain))
            for i in range(start, end):
                k = domain[i]
                M += values[i] * (x // k)
        
        return np.max(M) <= 1.0001

    def compute_metric(domain, values):
        """Compute a = -sum f(k)*log(k)/k."""
        total = 0.0
        for i in range(len(domain)):
            k = domain[i]
            total += values[i] * np.log(k) / k
        return -total

    best_score = -np.inf
    best_result = None

    # Try increasing K values
    K_candidates = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for K in K_candidates:
        if not time_ok():
            break

        if K > 200000:
            break

        t_start = time.time()

        # Compute mu up to K
        mu = mobius_sieve(K)

        # Domain: all square-free k in [1, K]
        domain = np.array([k for k in range(1, K + 1) if mu[k] != 0], dtype=np.int64)

        if len(domain) == 0:
            continue

        # Compute f(1) for normalization
        sum_rest = 0.0
        for i in range(len(domain)):
            k = domain[i]
            if k == 1:
                continue
            sum_rest += float(mu[k]) / k
        f1 = -sum_rest

        # Construct values
        values = np.zeros(len(domain), dtype=np.float64)
        for i in range(len(domain)):
            k = domain[i]
            if k == 1:
                values[i] = f1
            else:
                values[i] = float(mu[k])

        # Check feasibility
        if not time_ok():
            break

        max_M = np.inf
        # For efficiency, check a subset of x values first
        # Critical points: x = 1, 2, ..., K (where M changes behavior)
        # and x = K, 2K, 3K, ..., 10K
        
        # Quick check: for x < K, M(x) = (f1-1)*x + 1
        # This is linear in x, so max is at x=1 or x=K-1 depending on sign of (f1-1)
        if f1 - 1 >= 0:
            # M(x) increasing in x for x < K, max at x = K-1
            M_check = (f1 - 1) * (K - 1) + 1 if K > 1 else f1
        else:
            # M(x) decreasing in x for x < K, max at x = 1
            M_check = f1
        
        if M_check > 1.0001:
            continue

        # For x >= K, need to check more carefully
        # M(x) = sum_{k in domain} f(k) * floor(x/k)
        # Check x = K, 2K, ..., 10K and some intermediate points
        x_checks = list(range(K, min(K + 100, 10 * K + 1)))
        x_checks += [j * K for j in range(2, 11)]
        x_checks = sorted(set(x_checks))
        
        feasible = True
        x_arr = np.array(x_checks, dtype=np.int64)
        M_vals = np.zeros(len(x_arr), dtype=np.float64)
        
        for i in range(len(domain)):
            k = domain[i]
            M_vals += values[i] * (x_arr // k)
        
        if np.max(M_vals) > 1.0001:
            # Do full check
            if time_ok():
                if not check_feasibility(domain, values, K):
                    feasible = False
            else:
                feasible = False
        
        if feasible:
            a = compute_metric(domain, values)
            if a > best_score:
                best_score = a
                best_result = (domain.copy(), values.copy())
                print(f"K={K}, m={len(domain)}, f1={f1:.6f}, a={a:.6f}", flush=True)

        if time.time() - t_start > 60:
            break

    if best_result is not None:
        return best_result
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values


# EVOLVE-BLOCK-END
