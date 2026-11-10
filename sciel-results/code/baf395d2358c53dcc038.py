# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k)/k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Strategy: vectorized feasibility check via divisor convolution + iterative
    feasibility-pump repair (zero out largest-violation contributors).

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s limit

    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Compute Möbius function up to K using linear sieve
    def compute_mu(K):
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    # Vectorized M(x) computation using divisor-count convolution
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # = sum_{k in domain} f(k) * sum_{j=1}^{floor(x/k)} 1
    # = sum_{j=1}^{x} (sum_{k in domain, k | j, k <= x} f(k))  ... no that's not right
    # Actually: floor(x/k) = sum_{j=1}^{x} [k divides j and j <= x] ... no
    # floor(x/k) = number of multiples of k in [1, x] = sum_{j: k|j, j<=x} 1
    # So M(x) = sum_{k} f(k) * sum_{j: k|j, j<=x} 1 = sum_{j=1}^{x} sum_{k: k|j, k in domain} f(k)
    # Let g(j) = sum_{k: k|j, k in domain} f(k)  (this is a Dirichlet convolution)
    # Then M(x) = sum_{j=1}^{x} g(j) = prefix sum of g
    #
    # To compute g(j) = sum_{k|j} f(k) for all j up to max_x:
    # For each k in domain, add f(k) to all multiples of k up to max_x.
    # This is O(max_x * log(max_x)) total if done naively, but vectorizable per k.
    # Better: for each k, the multiples are k, 2k, 3k, ... up to max_x.
    # We can do this with a loop over unique k values, but vectorize the addition.

    def compute_M_vectorized(domain, values, max_x):
        """Compute M(x) for x in [0, max_x] using vectorized divisor addition."""
        g = np.zeros(max_x + 1, dtype=np.float64)
        # For each k in domain, add f(k) to g[k], g[2k], g[3k], ...
        # Group by k value for efficiency
        for idx in range(len(domain)):
            k = int(domain[idx])
            fk = values[idx]
            if fk == 0.0:
                continue
            # Add fk to all multiples of k up to max_x
            multiples = np.arange(k, max_x + 1, k)
            g[multiples] += fk
        # M(x) = prefix sum of g
        M = np.cumsum(g)
        return M

    def compute_M_fast(domain, values, max_x):
        """
        Faster M computation:
        M(x) = sum_k f(k) * floor(x/k)
        For small k (k <= sqrt(max_x)), floor(x/k) changes slowly.
        For large k, floor(x/k) is small (1 or 2).
        
        Actually the most efficient vectorized approach:
        For each k, floor(x/k) as a function of x is a step function.
        M(x) = sum_k f(k) * floor(x/k)
        
        We can compute this by: for each k, the contribution to M is f(k) * floor(x/k).
        floor(x/k) = j for x in [j*k, (j+1)*k - 1].
        
        But the divisor-convolution approach above is O(max_x * H(max_x)) which for
        max_x = 10K = 1,000,000 is about 10^6 * 14 = 1.4*10^7 operations, very fast.
        """
        g = np.zeros(max_x + 1, dtype=np.float64)
        for idx in range(len(domain)):
            k = int(domain[idx])
            fk = values[idx]
            if fk == 0.0:
                continue
            # Add fk to g at indices k, 2k, 3k, ...
            g[k::k] += fk
        M = np.cumsum(g)
        return M

    def feasibility_pump(domain, values, max_x, max_iterations=200):
        """
        Iteratively zero out domain elements that cause the largest M(x) violation.
        Returns (feasible_domain, feasible_values, is_feasible).
        """
        active = np.ones(len(domain), dtype=bool)
        d = domain[active].copy()
        v = values[active].copy()
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            if len(d) == 0:
                break
            
            # Compute S = sum v(k)/k
            S = np.sum(v / d)
            f1 = -S
            
            # Compute M(x) for x in [1, max_x]
            M_k_geq_2 = compute_M_fast(d, v, max_x)
            M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
            M[0] = 0.0
            
            # Check constraint
            if len(M) <= 1:
                break
            max_M = np.max(M[1:])
            if max_M <= 1.0001:
                return d, v, True
            
            # Find x where M is maximized
            x_star = int(np.argmax(M[1:]) + 1)
            
            # Compute marginal contribution of each k to M(x_star):
            # M(x_star) = sum_k f(k)*floor(x_star/k) + f1*x_star
            # Removing k changes M by -f(k)*floor(x_star/k) and changes f1 by +f(k)/k
            # New M(x_star) = M(x_star) - f(k)*floor(x_star/k) + (f(k)/k)*x_star
            # = M(x_star) + f(k) * (x_star/k - floor(x_star/k))
            # = M(x_star) + f(k) * (x_star mod k) / k
            #
            # Wait, that's the change. We want to REDUCE M(x_star).
            # If f(k) > 0 and x_star mod k > 0, removing k reduces M by f(k)*(x_star mod k)/k > 0. Good.
            # If f(k) < 0 and x_star mod k > 0, removing k INCREASES M by |f(k)|*(x_star mod k)/k. Bad.
            #
            # Actually let me re-derive:
            # M(x) = sum_{k in D} f(k)*floor(x/k) + f1*x  where f1 = -sum_{k in D} f(k)/k
            # M(x) = sum_k f(k)*floor(x/k) - x * sum_k f(k)/k
            #      = sum_k f(k) * (floor(x/k) - x/k)
            #      = sum_k f(k) * (-(x mod k)/k)
            #      = -sum_k f(k) * (x mod k) / k
            #
            # So M(x) = -sum_k f(k) * (x mod k) / k
            #
            # This is a much cleaner formula! Let me verify:
            # floor(x/k) - x/k = -{x/k} = -(x mod k)/k. Yes!
            #
            # So M(x) = -sum_{k in domain} f(k) * (x mod k) / k
            #
            # This means:
            # - If f(k) > 0 and x mod k > 0: M decreases (contribution is negative)
            # - If f(k) < 0 and x mod k > 0: M increases (contribution is positive)
            #
            # To reduce M(x_star), we should remove k where f(k) < 0 and (x_star mod k)/k is large,
            # OR where f(k) > 0 and (x_star mod k)/k is large (removing a negative contribution makes M larger... wait)
            #
            # M = -sum f(k)*(x mod k)/k
            # Contribution of k to M: c_k = -f(k)*(x mod k)/k
            # If c_k > 0, removing k reduces M by c_k. Good.
            # c_k > 0 means -f(k)*(x mod k)/k > 0, i.e., f(k) < 0 (since (x mod k)/k >= 0).
            # So we should remove k with f(k) < 0 and large (x mod k)/k.
            
            x_mod_k = x_star % d
            contrib = -v * (x_mod_k.astype(np.float64) / d)
            # contrib[k] is the contribution of element k to M(x_star)
            # We want to remove elements with the LARGEST positive contribution (to reduce M most)
            # Only consider elements with contrib > 0
            positive_mask = contrib > 0
            if not np.any(positive_mask):
                # No single removal helps at x_star; try removing the element with largest |contrib|
                # that has f(k) < 0
                neg_mask = v < 0
                if not np.any(neg_mask):
                    # All values are non-negative; removing any will increase M or not help
                    # Remove the one with smallest absolute value to minimize damage
                    min_idx = np.argmin(np.abs(v))
                    active_list = np.where(active)[0]
                    active[active_list[min_idx]] = False
                    continue
                # Among negative f(k), remove the one with largest (x mod k)/k
                neg_contrib = np.where(neg_mask, x_mod_k.astype(np.float64) / d, -np.inf)
                remove_idx = np.argmax(neg_contrib)
                active_list = np.where(active)[0]
                active[active_list[remove_idx]] = False
                continue
            
            # Remove the element with the largest positive contribution
            remove_idx = np.argmax(np.where(positive_mask, contrib, -np.inf))
            active_list = np.where(active)[0]
            active[active_list[remove_idx]] = False
        
        # Final check
        d = domain[active]
        v = values[active]
        if len(d) == 0:
            return d, v, False
        
        S = np.sum(v / d)
        f1 = -S
        M_k_geq_2 = compute_M_fast(d, v, max_x)
        M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
        M[0] = 0.0
        is_feasible = np.max(M[1:]) <= 1.0001
        return d, v, is_feasible

    # Try decreasing K values
    for K in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline:
            return fallback()

        mu = compute_mu(K)
        
        # Domain: k where mu[k] != 0, k >= 2
        mask = (mu != 0)
        mask[0] = False
        mask[1] = False
        domain = np.where(mask)[0]
        values = mu[domain].astype(np.float64)
        m = len(domain)

        if m < 1:
            continue

        max_x = 10 * K

        # First check if the full Möbius set is feasible
        S = np.sum(values / domain)
        f1 = -S
        M_k_geq_2 = compute_M_fast(domain, values, max_x)
        M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
        M[0] = 0.0

        if np.max(M[1:]) <= 1.0001:
            # Feasible as-is!
            a = -np.sum(values * np.log(domain.astype(np.float64)) / domain)
            return domain, values

        # Not feasible; run feasibility pump
        d, v, is_feasible = feasibility_pump(domain, values, max_x, max_iterations=500)
        
        if is_feasible and len(d) >= 1:
            a = -np.sum(v * np.log(d.astype(np.float64)) / d)
            return d, v

    return fallback()


# EVOLVE-BLOCK-END
