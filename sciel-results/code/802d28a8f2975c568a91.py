# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
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

    def compute_metric(domain, values):
        """Compute the metric a = -sum_k f(k) * log(k) / k after clipping and f(1) adjustment."""
        # Clip values to [-10, 10]
        clipped = np.clip(values, -10, 10)
        
        # Compute S = sum_k f(k)/k over clipped values
        S = np.sum(clipped / domain.astype(np.float64))
        
        # Adjust f(1): f(1) = f(1) - S (f(1) = 0 if 1 not in domain)
        if 1 in domain:
            idx_1 = np.where(domain == 1)[0][0]
            clipped[idx_1] = clipped[idx_1] - S
        else:
            # If 1 is not in domain, we need to add it with value -S
            # But actually the contract says f(1) = 0 before this if 1 is not in domain
            # So we just compute the metric with the clipped values as-is,
            # but the constraint sum_k f(k)/k = 0 is enforced by the evaluator
            # For our metric computation, we need to account for this
            pass
        
        # Recompute S after adjustment to verify
        S_new = np.sum(clipped / domain.astype(np.float64))
        
        # Metric: a = -sum_k f(k) * log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(clipped * log_k / domain.astype(np.float64))
        
        return a

    def check_feasibility(domain, values, K, batch_size=5000):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using vectorized ops."""
        max_x = 10 * K
        m = len(domain)
        
        # Clip values first
        clipped = np.clip(values, -10, 10)
        
        # Compute S = sum_k f(k)/k
        S = np.sum(clipped / domain.astype(np.float64))
        
        # Adjust f(1)
        if 1 in domain:
            idx_1 = np.where(domain == 1)[0][0]
            clipped[idx_1] = clipped[idx_1] - S
        else:
            # f(1) = 0 - S = -S, but 1 is not in domain so this term doesn't contribute
            # Actually, if 1 is not in domain, f(1) is treated as 0 before adjustment,
            # so after adjustment f(1) = -S, but since 1 is not in domain, it doesn't
            # contribute to M(x) or the metric. Wait, let me re-read...
            # "f(1) = 0 before this if 1 is not in the domain"
            # So if 1 is not in domain, f(1) starts as 0, then becomes 0 - S = -S
            # But if 1 is not in the domain, does f(1) contribute to M(x)?
            # M(x) = sum_k f(k) * floor(x/k) - if 1 is not in domain, k=1 is not summed
            # So f(1) doesn't contribute. But the constraint sum_k f(k)/k = 0 must hold.
            # If 1 is not in domain, then sum over k in domain of f(k)/k must equal 0
            # after the adjustment. But the adjustment is f(1) = f(1) - S, and if 1 is
            # not in domain, this doesn't change any values in the domain.
            # Hmm, this is confusing. Let me assume that if 1 is not in domain, the
            # constraint is automatically satisfied by construction or we need to ensure
            # sum_k f(k)/k = 0 for k in domain.
            pass
        
        # For x <= K, M(x) = sum_{k in domain, k <= x} f(k) * floor(x/k)
        # We need to check all x in [1, 10K]
        
        feasible = True
        for batch_start in range(1, max_x + 1, batch_size):
            if time.time() - start_time > deadline - 60:
                return False  # timeout, assume infeasible to be safe
            
            batch_end = min(batch_start + batch_size - 1, max_x)
            xs = np.arange(batch_start, batch_end + 1, dtype=np.int64)
            
            # Process in sub-batches to avoid memory issues
            sub_batch = 200
            for sb_start in range(0, len(xs), sub_batch):
                if time.time() - start_time > deadline - 60:
                    return False
                sb_end = min(sb_start + sub_batch, len(xs))
                xs_sub = xs[sb_start:sb_end]
                # shape: (n_x, m)
                floor_mat = (xs_sub[:, None] // domain[None, :])
                M_batch = np.sum(clipped[None, :] * floor_mat, axis=1)
                if np.any(M_batch > 1.0001):
                    return False
        
        return True

    def solve_lp(K, max_constraints=200):
        """
        Solve LP to maximize a = -sum_k f(k) * log(k) / k
        subject to M(x) <= 1.0001 for sampled x values.
        """
        if time.time() - start_time > deadline - 30:
            return None, None, -np.inf
        
        mu = mobius_sieve(K)
        # Use only indices where mu != 0, and include 1
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        if 1 not in nonzero_idx:
            nonzero_idx = np.concatenate([[1], nonzero_idx])
        
        domain = np.sort(nonzero_idx).astype(np.int64)
        m = len(domain)
        
        if m < 1 or m > 5000:
            return None, None, -np.inf
        
        # Objective: maximize a = -sum_k f(k) * log(k) / k
        # = minimize sum_k f(k) * log(k) / k
        log_k = np.log(domain.astype(np.float64))
        c = log_k / domain.astype(np.float64)
        
        # Constraints: M(x) = sum_k f(k) * floor(x/k) <= 1.0001
        # Sample x values from [1, 10K]
        max_x = 10 * K
        if max_x <= K:
            x_samples = np.array([1])
        else:
            # Sample x values, including 1 and some spread across [1, 10K]
            n_samples = min(max_constraints, max_x)
            x_samples = np.linspace(1, max_x, n_samples, dtype=np.int64)
            x_samples = np.unique(x_samples)
        
        n_constraints = len(x_samples)
        
        # Build A_ub: each row is floor(x_i / k) for k in domain
        A_ub = np.zeros((n_constraints, m), dtype=np.float64)
        for i, x in enumerate(x_samples):
            A_ub[i, :] = x // domain
        
        b_ub = np.full(n_constraints, 1.0001)
        
        # Bounds: f(k) in [-10, 10]
        bounds = [(-10, 10)] * m
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not result.success:
                return None, None, -np.inf
            
            values = result.x
            
            # Compute metric
            metric = compute_metric(domain, values)
            
            # Check full feasibility
            if time.time() - start_time > deadline - 60:
                return None, None, -np.inf
            
            feasible = check_feasibility(domain, values, K)
            if not feasible:
                return None, None, -np.inf
            
            return domain, values, metric
        
        except Exception:
            return None, None, -np.inf

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try different K values with LP
    for K in [500, 1000, 2000, 5000, 10000]:
        if time.time() - start_time > deadline - 60:
            break
        
        domain, values, metric = solve_lp(K)
        
        if domain is not None and metric > best_score:
            best_score = metric
            best_domain = domain.copy()
            best_values = values.copy()
        
        # Also try the raw Mobius function as a fallback
        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        if len(nonzero_idx) > 0:
            mob_domain = nonzero_idx.astype(np.int64)
            mob_values = mu[nonzero_idx].astype(np.float64)
            
            # Check feasibility
            if time.time() - start_time > deadline - 60:
                continue
            
            feasible = check_feasibility(mob_domain, mob_values, K)
            if feasible:
                metric = compute_metric(mob_domain, mob_values)
                if metric > best_score:
                    best_score = metric
                    best_domain = mob_domain.copy()
                    best_values = mob_values.copy()

    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
