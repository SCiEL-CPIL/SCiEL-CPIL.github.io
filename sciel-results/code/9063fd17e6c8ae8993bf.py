# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
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


def _compute_M_fast(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x].
    Uses a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        max_j = max_x // k
        if max_j == 0:
            continue
        
        # For each j=1..max_j, floor(x/k)=j for x in [j*k, (j+1)*k - 1]
        # So we add f_k*j to diff[j*k] and subtract at diff[(j+1)*k]
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        ends = np.minimum(np.arange(2, max_j + 2, dtype=np.int64) * k, max_x + 1)
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[1:max_x + 1]


def _evaluate_candidate(domain, values, max_x):
    """
    Evaluate a candidate (domain, values):
    - Clip to [-10, 10]
    - Apply S-correction so that sum f(k)/k = 0
    - Check M(x) <= 1.0001 for all x in [1, max_x]
    - Compute objective a = -sum f(k)*log(k)/k
    Returns (domain_corrected, values_corrected, a) or None if infeasible.
    """
    k_float = domain.astype(np.float64)
    
    # Clip to [-10, 10]
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k on clipped values
    S = np.sum(values_clipped / k_float)
    
    # Apply S correction: f(1) = f(1) - S
    values_corrected = values_clipped.copy()
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values_corrected[idx_1[0]] = values_clipped[idx_1[0]] - S
    else:
        # If 1 is not in domain, we can't adjust f(1), so skip this candidate
        return None
    
    # Compute M(x) for x in [1, max_x]
    M = _compute_M_fast(domain, values_corrected, max_x)
    
    # Check constraint: M(x) <= 1.0001 for all x in [1, max_x]
    if len(M) > 0 and np.max(M) > 1.0001:
        return None
    
    # Compute objective
    a = -np.sum(values_corrected * np.log(k_float) / k_float)
    
    return (domain, values_corrected, a)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 90.0

    best_result = None
    best_a = -np.inf

    # Strategy: Use the Möbius function μ(k) which satisfies
    # M(x) = sum_{k=1}^{x} μ(k) floor(x/k) = 1 for all x
    # and -sum μ(k) log(k)/k → 1 as K → ∞
    
    # Try increasing K values, starting with the largest for best score
    for K in [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000]:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Compute Möbius function
        mu = _compute_mobius(K)
        
        # Include all k where mu(k) != 0 to keep m smaller
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)
        
        if len(domain) == 0:
            continue
        
        max_x = 10 * K
        
        # Try the raw Möbius function (S-correction is applied in _evaluate_candidate)
        result = _evaluate_candidate(domain, values, max_x)
        if result is not None:
            dom, vals, a = result
            if a > best_a:
                best_a = a
                best_result = (dom, vals)
        
        # Try scaled versions: f(k) = c * mu(k)
        # The S-correction adjusts f(1) to make sum f(k)/k = 0
        # For small c, the constraint M(x) <= 1.0001 is easier to satisfy
        for c in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            if time.time() - start_time > deadline - safety_margin + 30:
                break
            values_scaled = c * values
            result = _evaluate_candidate(domain, values_scaled, max_x)
            if result is not None:
                dom, vals, a = result
                if a > best_a:
                    best_a = a
                    best_result = (dom, vals)
        
        if time.time() - start_time > deadline - safety_margin + 60:
            break

    # Also try the constant-f approach as a fallback/complement
    if time.time() - start_time < deadline - 60:
        for K in [200000, 100000, 50000]:
            if time.time() - start_time > deadline - 30:
                break
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            H_K = np.sum(1.0 / domain.astype(np.float64))
            # For constant f(k) = -c, M(x) = -c * sum_{k=1}^{min(x,K)} floor(x/k)
            # For x >= K, this is -c * H_K * x approximately... need to be more careful
            # Actually M(x) = -c * sum_{k=1}^{K} floor(x/k)
            # For x = 10K, sum_{k=1}^{K} floor(10K/k) ≈ 10K * H_K
            # So we need -c * 10K * H_K <= 1.0001, i.e., c >= -1.0001/(10K*H_K)
            # But c should be positive for M(x) <= 1 (since f(k) = -c < 0)
            # Wait, if f(k) = -c with c > 0, then M(x) = -c * sum floor(x/k) < 0 < 1.0001
            # So any c > 0 works for the constraint!
            # The objective is a = -sum f(k)*log(k)/k = -sum (-c)*log(k)/k = c * sum log(k)/k
            # So we want c as large as possible, but c <= 10 (clipping)
            # And we need sum f(k)/k = 0 after correction
            # sum f(k)/k = -c * H_K, so S = -c*H_K, f(1) = -c - S = -c + c*H_K = c*(H_K - 1)
            # f(1) must be in [-10, 10] after clipping, so c*(H_K-1) <= 10
            
            for c_frac in [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]:
                if time.time() - start_time > deadline - 15:
                    break
                c = 10.0 * c_frac
                values = np.full(K, -c, dtype=np.float64)
                result = _evaluate_candidate(domain, values, 10 * K)
                if result is not None:
                    dom, vals, a = result
                    if a > best_a:
                        best_a = a
                        best_result = (dom, vals)

    if best_result is None:
        # Fallback to a known feasible solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
