# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    return mu


def _evaluate_feasibility(domain, values, max_x):
    """Check if M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, max_x]."""
    m = len(domain)
    if m == 0:
        return True
    
    # For efficiency, we can check M(x) at critical points where floor(x/k) changes.
    # However, the task says to compute for every integer x in [1, 10K].
    # Let's use a vectorized approach if possible, or loop if m is small.
    
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # We need max_{x in 1..max_x} M(x) <= 1.0001
    
    # If m is small, we can just loop over x.
    # If m is large, we might need to be smarter.
    # Given K <= 200000, 10K <= 2000000.
    # m is likely small in our construction.
    
    max_m = 0.0
    for x in range(1, max_x + 1):
        m_x = 0.0
        for k, f_k in zip(domain, values):
            m_x += f_k * (x // k)
        if m_x > max_m:
            max_m = m_x
        if max_m > 1.0001:
            return False
    return True


def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    logs = np.log(domain.astype(np.float64))
    return -np.sum(values * logs / domain)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Try increasing K values for the Mobius function
    # The objective approaches 1.0 as K -> infinity.
    # We need to find the largest K such that the evaluation fits in the time limit.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Start with a small K and increase
    K_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_values:
        if time.time() > deadline - 60:  # Leave 60s for final evaluation
            break
            
        mu = _compute_mobius(K)
        
        # Domain is all k where mu(k) != 0
        mask = mu != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[mask].astype(np.float64)
        
        # Check feasibility: M(x) = sum f(k) floor(x/k) <= 1.0001 for x in [1, 10K]
        # For the Mobius function, M(x) = 1 for all x >= 1 (by the identity sum_{d|x} mu(d) = 1 if x=1 else 0? No.)
        # Actually, sum_{d|x} mu(d) = 0 if x > 1, and 1 if x = 1.
        # Wait, the identity is sum_{d|n} mu(d) = 0 for n > 1.
        # M(x) = sum_{k=1}^K mu(k) floor(x/k).
        # This is the Mertens function M(x) approximated? No.
        # sum_{k=1}^x mu(k) floor(x/k) = sum_{n=1}^x sum_{k|n} mu(k) = sum_{n=1}^x [n=1] = 1.
        # So M(x) = 1 for all x >= 1 if we sum up to x.
        # If we sum up to K < x, M(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
        # Since floor(x/k) = 1 for k > x/2, etc.
        # The error term is related to the Mertens function.
        # It is known that M(x) = O(x / exp(c (log x)^{3/5} ...)).
        # For finite K, M(x) for x > K might exceed 1?
        # Let's just evaluate it numerically for safety.
        
        max_x = 10 * K
        if len(domain) > 10000:
            # Too many points for naive O(m * max_x) evaluation
            # We might need a faster evaluation or skip this K
            continue
            
        if _evaluate_feasibility(domain, values, max_x):
            score = _compute_objective(domain, values)
            if score > best_score:
                best_score = score
                best_domain = domain.copy()
                best_values = values.copy()
                
        if time.time() > deadline - 30:
            break
            
    if best_domain is None:
        # Fallback to the original small solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values


# EVOLVE-BLOCK-END
