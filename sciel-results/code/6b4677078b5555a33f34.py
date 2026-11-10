# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mu_and_omega(n):
    """Compute Mobius function and Omega (number of prime factors with multiplicity) for 1..n."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    omega = np.zeros(n + 1, dtype=np.int32)  # total number of prime factors with multiplicity
    
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            omega[i] = 1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            omega[i * p] = omega[i] + 1
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # mu[1] should be 1
    mu[1] = 1
    omega[1] = 0
    
    return mu, omega


def _compute_M_vectorized(domain, values, x_max):
    """Compute M(x) = sum_k f(k) * floor(x/k) for x = 1..x_max."""
    M = np.zeros(x_max, dtype=np.float64)
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    
    for k, f in zip(domain, values):
        if f != 0.0:
            M += f * (x_vals // k)
    
    return M


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].
    
    Uses Mobius function as base (M(x)=1 exactly), then adds a perturbation
    in the direction that decreases a while maintaining feasibility.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit
    
    # Start with Mobius function up to K
    K = 500
    mu, omega = _compute_mu_and_omega(K)
    
    # Include all k where mu(k) != 0 (square-free numbers)
    mask = mu != 0
    domain = np.where(mask)[0].astype(np.int64)
    values = mu[domain].astype(np.float64)
    
    # Ensure 1 is in domain
    if len(domain) == 0 or domain[0] != 1:
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[1.0], values])
    
    K_max = domain[-1]
    x_max_eval = 10 * K_max  # Full evaluation range
    
    # Verify base feasibility: M(x) should be 1 for all x >= 1 with Mobius
    # But let's check numerically for safety
    if x_max_eval <= 20000:
        M_base = _compute_M_vectorized(domain, values, x_max_eval)
        if np.max(M_base) > 1.0001:
            # Fallback: use a small known-feasible set
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Now add perturbation: f(k) = mu(k) + c * g(k)
    # where g(k) = (-1)^{omega(k)} / k for square-free k, 0 otherwise
    # Constraint: sum_k c*g(k)/k = c * sum_k (-1)^{omega(k)}/k^2 = 0
    # This is automatically 0 since sum_{k=1}^inf (-1)^{omega(k)}/k^2 = 0 (Euler product)
    # For finite sum, it's approximately 0 but not exactly. We need to enforce it.
    
    # g(k) = (-1)^{omega(k)} / k for k in domain
    omega_domain = omega[domain]
    sign_omega = np.where(omega_domain % 2 == 0, 1.0, -1.0)
    g = sign_omega / domain.astype(np.float64)
    
    # Enforce sum_k g(k)/k = 0 by projecting out the component
    # h(k) = g(k) - alpha * (1/k) where alpha = sum_k g(k)/k^2 / sum_k 1/k^2
    inv_k = 1.0 / domain.astype(np.float64)
    alpha = np.sum(g * inv_k) / np.sum(inv_k * inv_k)
    h = g - alpha * inv_k
    
    # Now sum_k h(k)/k = 0 exactly
    # Check: sum_k h(k)/k = sum g(k)/k - alpha * sum 1/k^2 = alpha*sum(1/k^2) - alpha*sum(1/k^2) = 0 ✓
    
    # The perturbation direction: f(k) = mu(k) + c * h(k)
    # a(c) = -sum_k (mu(k) + c*h(k)) * log(k) / k
    #       = a_base - c * sum_k h(k)*log(k)/k
    # To maximize a (make it more positive... wait, we want a to be as large as possible)
    # Actually the score is proportional to a, and we want higher score.
    # a = -sum f(k)*log(k)/k
    # For mu: a_base = -sum mu(k)*log(k)/k
    # We want to increase a, so we need -c * sum h(k)*log(k)/k > 0
    # i.e., c * sum h(k)*log(k)/k < 0
    
    log_k = np.log(domain.astype(np.float64))
    log_k[0] = 0.0  # log(1) = 0
    
    a_base = -np.sum(values * log_k / domain.astype(np.float64))
    direction_sum = np.sum(h * log_k / domain.astype(np.float64))
    
    # We want c * direction_sum < 0 to increase a
    # So c should have opposite sign to direction_sum
    if direction_sum > 0:
        c_sign = -1.0
    elif direction_sum < 0:
        c_sign = 1.0
    else:
        c_sign = 0.0
    
    # M(x) with perturbation: M(x) = 1 + c * sum_k h(k)*floor(x/k)
    # We need 1 + c * sum_k h(k)*floor(x/k) <= 1.0001 for all x
    # i.e., c * sum_k h(k)*floor(x/k) <= 0.0001 for all x
    # and also we should keep M(x) >= -10 (clipping) but that's loose
    
    # Find the maximum of sum_k h(k)*floor(x/k) over x in [1, 10K]
    # and the minimum, to determine the feasible range of c
    
    if time.time() > deadline - 100:
        # Not enough time, return base
        return domain, values
    
    # Compute H(x) = sum_k h(k) * floor(x/k) for x in [1, 10K]
    x_max = x_max_eval
    H = np.zeros(x_max, dtype=np.float64)
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    
    for k, hk in zip(domain, h):
        if hk != 0.0:
            H += hk * (x_vals // k)
    
    H_max = np.max(H)
    H_min = np.min(H)
    
    # Constraint: c * H(x) <= 0.0001 for all x
    # If c > 0: c * H_max <= 0.0001 => c <= 0.0001 / H_max (if H_max > 0)
    # If c < 0: c * H_min <= 0.0001 => c >= 0.0001 / H_min (if H_min < 0)
    
    if c_sign > 0:
        # c > 0, need c * H_max <= 0.0001
        if H_max > 0:
            c_max = 0.0001 / H_max
        else:
            c_max = 1e10  # No constraint from positive side
        c = c_max * 0.999  # Stay safely within bounds
    elif c_sign < 0:
        # c < 0, need c * H_min <= 0.0001
        # c is negative, H_min is likely negative
        # c * H_min: negative * negative = positive
        if H_min < 0:
            c_min = 0.0001 / H_min  # This is negative
            # c >= c_min (more negative is worse)
            # c should be in [c_min, 0)
            c = c_min * 0.999  # c_min is negative, so this makes c less negative (closer to 0)
            # Wait: c_min = 0.0001/H_min, H_min < 0, so c_min < 0
            # We need c >= c_min, so c can be from c_min up to 0
            # To maximize |c| (and thus the improvement), use c close to c_min
            # But we also need M(x) >= -10 after clipping, which is very loose
            c = c_min * 0.999
        else:
            c = 0.0  # No feasible perturbation
    else:
        c = 0.0
    
    # Apply perturbation
    if c != 0.0:
        values = values + c * h
    
    # Verify feasibility
    if x_max_eval <= 20000:
        M_final = _compute_M_vectorized(domain, values, x_max_eval)
        if np.max(M_final) > 1.0001:
            # Revert to base
            values = mu[domain].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
