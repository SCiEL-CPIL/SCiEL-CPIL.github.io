# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
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

def build_lp_matrices_fast(K, x_samples):
    """
    Build the LP matrices efficiently using vectorized operations.
    
    Variables: f(2), f(3), ..., f(K)
    Objective: minimize sum f(k)*log(k)/k (to maximize -sum f(k)*log(k)/k)
    Constraints: M(x) <= 1 for x in x_samples
      where M(x) = sum_{k=2}^{K} f(k) * (floor(x/k) - 1/k)
    """
    n_vars = K - 1  # f(2) through f(K)
    ks = np.arange(2, K + 1, dtype=np.float64)
    
    # Objective: c = log(k)/k
    c = np.log(ks) / ks
    
    # Constraints: A_ub[idx, i] = floor(x//k) - 1/k for k <= x, else -1/k
    n_constraints = len(x_samples)
    A_ub = np.empty((n_constraints, n_vars), dtype=np.float64)
    
    for idx, x in enumerate(x_samples):
        # For k <= x: floor(x/k) - 1/k
        # For k > x: 0 - 1/k = -1/k
        k_max = min(x, K)
        # Initialize with -1/k for all
        A_ub[idx, :] = -1.0 / ks
        # Overwrite for k <= x
        if k_max >= 2:
            n_active = k_max - 1  # number of k values from 2 to k_max
            A_ub[idx, :n_active] = x // ks[:n_active] - 1.0 / ks[:n_active]
    
    b_ub = np.ones(n_constraints)
    
    return c, A_ub, b_ub

def solve_lp_fast(K, x_samples, deadline, mu=None):
    """Solve the LP for given K and x_samples with optional warm start."""
    c, A_ub, b_ub = build_lp_matrices_fast(K, x_samples)
    n_vars = K - 1
    
    # Add bounds to help the solver
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Try to use a warm start if we have a good initial guess
    x0 = None
    if mu is not None:
        # Use scaled Möbius as initial guess
        mu_scaled = mu[2:K + 1].astype(np.float64) * 0.5
        x0 = mu_scaled
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-8})
    except Exception:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return None, None
    
    f_vals = res.x.copy()
    
    # Clip to [-10, 10] as per evaluator
    f_vals = np.clip(f_vals, -10.0, 10.0)
    
    # Compute f(1) = -sum f(k)/k
    ks = np.arange(2, K + 1, dtype=np.float64)
    f1 = -np.sum(f_vals / ks)
    
    # Build full values array
    values_full = np.zeros(K + 1, dtype=np.float64)
    values_full[1] = f1
    values_full[2:K + 1] = f_vals
    
    # Remove near-zero values
    mask = np.abs(values_full[1:]) > 1e-12
    domain_out = np.nonzero(mask)[0] + 1
    values_out = values_full[1:][mask]
    
    if len(domain_out) < 1:
        return None, None
    
    # Compute score
    log_k = np.log(domain_out.astype(np.float64))
    a = -np.sum(values_out * log_k / domain_out.astype(np.float64))
    
    return domain_out, values_out, a

def check_feasibility_fast(domain, values, K, max_x, check_xs):
    """Check M(x) <= 1.0001 for sample x values using vectorized approach."""
    f_full = np.zeros(K + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if k <= K:
            f_full[k] = values[i]
    
    for x in check_xs:
        # M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
        k_max = min(x, K)
        ks = np.arange(1, k_max + 1)
        floors = x // ks
        M_x = np.sum(f_full[1:k_max + 1] * floors)
        if M_x > 1.0001:
            return False
    return True

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the score a = -sum f(k)*log(k)/k subject to M(x) <= 1 for all x.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Precompute Möbius for potential warm start
    K_max_try = 50000
    mu = compute_mobius(K_max_try)
    
    # Try different K values, starting with larger ones
    K_list = [50000, 40000, 30000, 20000, 15000, 10000]
    
    for K in K_list:
        if time.time() > deadline - 90:
            break
        
        max_x = 10 * K
        
        # Use adaptive x sampling: dense near 0, sparse far
        # Critical region is small x where M(x) is most likely to exceed 1
        x_samples = []
        # Dense sampling for small x
        for x in range(1, min(200, max_x + 1)):
            x_samples.append(x)
        # Medium density
        step = max(1, max_x // 500)
        for x in range(200, max_x + 1, step):
            x_samples.append(x)
        # Ensure max_x is included
        if max_x not in x_samples:
            x_samples.append(max_x)
        
        x_samples = sorted(set(x_samples))
        
        domain, values, a = solve_lp_fast(K, x_samples, deadline, mu)
        
        if domain is not None and len(domain) > 0:
            # Verify feasibility on a denser set of x values
            check_xs = list(range(1, min(300, max_x + 1)))
            step = max(1, max_x // 200)
            for x in range(300, max_x + 1, step):
                check_xs.append(x)
            if max_x not in check_xs:
                check_xs.append(max_x)
            
            feasible = check_feasibility_fast(domain, values, K, max_x, check_xs)
            
            if feasible:
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
