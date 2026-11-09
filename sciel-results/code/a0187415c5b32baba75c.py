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

def check_feasibility_fast(f, K, max_x):
    """Check if M(x) <= 1.0001 for all x in [1, max_x] using difference array."""
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, K + 1):
        fk = f[k]
        if fk == 0:
            continue
        j_max = max_x // k
        if j_max == 0:
            continue
        for j in range(1, j_max + 1):
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
    M = np.cumsum(diff)
    return np.max(M[1:max_x + 1]) <= 1.0001

def solve_lp(K, time_limit):
    """Solve LP for given K. Returns (domain, values, score) or None."""
    n_vars = K - 1  # f(2)..f(K)
    if n_vars <= 0:
        return None
    
    ks = np.arange(2, K + 1, dtype=np.float64)
    c = np.log(ks) / ks  # minimize sum c_k * g_k
    
    max_x = 10 * K
    
    # Build constraint matrix: M(x) <= 1 for all x in [1, max_x]
    # M(x) = f(1)*x + sum_{k=2}^{K} f(k)*floor(x/k)
    # With sum f(k)/k = 0 => f(1) = -sum_{k=2}^{K} f(k)/k
    # M(x) = -x*sum_{k=2}^{K} f(k)/k + sum_{k=2}^{K} f(k)*floor(x/k)
    #       = sum_{k=2}^{K} f(k)*(floor(x/k) - x/k)
    #       = -sum_{k=2}^{K} f(k)*{x/k}
    # So constraint: -sum_{k=2}^{K} f(k)*{x/k} <= 1
    
    A_rows = []
    for x in range(1, max_x + 1):
        row = np.empty(n_vars, dtype=np.float64)
        for i, k in enumerate(ks):
            row[i] = -((x % k) / k)
        A_rows.append(row)
    
    A_ub = np.array(A_rows)
    b_ub = np.ones(max_x, dtype=np.float64)
    
    bounds = [(-10, 10)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'time_limit': time_limit})
        if res.success:
            g = res.x
            f = np.zeros(K + 1, dtype=np.float64)
            f[2:K + 1] = g
            S = np.sum(f[2:K + 1] / ks)
            f[1] = -S
            
            # Verify feasibility
            if not check_feasibility_fast(f, K, max_x):
                return None
            
            # Compute score
            a = -np.sum(f[2:K + 1] * c)
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f[1:K + 1]
            values = np.clip(values, -10, 10)
            
            return domain, values, a
    except Exception:
        pass
    
    return None

def mobius_solution(K, time_limit):
    """
    Try the Möbius function approach: f(k) = mu(k) for square-free k.
    This leverages the identity sum mu(k)/k = 0 (approximately for finite K)
    and the fact that M(x) = sum mu(k)*floor(x/k) = 1 for x >= 1 (in the limit).
    """
    elapsed = time.time()
    if time_limit is not None and elapsed > time_limit:
        return None
    
    mu = compute_mobius(K)
    
    # Check which k have mu(k) != 0 (square-free)
    mask = mu[1:K + 1] != 0
    n_sqfree = np.sum(mask)
    
    if n_sqfree < 2:
        return None
    
    # Build f: f(k) = mu(k) for square-free k, 0 otherwise
    f = np.zeros(K + 1, dtype=np.float64)
    f[1:K + 1] = mu[1:K + 1].astype(np.float64)
    
    # Compute S = sum f(k)/k
    ks = np.arange(1, K + 1, dtype=np.float64)
    S = np.sum(f[1:K + 1] / ks)
    
    # The evaluator will set f(1) = f(1) - S, so effectively f(1) becomes -S
    # But we need to check feasibility with the adjusted f(1)
    f_adjusted = f.copy()
    f_adjusted[1] = f[1] - S
    
    # Check feasibility
    max_x = 10 * K
    if not check_feasibility_fast(f_adjusted, K, max_x):
        return None
    
    # Compute score: a = -sum f(k)*log(k)/k
    log_ks = np.log(ks)
    a = -np.sum(f_adjusted[1:K + 1] * log_ks[1:K + 1] / ks[1:K + 1])
    
    # Build domain and values
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = f_adjusted[1:K + 1]
    values = np.clip(values, -10, 10)
    
    return domain, values, a

def hybrid_solution(K, time_limit):
    """
    Hybrid approach: start from Möbius, then use LP to refine.
    Use Möbius as initial guess and warm start, or use it to guide the LP.
    """
    elapsed = time.time()
    if time_limit is not None and elapsed > time_limit:
        return None
    
    # First try pure Möbius
    result = mobius_solution(K, time_limit)
    if result is not None:
        return result
    
    # If Möbius doesn't work directly, try LP with Möbius-inspired bounds
    return solve_lp(K, time_limit)

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Strategy 1: Try Möbius-based solutions for various K
    # The Möbius function naturally satisfies sum mu(k)/k ~ 0 and M(x) ~ 1
    K_mobius = [5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_mobius:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 30:
            break
        
        time_limit = start_time + min(remaining - 10, 120)
        result = mobius_solution(K, time_limit)
        
        if result is not None:
            domain, values, a = result
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
    
    # Strategy 2: Try LP for smaller K values where it's tractable
    K_lp = [100, 200, 500, 1000, 2000]
    
    for K in K_lp:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 60:
            break
        
        lp_time = min(remaining - 30, 180)
        if lp_time < 10:
            break
        
        result = solve_lp(K, start_time + lp_time)
        
        if result is not None:
            domain, values, a = result
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
    
    # Fallback: simple known solution
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
