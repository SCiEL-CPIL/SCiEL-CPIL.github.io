# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin
    
    # Seed values
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_score = -np.inf
    best_domain = seed_domain
    best_values = seed_values
    
    # Compute Mobius function up to 200
    K_max = 200
    mu = np.ones(K_max + 1, dtype=np.int64)
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K_max:
                for j in range(i2, K_max + 1, i2):
                    mu[j] = 0
    
    # Precompute log(k)/k
    log_k_over_k = np.zeros(K_max + 1, dtype=np.float64)
    for k in range(2, K_max + 1):
        log_k_over_k[k] = np.log(k) / k
    
    # Helper to evaluate a candidate
    def evaluate(domain, values):
        if len(domain) == 0:
            return -np.inf
        # Check constraints
        # 1. Distinct positive integers
        if len(np.unique(domain)) != len(domain):
            return -np.inf
        if np.any(domain <= 0):
            return -np.inf
        if np.max(domain) > 200000:
            return -np.inf
        if not np.all(np.isfinite(domain)):
            return -np.inf
        if not np.all(np.isfinite(values)):
            return -np.inf
            
        # 2. Clip values
        vals_clipped = np.clip(values, -10, 10)
        
        # 3. Normalize f(1)
        # S = sum f(k)/k
        S = np.sum(vals_clipped / domain)
        if 1 in domain:
            idx_1 = np.where(domain == 1)[0][0]
            vals_clipped[idx_1] = vals_clipped[idx_1] - S
        else:
            # If 1 is not in domain, f(1) is effectively 0 before adjustment?
            # "f(1) = 0 before this if 1 is not in the domain"
            # So we add 1 to domain with value -S?
            # The evaluator computes M(x) = sum f(k) floor(x/k).
            # If 1 is not in domain, floor(x/1) term is missing.
            # But the rule says "f(1) is replaced by f(1) - S ... (f(1)=0 before this if 1 not in domain)"
            # This implies we should treat it as if 1 is in the domain with value 0, then adjust.
            # So effectively, we add 1 to the domain with value -S.
            domain = np.append(domain, 1)
            vals_clipped = np.append(vals_clipped, -S)
            # Re-sort for consistency (not strictly needed for math, but good for M(x) check)
            sort_idx = np.argsort(domain)
            domain = domain[sort_idx]
            vals_clipped = vals_clipped[sort_idx]

        # 4. Check M(x) <= 1.0001
        K = np.max(domain)
        # M(x) for x in [1, 10K]
        # Efficient check: M(x) is piecewise constant, changes at multiples of k.
        # But 10K can be large.
        # We can check at critical points or sample.
        # For small K, we can check all x.
        if 10 * K <= 100000:
            x_vals = np.arange(1, 10 * K + 1)
            # M(x) = sum vals_clipped[i] * floor(x / domain[i])
            # Vectorized: outer product
            # This might be slow if K is large, but K<=200 here.
            # m * 10K operations. 200 * 2000 = 400k. Fast.
            M_vals = np.zeros(len(x_vals))
            for i in range(len(domain)):
                M_vals += vals_clipped[i] * np.floor(x_vals / domain[i])
            if np.max(M_vals) > 1.0001:
                return -np.inf
        else:
            # If K is large, we might skip detailed check or use approximation
            # For now, assume small K
            pass

        # 5. Calculate a
        # a = -sum f(k) log(k) / k
        # f(1) term is 0 since log(1)=0
        a = -np.sum(vals_clipped * np.log(domain) / domain)
        return a

    # Evaluate seed
    score_seed = evaluate(seed_domain, seed_values)
    if score_seed > best_score:
        best_score = score_seed
        best_domain = seed_domain
        best_values = seed_values

    # Local Search: Try adding/removing elements from the seed domain
    # Or try random subsets of primes/small numbers
    
    # Candidate pool
    candidates = list(range(1, 50))
    
    # Simple random search for a better subset
    num_iterations = 5000
    for _ in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random subset size
        m = np.random.randint(3, 15)
        # Pick random domain from candidates
        d = np.random.choice(candidates, size=m, replace=False)
        d = np.sort(d)
        
        # Assign values: try mu(d)
        v = mu[d].astype(np.float64)
        
        # Try scaling
        scale = np.random.uniform(0.5, 2.0)
        v = v * scale
        
        score = evaluate(d, v)
        if score > best_score:
            best_score = score
            best_domain = d
            best_values = v

    return best_domain, best_values

# EVOLVE-BLOCK-END
