# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # Try increasing K to approach the theoretical optimum a -> 1
    # Start with a reasonable K and increase if time permits
    # The cost is O(K log K) for M(x) computation
    
    best_domain = None
    best_values = None
    best_a = -np.inf
    
    # We will test a few K values
    # K=30 is the seed.
    # Let's try K=100, 500, 1000, 5000, 10000
    
    K_values = [100, 500, 1000, 5000, 10000]
    
    for K in K_values:
        if time.time() > deadline - 10.0:
            break
            
        # Compute Mobius function up to K
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        primes = []
        
        # Sieve of Eratosthenes to get primes and mu
        # Standard linear sieve for mu
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
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
                    
        # Filter domain where mu != 0
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
        
        if len(domain) == 0:
            continue
            
        # Evaluate feasibility and objective
        # 1. Clip values (not needed for mu, but good practice)
        values_clipped = np.clip(values, -10.0, 10.0)
        
        # 2. Compute S = sum f(k)/k
        S = np.sum(values_clipped / domain)
        
        # 3. Adjust f(1)
        # Find index of 1 in domain
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            f1_idx = idx_1[0]
            f1_original = values_clipped[f1_idx]
            f1_new = f1_original - S
            values_clipped[f1_idx] = f1_new
        else:
            # If 1 is not in domain, f(1) is 0 initially, so f(1) becomes -S
            # But domain must contain distinct positive integers.
            # The problem says "f(1) = 0 before this if 1 is not in the domain".
            # This implies we might need to add 1 to the domain?
            # "domain holds distinct positive integers".
            # If 1 is not in domain, the sum is over existing k.
            # The adjustment is "f(1) is replaced by f(1) - S".
            # If 1 is not in domain, this step is ambiguous or implies adding it?
            # "so that afterwards sum_k f(k)/k = 0".
            # If 1 is not in domain, we cannot change f(1) to satisfy the sum unless we add it.
            # The seed includes 1. Let's assume we should include 1.
            # In our construction, 1 is always in domain (mu(1)=1).
            pass

        # 4. Compute M(x) for x in [1, 10K]
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        # Efficient computation of M(x) = sum f(k) floor(x/k)
        # Iterate k, add f(k) to M(x) for x = k, 2k, ...
        for k, f_k in zip(domain, values_clipped):
            if f_k == 0:
                continue
            # Add f_k to M[x] for x in k, 2k, ..., max_x
            # M[k::k] += f_k
            M[k:max_x+1:k] += f_k
            
        # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
        # Note: M(0) is 0, we care about x >= 1
        if np.max(M[1:max_x+1]) > 1.0001:
            # Infeasible
            continue
            
        # 5. Compute objective a = - sum f(k) log(k) / k
        a = -np.sum(values_clipped * np.log(domain) / domain)
        
        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values_clipped.copy()

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values

# EVOLVE-BLOCK-END
