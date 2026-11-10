# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function mu(k) which satisfies the identity sum_{d|n} mu(d) = 0 for n>1.
    This ensures M(x) = sum_k f(k) floor(x/k) stays close to 1.
    """
    start_time = time.time()
    DEADLINE = 625.0  # safety margin before 630s

    def compute_mobius(K):
        """Compute Mobius function mu[1..K] using a linear sieve."""
        mu = np.ones(K + 1, dtype=np.float64)
        is_composite = np.zeros(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility_vectorized(domain, values, K):
        """
        Check M(x) <= 1.0001 for all x in [1, 10K].
        M(x) = sum_k f(k) floor(x/k) = sum_{n=1}^x h(n) where h(n) = sum_{d|n} f(d).
        Uses vectorized slice addition for h.
        """
        max_x = 10 * K
        h = np.zeros(max_x + 1, dtype=np.float64)
        
        # h[n] = sum_{d|n} f(d)
        # For each d in domain, add f(d) to h[d], h[2d], h[3d], ...
        for k, f in zip(domain, values):
            if f != 0.0:
                h[k::k] += f
        
        # M(x) is cumulative sum of h[1..x]
        # h[0] is unused (n starts at 1)
        M = np.cumsum(h[1:max_x + 1])
        
        return np.max(M)

    def compute_objective(domain, values):
        """Compute a = -sum f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    # We aim for the largest K to maximize the approximation of the infinite sum.
    # K=200,000 is the maximum allowed.
    K = 200000
    
    if time.time() > DEADLINE - 60:
        # Fallback if we are already too late (unlikely)
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    mu = compute_mobius(K)
    
    # Build domain and values from non-zero mu
    nonzero_mask = mu[1:K + 1] != 0.0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:K + 1][nonzero_mask].copy()
    
    # Ensure 1 is in domain (mu[1] = 1, so it should be)
    # The mask includes index 0 (k=1) if mu[1] != 0. mu[1]=1, so yes.
    
    # Calculate S = sum f(k)/k
    S = np.sum(values / domain)
    
    # Adjust f(1) to satisfy sum f'(k)/k = 0
    # f'(1) = f(1) - S
    # Find index of 1 in domain
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] = values[idx_1[0]] - S
    else:
        # If 1 is not in domain (shouldn't happen for mu), add it
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[1.0 - S], values])
    
    # Clip values to [-10, 10]
    values = np.clip(values, -10.0, 10.0)
    
    # Check feasibility
    max_M = check_feasibility_vectorized(domain, values, K)
    
    if max_M <= 1.0001:
        # Feasible. Return this solution.
        # The objective is maximized by having large K and correct mu values.
        return domain, values
    else:
        # If infeasible (unlikely for large K with mu, but possible if numerical errors or specific K behavior),
        # we might need to scale down or use a smaller K.
        # However, for mu, M(x) should be <= 1.
        # Let's try a slightly smaller K if it fails, just in case.
        for K_try in [100000, 50000, 10000]:
            if time.time() > DEADLINE - 60:
                break
            mu = compute_mobius(K_try)
            nonzero_mask = mu[1:K_try + 1] != 0.0
            domain = np.where(nonzero_mask)[0] + 1
            values = mu[1:K_try + 1][nonzero_mask].copy()
            
            S = np.sum(values / domain)
            idx_1 = np.where(domain == 1)[0]
            if len(idx_1) > 0:
                values[idx_1[0]] = values[idx_1[0]] - S
            else:
                domain = np.concatenate([[1], domain])
                values = np.concatenate([[1.0 - S], values])
            
            values = np.clip(values, -10.0, 10.0)
            
            max_M = check_feasibility_vectorized(domain, values, K_try)
            if max_M <= 1.0001:
                return domain, values
        
        # Fallback: use the original seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
