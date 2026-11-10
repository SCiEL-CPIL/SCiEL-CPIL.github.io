# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
        for p in primes:
            if p * i > K:
                break
            if i % p == 0:
                mu[i * p] = 0
                break
    mu[0] = 0
    mu[1] = 1
    return mu


def _check_feasibility(domain, values, K):
    """Check if M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K]."""
    m = len(domain)
    if m == 0:
        return True
    
    # We only need to check x where floor(x/k) changes, which is at multiples of k.
    # But the constraint is for ALL x in [1, 10*K].
    # M(x) is a step function that is constant on intervals [n, n+1).
    # So we need to check M(x) for x = 1, 2, ..., 10*K.
    # However, M(x) = sum_{k: k<=x} f(k) * floor(x/k).
    # For x > K, floor(x/k) is still changing.
    
    # Efficient check: 
    # M(x) = sum_{k=1}^{min(K,x)} f(k) * floor(x/k)
    # We can compute this for all x = 1..10K in O(K log K) or similar.
    # But 10K can be up to 2,000,000.
    # Let's use the divisor sum approach.
    
    max_x = 10 * K
    if max_x > 2000000: # Safety cap, though K<=200000 implies 10K<=2e6
        max_x = 2000000
        
    # We need to check if max_{x in [1, 10K]} M(x) <= 1.0001
    # M(x) = sum_{k | x? No.} 
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    
    # Let's just compute M(x) for x = 1, 2, ..., 10K.
    # To do this efficiently, we can use the fact that floor(x/k) is constant for ranges of x.
    # But checking all 10K points might be slow if done naively for each k.
    # Vectorized approach:
    # Create an array M of length 10K+1 initialized to 0.
    # For each k in domain:
    #   M[1:10K+1] += f(k) * floor((1:10K) / k)
    # This is O(m * 10K). If m is small, this is fine.
    # If m is large (up to K=200000), this is 200000 * 2000000 = 4e11, too slow.
    
    # Better way:
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # We can compute this for all x in O(K log K) using the divisor trick?
    # No, that's for sum_{d|x} f(d).
    # Here it's floor(x/k).
    
    # Let's stick to the constraint: K <= 200000.
    # If we use the full Mobius function, m=K=200000.
    # 10K = 2,000,000.
    # Naive loop: for k in 1..200000: for x in 1..2000000: M[x] += f[k]*floor(x/k).
    # This is 4e11 operations. Too slow in Python/NumPy if not optimized.
    
    # NumPy optimization:
    # M = np.zeros(10*K+1)
    # for k in domain:
    #    M[1:] += values[i] * (np.arange(1, 10*K+1) // k)
    # This is m * 10K operations.
    # If m is small (e.g. 5), it's 5 * 2e6 = 1e7, very fast.
    # If m is large (200000), it's 4e11, too slow.
    
    # So we must keep m small or use a faster feasibility check.
    # The task says "cost about m * 10K". This implies the evaluator does this.
    # We should mimic this or do something faster.
    
    # If we use a sparse domain (small m), we can afford the naive check.
    # The seed program uses m=5.
    # If we want to improve the score, we might need to increase m or optimize values.
    # But increasing m increases the cost.
    
    # Let's assume we keep m small for now and just optimize values.
    
    x_range = np.arange(1, 10 * K + 1, dtype=np.int64)
    M = np.zeros(len(x_range), dtype=np.float64)
    
    for i in range(m):
        k = domain[i]
        f_k = values[i]
        if f_k == 0:
            continue
        M += f_k * (x_range // k)
        
    return np.max(M) <= 1.0001


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
    deadline = start_time + 590  # 10s margin
    
    # Start with the known feasible baseline
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # Try to improve by adding more terms from the Mobius sequence
    # The ideal function is mu(k).
    # Let's compute mu(k) for k up to some limit and try to include more terms.
    # However, we must ensure feasibility.
    
    # The seed is already quite good. Let's try to optimize the existing values
    # or add a few more terms if feasible.
    
    # Let's try adding k=7 (mu(7)=-1), k=11 (mu(11)=-1), etc.
    # But we need to check feasibility.
    
    # Let's try a simple gradient-free optimization on the existing values
    # or just try a few specific perturbations.
    
    # Actually, the score is 1.0000 for the seed. The optimum is 1.0.
    # Is it possible to exceed 1.0?
    # The problem states "optimum 1.0". This might mean the supremum is 1.0.
    # If so, we can't beat 1.0.
    # However, the score is value/seed. If seed is 1.0, and we get 1.0, score is 1.0.
    # If we get slightly less, score < 1.0.
    # If we can get exactly 1.0, score is 1.0.
    # Maybe we can get a numerical advantage? Unlikely.
    
    # Let's try to compute the exact value for the full Mobius function up to K=200000.
    # But the cost is high.
    
    # Let's stick to the seed or try to refine it slightly.
    # Since the seed is already at the theoretical optimum (or very close),
    # and we are constrained by time, let's just return the seed.
    
    # However, the prompt asks for improvements.
    # Let's try to see if we can reduce the error in the approximation.
    # The seed uses a sparse set.
    # Maybe a denser set approximates mu(k) better?
    # But M(x) <= 1 constraint is tight for mu(k).
    # For mu(k), M(x) = 1 for x >= 1 (if infinite sum).
    # For finite sum, M(x) deviates.
    # The constraint is M(x) <= 1.0001.
    
    # Let's try to add k=7 to the domain.
    # New domain: [1, 2, 3, 5, 7, 30]
    # New values: [f1, -1, -1, -1, -1, f30]
    # We need to re-optimize f1 and f30? Or just set mu values?
    # If we set mu values: f1=1, f2=-1, f3=-1, f5=-1, f7=-1, f30=1.
    # Then sum f(k)/k = 1 - 1/2 - 1/3 - 1/5 - 1/7 + 1/30
    # = 1 - 0.5 - 0.333 - 0.2 - 0.142 + 0.033
    # = 1 - 1.175 = -0.175.
    # Evaluator subtracts S = sum f(k)/k from f(1).
    # So f(1) becomes 1 - (-0.175) = 1.175.
    # Then sum f(k)/k = 0.
    
    # Let's check feasibility for this new set.
    # This is getting complex. Given the time limit and that the seed is 1.0,
    # it's safer to return the seed unless we can quickly verify an improvement.
    
    # Let's try one quick improvement: optimize the values of the seed domain.
    # The objective is a = -sum f(k) log(k)/k.
    # Constraints: sum f(k)/k = 0 (after adjustment) and M(x) <= 1.0001.
    # The adjustment makes f(1) dependent on other f(k).
    # Let g(k) = f(k) for k > 1.
    # f(1) = -sum_{k>1} f(k)/k.
    # Objective a = -f(1)*log(1)/1 - sum_{k>1} f(k) log(k)/k
    # Since log(1)=0, a = -sum_{k>1} f(k) log(k)/k.
    # So f(1) doesn't affect the objective directly, only through constraints.
    # Constraint M(x) = f(1)*floor(x/1) + sum_{k>1} f(k) floor(x/k) <= 1.0001.
    # f(1) = -sum_{k>1} f(k)/k.
    # So M(x) = -sum_{k>1} f(k)/k * x + sum_{k>1} f(k) floor(x/k)
    # = sum_{k>1} f(k) (floor(x/k) - x/k).
    # Note that floor(x/k) - x/k = -frac(x/k).
    # So M(x) = -sum_{k>1} f(k) frac(x/k).
    # We need M(x) <= 1.0001 for all x in [1, 10K].
    # Since frac(x/k) is in [0, 1), M(x) is a sum of terms -f(k)*frac(x/k).
    
    # If we choose f(k) = mu(k) for k>1, then for the infinite sum, M(x) = 1.
    # For finite sum, M(x) will be close to 1.
    
    # The seed values are:
    # k=2: -1, k=3: -1, k=5: -1, k=30: 1.
    # These are mu(2), mu(3), mu(5), mu(30).
    # It seems the seed is already using mu(k) for k>1.
    # And f(1) is adjusted.
    
    # So the seed is essentially the truncated Mobius function.
    # Can we do better?
    # Maybe include more primes?
    # Let's try adding k=7.
    # domain = [1, 2, 3, 5, 7, 30]
    # values for k>1: -1, -1, -1, -1, 1.
    # This is still mu(k).
    # Let's check if this is feasible.
    
    domain_new = np.array([1, 2, 3, 5, 7, 30], dtype=np.int64)
    values_new = np.array([0.0, -1.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    # f(1) will be adjusted by evaluator, but we can compute it here for feasibility check.
    # S = sum f(k)/k = -1/2 -1/3 -1/5 -1/7 + 1/30
    # f(1) = -S
    
    S = np.sum(values_new[1:] / domain_new[1:])
    f1 = -S
    values_new[0] = f1
    
    if _check_feasibility(domain_new, values_new, 30):
        domain = domain_new
        values = values_new
    else:
        # Try even more terms? Or stick to seed.
        pass
        
    return domain, values


# EVOLVE-BLOCK-END
