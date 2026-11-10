# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
    1. Use the Möbius function mu(k) for k up to K.
    2. The identity sum_{k<=x} mu(k) floor(x/k) = 1 for x >= 1 holds exactly for the infinite sum.
       For a finite sum up to K, M(x) = sum_{k<=K} mu(k) floor(x/k).
       For x <= K, this is exactly 1.
       For K < x <= 10K, M(x) = 1 - sum_{k>x} mu(k) floor(x/k) ... wait.
       Actually, sum_{k=1}^infty mu(k) floor(x/k) = 1 for all x >= 1.
       So sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^infty mu(k) floor(x/k).
       For x in [1, 10K], the tail sum is small.
       Specifically, for x > K, floor(x/k) is non-zero only for k <= x.
       So the tail is sum_{k=K+1}^x mu(k) floor(x/k).
       This can be estimated. If it's too large (making M(x) < 1 or > 1), we might need to adjust.
       However, the constraint is M(x) <= 1.0001.
       Since the infinite sum is 1, the finite sum M(x) = 1 - Tail(x).
       So M(x) = 1 - Tail(x).
       If Tail(x) > 0, then M(x) < 1, which satisfies M(x) <= 1.0001.
       If Tail(x) < 0, then M(x) > 1. We need M(x) <= 1.0001.
       So we need Tail(x) >= -0.0001.
       
       The tail T(x) = sum_{k=K+1}^x mu(k) floor(x/k).
       For large K, this tail is small.
       Let's check K=200000.
       The maximum possible violation is if the tail is significantly negative.
       However, the M(x) function for the Mobius function is known to be very close to 1.
       In fact, the Mertens function M(n) = sum_{k<=n} mu(k) is small compared to n.
       The error term in the PNT for the counting function of square-free numbers is related to M(n).
       
       Let's verify feasibility for K=200000.
       We compute M(x) for x in [1, 10K].
       If max M(x) > 1.0001, we are infeasible.
       If feasible, we calculate the score.
       
       The score is a = -sum f(k) log(k) / k.
       For f(k) = mu(k), this is -sum mu(k) log(k) / k.
       It is known that sum_{k=1}^infty mu(k) log(k) / k = -1.
       So the sum approaches -1, and a approaches 1.
       
       We also need to handle the normalization step in the evaluator:
       "f(1) is replaced by f(1) - S with S = sum_k f(k)/k ... so that afterwards sum_k f(k)/k = 0"
       
       Let's compute S for the Mobius function.
       S = sum_{k=1}^K mu(k) / k.
       This sum converges to 0 as K -> infinity.
       For K=200000, S is very close to 0.
       The evaluator subtracts S from f(1).
       f(1) is mu(1) = 1.
       New f(1) = 1 - S.
       The new sum is sum_{k=2}^K mu(k)/k + (1-S)/1 = (sum_{k=1}^K mu(k)/k - 1) + 1 - S = S - 1 + 1 - S = 0.
       This is consistent.
       
       How does this affect M(x)?
       M_new(x) = M_old(x) - S * floor(x/1) = M_old(x) - S * x.
       Since S is small, this is a small perturbation.
       We need to ensure M_new(x) <= 1.0001.
       
       How does this affect the score?
       a_new = - [ (1-S) log(1)/1 + sum_{k=2}^K mu(k) log(k)/k ]
             = - [ 0 + sum_{k=2}^K mu(k) log(k)/k ]
             = - [ sum_{k=1}^K mu(k) log(k)/k - mu(1) log(1)/1 ]
             = - sum_{k=1}^K mu(k) log(k)/k.
       So the score is unchanged by the normalization of f(1) because log(1)=0.
       
       Therefore, using the Mobius function is a very strong candidate.
       We just need to ensure feasibility.
       
       Let's implement:
       1. Compute mu(k) for k=1..K.
       2. Check feasibility of M(x) = sum mu(k) floor(x/k) for x=1..10K.
          Note: The evaluator applies the shift S to f(1) BEFORE checking M(x).
          So we must check M_new(x) = M_old(x) - S*x.
       3. If feasible, return domain=1..K, values=mu(1)..mu(K).
       
       To speed up M(x) computation:
       M(x) = sum_{k=1}^K mu(k) floor(x/k).
       We can precompute prefix sums of mu(k)? No, floor(x/k) is not a simple prefix.
       But we can iterate k and add mu(k) to all multiples of k?
       M(x) = sum_{j=1}^x (sum_{k|j} mu(k)) = sum_{j=1}^x [j=1].
       This identity is exact for infinite sums.
       For finite K, M(x) = sum_{j=1}^x (sum_{k|j, k<=K} mu(k)).
       Let g(j) = sum_{k|j, k<=K} mu(k).
       Then M(x) = sum_{j=1}^x g(j).
       This is much faster to compute if we can compute g(j) efficiently.
       g(j) is the Dirichlet convolution of mu and 1, truncated.
       g(j) = 1 if j=1 and K>=1.
       If j > K, g(j) = sum_{k|j} mu(k) = 0 (since sum_{k|j} mu(k) = 0 for j>1).
       Wait, sum_{k|j} mu(k) is 0 for all j > 1.
       So if K >= j, g(j) = 0 for j > 1.
       If K < j, g(j) = sum_{k|j, k<=K} mu(k).
       
       So for x <= K:
       M(x) = g(1) + sum_{j=2}^x g(j) = 1 + 0 = 1.
       So M(x) = 1 exactly for all x <= K.
       
       For x > K:
       M(x) = 1 + sum_{j=K+1}^x g(j).
       Here g(j) = sum_{k|j, k<=K} mu(k).
       Since sum_{k|j} mu(k) = 0, we have sum_{k|j, k<=K} mu(k) = - sum_{k|j, k>K} mu(k).
       So M(x) = 1 - sum_{j=K+1}^x sum_{k|j, k>K} mu(k).
       Swap sums: sum_{k>K} mu(k) sum_{j=K+1}^x [k|j].
       The inner sum is floor(x/k) - floor(K/k).
       Since k > K, floor(K/k) = 0.
       So M(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
       This matches our earlier derivation.
       
       So we need to compute T(x) = sum_{k=K+1}^x mu(k) floor(x/k) for x in (K, 10K].
       And check if 1 - T(x) <= 1.0001  =>  -T(x) <= 0.0001  =>  T(x) >= -0.0001.
       
       Also, the evaluator shifts f(1) by -S.
       M_new(x) = M_old(x) - S*x.
       For x <= K, M_old(x) = 1. So M_new(x) = 1 - S*x.
       We need 1 - S*x <= 1.0001  =>  -S*x <= 0.0001.
       If S > 0, this is always true.
       If S < 0, we need S*x >= -0.0001.
       Since x can be up to 10K, we need S >= -0.0001 / (10K).
       S = sum mu(k)/k. This sum is positive or negative?
       It oscillates but tends to 0. For K=200000, it's very small.
       
       For x > K, M_new(x) = 1 - T(x) - S*x.
       We need 1 - T(x) - S*x <= 1.0001  =>  -T(x) - S*x <= 0.0001.
       
       Let's compute S and check the constraints.
       If infeasible, we can try to reduce K or adjust f(1) manually?
       No, the evaluator does the adjustment. We just provide f.
       If infeasible, we might need to choose a smaller K where the tail is better behaved, or add a small constant to f(1) to compensate?
       But f(1) is adjusted by the evaluator.
       Actually, we can choose f(1) arbitrarily. The evaluator changes it.
       Let's stick with mu(k) for k>1 and see if mu(1)=1 works.
       
       If it's infeasible, we can try to set f(1) such that the resulting M(x) is feasible.
       But the evaluator forces sum f(k)/k = 0.
       So f(1) is determined by the other values.
       f(1) = - sum_{k=2}^K f(k)/k.
       If we set f(k) = mu(k) for k>1, then f(1) is fixed.
       This is exactly S = sum mu(k)/k, so f(1) = - (S - mu(1)/1) = 1 - S.
       Which is what the evaluator does if we provide mu(1)=1.
       
       So the plan is solid. Just check feasibility.
    """
    start_time = time.time()
    deadline = start_time + 570
    
    # Choose K
    # Larger K gives better score (closer to 1.0)
    # But checking feasibility for 10K points takes time.
    # M(x) for x in [1, 10K] can be computed in O(K log K) or similar.
    # Let's try K = 50000 first to be safe on time, then increase if fast.
    # Actually, K=200000 is allowed.
    # Let's estimate time for K=200000.
    # Computing mu: O(K log log K) ~ fast.
    # Computing S: O(K).
    # Checking feasibility:
    # We need to check x in [1, 10K].
    # For x <= K, M_new(x) = 1 - S*x. Max at x=K if S<0.
    # For x > K, M_new(x) = 1 - T(x) - S*x.
    # T(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # This sum has at most x-K terms.
    # Summing over all x from K+1 to 10K is too slow (O(K^2)).
    # We can optimize T(x) calculation.
    # T(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # Note that floor(x/k) is constant for ranges of k.
    # But k is restricted to > K.
    # For a fixed x, k goes from K+1 to x.
    # We can iterate x from K+1 to 10K.
    # For each x, we need sum_{k=K+1}^x mu(k) floor(x/k).
    # This is still O(K^2) in worst case.
    
    # Alternative:
    # M(x) = sum_{j=1}^x g(j).
    # g(j) = sum_{k|j, k<=K} mu(k).
    # We can compute g(j) for j=1..10K using a sieve-like approach.
    # Initialize g = 0.
    # For each k=1..K:
    #   For each multiple j of k up to 10K:
    #     g[j] += mu(k)
    # This is O(10K * (1 + 1/2 + ... + 1/K)) = O(K log K).
    # For K=200000, 10K=2,000,000.
    # Sum 1/k ~ log K ~ 12.
    # Operations ~ 24 million. This is very fast in Python with NumPy?
    # Maybe not with pure Python loops.
    # But we can use NumPy to speed up the multiples update?
    # Or just use a simple loop if optimized.
    # Actually, 24 million iterations in Python might take ~1-2 seconds.
    # Then prefix sum of g to get M(x) is O(10K).
    # Then check max M(x) - S*x <= 1.0001.
    
    # Let's implement this.
    
    K = 200000
    max_x = 10 * K
    
    # 1. Compute mu(k) for k=1..K
    mu = np.ones(K+1, dtype=np.int64)
    is_prime = np.ones(K+1, dtype=bool)
    primes = []
    for i in range(2, K+1):
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
    # mu[1] = 1 is already set.
    
    # 2. Compute S = sum mu(k)/k
    # S = np.sum(mu[1:] / np.arange(1, K+1))
    # This might have floating point issues but should be fine.
    S = np.sum(mu[1:K+1].astype(np.float64) / np.arange(1, K+1))
    
    # 3. Compute g(j) for j=1..max_x
    g = np.zeros(max_x + 1, dtype=np.float64)
    # g[j] = sum_{k|j, k<=K} mu(k)
    # We can do this by iterating k and adding mu[k] to g[multiples of k]
    # To speed up, we can use slicing.
    for k in range(1, K+1):
        if mu[k] != 0:
            # Add mu[k] to g[k], g[2k], g[3k], ... up to max_x
            g[k::k] += mu[k]
    
    # 4. Compute M_old(x) = prefix sum of g
    M_old = np.cumsum(g)
    
    # 5. Compute M_new(x) = M_old(x) - S * x
    x_vals = np.arange(1, max_x + 1)
    M_new = M_old - S * x_vals
    
    # 6. Check feasibility
    max_M_new = np.max(M_new)
    if max_M_new > 1.0001:
        # Infeasible. Try smaller K?
        # For now, return a known feasible solution (seed)
        # Or try K=100000?
        # Let's just return the seed solution to be safe.
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    # 7. If feasible, return mu(1..K)
    # domain = 1..K
    # values = mu(1..K)
    domain = np.arange(1, K+1, dtype=np.int64)
    values = mu[1:K+1].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
