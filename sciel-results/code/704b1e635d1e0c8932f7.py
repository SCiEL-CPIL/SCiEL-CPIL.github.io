# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
    1. Use the Mobius function mu(k) as the base solution, which yields M(x) = 1 in the limit.
    2. For a finite K, M_K(x) = 1 - T_K(x), where T_K(x) is the tail sum.
    3. The constraint is M_K(x) <= 1.0001, which implies T_K(x) >= -0.0001.
    4. We search for the largest K such that min_{x in [1, 10K]} T_K(x) >= -0.0001.
    5. This maximizes the objective a = -sum mu(k) log(k)/k, which converges to 1.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0

    def elapsed_ok():
        return (time.time() - start_time) < (deadline - safety_margin)

    # Compute Mobius function up to a large limit
    # We need mu up to max_x = 10*K. If K is ~20000, max_x is 200000.
    # Let's compute mu up to 200,000 to be safe for K up to 20,000.
    # If we can push K higher, we need larger mu.
    # Max K allowed by problem is 200,000. Then max_x = 2,000,000.
    # Computing mu up to 2,000,000 is fast (< 1s).
    
    MAX_K_GLOBAL = 200000
    MAX_X_GLOBAL = 10 * MAX_K_GLOBAL
    
    # Linear sieve for Mobius
    def compute_mobius(n):
        mu = np.ones(n + 1, dtype=np.int64)
        mu[0] = 0
        primes = []
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
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

    # We will compute mu up to MAX_X_GLOBAL to allow checking tails for large K
    # But storing mu up to 2M is fine.
    mu_global = compute_mobius(MAX_X_GLOBAL)
    
    # Precompute prefix sums of mu (Mertens function M(n))
    # M(n) = sum_{i=1}^n mu(i)
    M_prefix = np.cumsum(mu_global)
    
    # Helper to compute T_K(x) = sum_{k=K+1}^x mu(k) * floor(x/k)
    # Using the formula: T_K(x) = sum_{j=1}^{floor(x/K)} j * (M(floor(x/j)) - M(floor(x/(j+1))))
    # Note: The sum for j goes up to floor(x/K) because for k > x, floor(x/k) = 0.
    # Also, for k <= K, the term is excluded.
    # The decomposition:
    # floor(x/k) = j for k in (x/(j+1), x/j].
    # We sum over j. The range of k is [K+1, x].
    # So j ranges from 1 to floor(x/K).
    # For a given j, k is in [max(K+1, floor(x/(j+1))+1), floor(x/j)].
    # Let L_j = max(K+1, floor(x/(j+1))+1)
    # Let R_j = floor(x/j)
    # If L_j <= R_j, contribution is j * (M(R_j) - M(L_j-1)).
    
    def calc_T_K(x, K):
        if x <= K:
            return 0.0
        max_j = x // K
        total = 0.0
        # Iterate j from 1 to max_j
        # For large x, max_j can be up to 10 (since x <= 10K).
        # So this is very fast.
        for j in range(1, max_j + 1):
            R_j = x // j
            L_j_raw = x // (j + 1) + 1
            L_j = max(K + 1, L_j_raw)
            if L_j <= R_j:
                # M(R_j) - M(L_j-1)
                # M_prefix is 0-indexed, M(n) is sum up to n.
                # M_prefix[n] = sum_{i=1}^n mu(i)
                total += j * (M_prefix[R_j] - M_prefix[L_j - 1])
        return total

    # We want to find the largest K such that min_{x in [1, 10K]} T_K(x) >= -0.0001.
    # T_K(x) is likely minimized when x is a multiple of something or at specific points.
    # However, checking all x in [1, 10K] is O(K) per K, and we might test many K.
    # Let's sample x or check critical points.
    # Critical points for M(x) deviations are often near primes or powers.
    # But T_K(x) is smoother.
    # Let's just check x = 1, 2, ..., 10K for small K, and sample for large K.
    # Actually, since max_j <= 10, calc_T_K is O(10).
    # Checking 10K values takes O(100K) operations.
    # If K=20,000, that's 2,000,000 ops. Very fast in Python if vectorized or looped?
    # Python loop of 2M iterations is ~0.1s. Acceptable.
    
    best_score = -np.inf
    best_K = 0
    best_domain = None
    best_values = None
    
    # We can binary search for the max K? 
    # Is the feasibility monotonic? 
    # If K is feasible, is K-1 feasible?
    # T_{K-1}(x) = T_K(x) + mu(K) * floor(x/K).
    # This doesn't guarantee monotonicity of the minimum.
    # However, usually larger K is better for the objective.
    # Let's just iterate K from large to small until we find a feasible one.
    # Or iterate from small to large and keep the best?
    # The objective a(K) increases with K (converges to 1).
    # So we want the MAX feasible K.
    
    # Let's check feasibility for K candidates.
    # Start from high K and go down? Or just check a few high Ks.
    # The constraint is tight. 
    # Let's try K = 20000, 10000, 5000, ...
    
    # To be safe, let's check a range of Ks.
    # We'll check K from 200000 down to 1? No, too slow if many fail.
    # Let's check K = 200000, 100000, 50000, 20000, 10000, 5000, 1000, 100, 10, 1.
    # If a large K fails, a smaller one might pass.
    
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 10, 1]
    
    # We need to verify M(x) <= 1.0001 for ALL x in [1, 10K].
    # This is expensive if we do it naively for every candidate.
    # But calc_T_K is fast.
    
    for K in K_candidates:
        if not elapsed_ok():
            break
            
        if K > MAX_K_GLOBAL:
            continue
            
        # Check feasibility
        feasible = True
        # We must check all x in [1, 10K].
        # For K=200,000, 10K=2,000,000.
        # Looping 2M times in Python is ~0.2-0.5s.
        # Doing this for 12 candidates is ~5s. Acceptable.
        
        # Optimization: Only check x where T_K(x) might be minimal.
        # But we don't know where that is.
        # Let's just check all x for correctness.
        
        # Vectorized approach for checking min T_K(x)?
        # T_K(x) depends on x non-trivially.
        # Let's just loop.
        
        min_T = np.inf
        # We can skip x < K+1 because T_K(x) = 0 for x <= K.
        # So x from K+1 to 10K.
        
        # To speed up, we can sample if K is very large?
        # No, the constraint is strict. We must check all.
        
        # However, we can stop early if we find a violation.
        
        for x in range(K + 1, 10 * K + 1):
            T_val = calc_T_K(x, K)
            if T_val < -0.0001:
                feasible = False
                break
            # Track min for potential debugging, not strictly needed
            if T_val < min_T:
                min_T = T_val
                
        if feasible:
            # Compute objective for this K
            # a = -sum_{k=1}^K mu(k) log(k) / k
            domain = np.arange(1, K + 1, dtype=np.int64)
            f_values = mu_global[1:K+1].astype(np.float64)
            
            # Clip values to [-10, 10] - mu is -1,0,1 so no clipping needed
            # But the evaluator clips.
            
            # Calculate S = sum f(k)/k
            # Note: f(1) is adjusted.
            # The evaluator does:
            # 1. Clip values.
            # 2. S = sum_k f(k)/k (using clipped values).
            # 3. f(1) = f(1) - S.
            # 4. Check sum f(k)/k == 0.
            # 5. Check M(x) <= 1.0001.
            #    Note: M(x) uses the ADJUSTED f(1).
            #    M(x) = sum f(k) floor(x/k).
            #    The adjustment changes f(1) by -S.
            #    Delta M(x) = -S * floor(x/1) = -S * x.
            #    So M_adjusted(x) = M_original(x) - S * x.
            
            # We need M_adjusted(x) <= 1.0001.
            # M_original(x) for mu is 1 - T_K(x).
            # So we need 1 - T_K(x) - S * x <= 1.0001.
            # -T_K(x) - S * x <= 0.0001.
            # T_K(x) + S * x >= -0.0001.
            
            # We previously checked T_K(x) >= -0.0001 assuming S=0.
            # But S is NOT 0 for finite K.
            # S = sum_{k=1}^K mu(k) / k.
            # This sum converges to 0, but for finite K it is non-zero.
            # Let's estimate S.
            
            S = np.sum(f_values / domain)
            
            # Now we need to re-verify feasibility with the S term.
            # Condition: T_K(x) + S * x >= -0.0001 for all x in [1, 10K].
            
            feasible = True
            for x in range(1, 10 * K + 1):
                # T_K(x) is 0 for x <= K
                if x <= K:
                    T_val = 0.0
                else:
                    T_val = calc_T_K(x, K)
                
                val = T_val + S * x
                if val < -0.0001:
                    feasible = False
                    break
            
            if feasible:
                # Calculate final objective
                # The metric is a = -sum f(k) log(k) / k.
                # Which f(k)? The adjusted ones?
                # "The metric is a = -sum_k f(k) * log(k) / k"
                # Usually this refers to the final function returned/evaluated.
                # The evaluator computes the metric AFTER adjustment?
                # "values are clipped... f(1) is replaced... metric is a..."
                # Yes, it uses the adjusted values.
                
                f_adj = f_values.copy()
                f_adj[0] = f_adj[0] - S # f(1) is index 0
                
                a = -np.sum(f_adj * np.log(domain) / domain)
                
                if a > best_score:
                    best_score = a
                    best_K = K
                    best_domain = domain.copy()
                    best_values = f_adj.copy() # Return the adjusted values? 
                    # The function returns (domain, values).
                    # The evaluator applies the adjustment itself?
                    # "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0."
                    # This implies the evaluator does the adjustment.
                    # So we should return the RAW values (before adjustment)?
                    # "returning (domain, values): two 1-D arrays... values holds the finite reals f(k)."
                    # "Evaluation: values are clipped... f(1) is replaced..."
                    # Yes, the evaluator does the adjustment.
                    # So we should return the raw mu values.
                    best_values = mu_global[1:K+1].astype(np.float64).copy()
                    
                    # We can break here if we found a very large K, 
                    # but smaller Ks might have better a? 
                    # a(K) increases with K. So larger K is better.
                    # Since we iterate from high to low, the first feasible one is the best.
                    # Wait, is a(K) monotonically increasing?
                    # a(K) = -sum_{k=1}^K mu(k) log k / k.
                    # The terms are not all positive.
                    # But the limit is 1.
                    # Generally, larger K gives a value closer to 1.
                    # So the largest feasible K is the best.
                    break

    if best_domain is not None:
        return best_domain, best_values

    # Fallback
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
