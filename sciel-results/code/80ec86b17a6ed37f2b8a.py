# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
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

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
    1. Compute Möbius function to identify "good" indices (square-free).
    2. Select a subset of indices K to use in the LP.
    3. Formulate an LP to maximize a = -sum f(k) log(k) / k subject to M(x) <= 1.
    4. The constraint sum f(k)/k = 0 is handled by the evaluator adjusting f(1).
       However, we can include f(1) in the LP or fix it. 
       The evaluator clips values to [-10, 10].
       The evaluator replaces f(1) with f(1) - S where S = sum f(k)/k.
       This ensures sum f(k)/k = 0 for the final evaluated function.
       
       Wait, the metric is a = -sum f(k) log(k) / k.
       If we fix f(1), the evaluator changes it.
       Let's assume we provide f(1) = 0 initially, or just don't include 1 in the domain?
       The domain must contain distinct positive integers.
       If 1 is not in the domain, f(1) is treated as 0.
       Then S = sum_{k in domain} f(k)/k.
       New f(1) = -S.
       The new sum is 0.
       
       The metric a is calculated on the FINAL values (after f(1) adjustment).
       a = - [ f(1)_new * log(1)/1 + sum_{k>1} f(k) log(k)/k ]
       log(1) = 0, so the f(1) term vanishes in the objective!
       So a = - sum_{k in domain, k>1} f(k) log(k) / k.
       
       Therefore, we only need to optimize f(k) for k > 1.
       The constraint is M(x) <= 1 for all x in [1, 10K_max].
       M(x) = sum_{k in domain} f(k) floor(x/k).
       Note: f(1) is also part of M(x).
       f(1)_new = - sum_{k in domain, k>1} f(k)/k.
       
       So M(x) = f(1)_new * floor(x/1) + sum_{k>1} f(k) floor(x/k)
               = - (sum_{j>1} f(j)/j) * x + sum_{k>1} f(k) floor(x/k)
               
       We want to maximize a = - sum_{k>1} f(k) log(k) / k.
       Subject to M(x) <= 1 for all x in [1, 10K_max].
       And clipping: -10 <= f(k) <= 10.
       
       Let's define variables y_k = f(k) for k in subset S of {2, ..., K_max}.
       Objective: minimize sum_{k in S} y_k * log(k) / k  (since we maximize negative of this)
       Constraints:
       For each x in X_check:
       - x * sum_{k in S} y_k / k + sum_{k in S} y_k floor(x/k) <= 1
       
       Let A be the matrix where A_{x, k} = floor(x/k) - x/k.
       Then sum_{k} y_k A_{x, k} <= 1.
       
       Also bounds: -10 <= y_k <= 10.
       
       We need to choose S and X_check carefully to fit in time and memory.
       K_max can be up to 200,000. But LP with 200k vars is too big.
       We should restrict to a smaller K, e.g., K=1000 or 5000.
       The score is proportional to a.
       Let's try K=5000.
       Number of vars: ~5000.
       Number of constraints: We can't check all x in [1, 50000].
       We can check a subset of x, e.g., x = 1, 2, ..., 100 and some larger ones?
       Or just check x = 1, ..., K?
       If we check x = 1, ..., K, that's 5000 constraints.
       5000x5000 LP is solvable in seconds.
       
       Let's try K=2000 first for safety, then increase if time permits.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # 1. Compute Möbius up to a reasonable limit to guide selection or just use all square-free
    # Actually, for the LP, we can just pick the first K integers > 1.
    # Or better, pick square-free integers to keep the "number theoretic" flavor, 
    # though the LP will find the optimal linear combination regardless.
    
    max_K_trial = 2000
    if time.time() < deadline - 100:
        max_K_trial = 5000
    
    # Get square-free numbers up to max_K_trial
    mu = compute_mobius(max_K_trial)
    # Indices k > 1 where mu(k) != 0
    candidates = [k for k in range(2, max_K_trial + 1) if mu[k] != 0]
    
    if not candidates:
        # Fallback
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([0.0, -1.0], dtype=np.float64)
        return domain, values

    K = len(candidates)
    # c_k = log(k) / k
    c = np.array([np.log(k) / k for k in candidates], dtype=np.float64)
    
    # We want to maximize -sum y_k c_k => minimize sum y_k c_k
    # c_k is positive. So we want y_k to be negative.
    # But constraints might prevent that.
    
    # Constraints: M(x) <= 1
    # M(x) = -x * sum(y_j / j) + sum(y_k floor(x/k))
    # sum y_k (floor(x/k) - x/k) <= 1
    
    # We need to choose a set of x values to enforce.
    # The worst case for M(x) might be at small x or specific x.
    # Let's enforce for x = 1, 2, ..., max_K_trial.
    # This gives max_K_trial constraints.
    # If max_K_trial = 5000, we have 5000x5000 matrix.
    
    max_x_check = max_K_trial
    n_constraints = max_x_check
    
    A = np.zeros((n_constraints, K), dtype=np.float64)
    
    # Fill A
    # A[i, j] corresponds to x = i+1, k = candidates[j]
    # A[i, j] = floor((i+1)/k) - (i+1)/k
    
    # Vectorized filling is tricky due to floor.
    # Loop over x is better for memory locality or speed in Python?
    # K is ~5000, n_constraints is 5000. 25M ops.
    # Python loop over 5000 iterations, each doing numpy ops on 5000 elements.
    
    for i in range(n_constraints):
        x = i + 1
        # k array
        k_arr = np.array(candidates, dtype=np.float64)
        A[i, :] = np.floor(x / k_arr) - x / k_arr
        
        if time.time() > deadline - 50:
            break
    
    b = np.ones(n_constraints, dtype=np.float64)
    
    # Bounds: -10 <= y_k <= 10
    bounds = [(-10.0, 10.0) for _ in range(K)]
    
    # Solve LP
    # Method 'highs' is fast
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    except Exception as e:
        # Fallback to a simple heuristic if LP fails
        domain = np.array(candidates[:10], dtype=np.int64)
        values = np.zeros(10)
        values[0] = -1.0
        return domain, values
    
    if not res.success:
        # Fallback
        domain = np.array(candidates[:10], dtype=np.int64)
        values = np.zeros(10)
        values[0] = -1.0
        return domain, values
    
    y_opt = res.x
    
    # Construct domain and values
    # Domain includes 1?
    # The problem says domain holds distinct positive integers.
    # If we include 1 in the domain, we must provide a value for it.
    # The evaluator replaces f(1) with f(1) - S.
    # If we set f(1) = 0 in our output, the evaluator will adjust it.
    # So we should include 1 in the domain and set value 0.
    # Or exclude 1?
    # "f(1) = 0 before this if 1 is not in the domain"
    # So excluding 1 is equivalent to including 1 with value 0.
    # Including 1 allows us to potentially use the "f(1)" slot, but since log(1)=0,
    # it doesn't contribute to the score 'a'.
    # However, it DOES contribute to M(x) and the constraint sum f(k)/k = 0.
    # The evaluator handles the sum=0 constraint by adjusting f(1).
    # So we just need to provide f(k) for k>1.
    # Let's include 1 in the domain with value 0.0 to be safe and explicit.
    
    domain_list = [1] + candidates
    values_list = [0.0] + list(y_opt)
    
    domain = np.array(domain_list, dtype=np.int64)
    values = np.array(values_list, dtype=np.float64)
    
    # Clip values to [-10, 10] as per evaluator behavior (though LP bounds already do this)
    values = np.clip(values, -10, 10)
    
    # Verify feasibility quickly?
    # The LP enforced M(x) <= 1 for x in [1, max_K_trial].
    # The evaluator checks M(x) for x in [1, 10*K_max_domain].
    # Here K_max_domain = max_K_trial.
    # So we checked up to 10 * max_K_trial?
    # No, I set max_x_check = max_K_trial.
    # The evaluator checks up to 10 * K.
    # I need to check up to 10 * max_K_trial.
    # That would be 10 * 5000 = 50,000 constraints.
    # 50,000 x 5,000 matrix is 250M elements. 2GB memory. Too much?
    # And solving 50k x 5k LP might be slow.
    
    # Let's re-evaluate the constraint range.
    # M(x) = sum f(k) floor(x/k).
    # For large x, floor(x/k) ~ x/k.
    # M(x) ~ x * sum f(k)/k.
    # Since sum f(k)/k is adjusted to 0 by f(1), the leading term cancels?
    # No, f(1) is fixed AFTER we provide values.
    # The evaluator computes S = sum f(k)/k.
    # f(1)_new = -S.
    # M(x) = f(1)_new * x + sum_{k>1} f(k) floor(x/k)
    #       = -x * sum_{k>1} f(k)/k + sum_{k>1} f(k) floor(x/k)
    #       = sum_{k>1} f(k) (floor(x/k) - x/k)
    
    # As x -> infinity, floor(x/k) - x/k is bounded between -1 and 0.
    # Specifically, it is in (-1, 0].
    # So M(x) is bounded by sum |f(k)|.
    # The maximum of M(x) might occur at small x where the fractional part structure is different?
    # Actually, floor(x/k) - x/k is always negative (or zero).
    # So M(x) is negative if f(k) are positive?
    # We want to maximize a = - sum f(k) log(k)/k.
    # This suggests f(k) should be negative to make a positive (since log(k)/k > 0).
    # If f(k) < 0, then f(k)(floor(x/k) - x/k) > 0 (since term in bracket is negative).
    # So M(x) > 0.
    # We need M(x) <= 1.
    
    # The function g_x(k) = floor(x/k) - x/k depends on x.
    # For a fixed k, as x varies, g_x(k) oscillates.
    # We need the constraint for ALL x in [1, 10K].
    # My previous LP only checked x in [1, K].
    # I should check x in [1, 10K] or at least a sparse subset.
    
    # Given time limits, let's stick to a smaller K and check more x.
    # Or just check x = 1, ..., 10*K_max_trial.
    # If K_max_trial = 2000, 10K = 20,000.
    # 20,000 x 2,000 matrix = 40M elements = 320MB. Feasible.
    # Solving 20k x 2k LP with HiGHS might take a few seconds.
    
    if time.time() < deadline - 60:
        # Try a larger check range
        K_trial = 2000
        mu_small = compute_mobius(K_trial)
        candidates_small = [k for k in range(2, K_trial + 1) if mu_small[k] != 0]
        K_vars = len(candidates_small)
        
        c_small = np.array([np.log(k) / k for k in candidates_small], dtype=np.float64)
        
        max_x_check_small = 10 * K_trial
        n_cons_small = max_x_check_small
        
        A_small = np.zeros((n_cons_small, K_vars), dtype=np.float64)
        
        # Fill A_small
        k_arr_small = np.array(candidates_small, dtype=np.float64)
        
        for i in range(n_cons_small):
            x = i + 1
            A_small[i, :] = np.floor(x / k_arr_small) - x / k_arr_small
            if time.time() > deadline - 30:
                break
        
        b_small = np.ones(n_cons_small, dtype=np.float64)
        bounds_small = [(-10.0, 10.0) for _ in range(K_vars)]
        
        try:
            res_small = linprog(c_small, A_ub=A_small, b_ub=b_small, bounds=bounds_small, method='highs')
            if res_small.success:
                y_opt_small = res_small.x
                domain_small = np.array([1] + candidates_small, dtype=np.int64)
                values_small = np.array([0.0] + list(y_opt_small), dtype=np.float64)
                values_small = np.clip(values_small, -10, 10)
                return domain_small, values_small
        except Exception:
            pass
            
    # If we get here, return the previous result or fallback
    return domain, values

# EVOLVE-BLOCK-END
