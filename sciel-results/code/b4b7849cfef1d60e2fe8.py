# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

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
    deadline = start_time + 600  # 10 minutes
    
    # We will use the Mobius function as a basis.
    # Let h(k) = alpha * mu(k) for k > 1, and h(1) determined by the sum constraint.
    # Actually, the evaluator adjusts f(1).
    # We define f(k) for k=1..K.
    # f(k) = mu(k) for k > 1.
    # f(1) is whatever, it gets adjusted.
    # But the constraint is on the adjusted function.
    # Let's stick to the heuristic:
    # Choose K.
    # Compute mu(k).
    # Compute min_x sum_{k=2}^K mu(k) {x/k}.
    # Set alpha = -1 / min_val.
    # f(k) = alpha * mu(k) for k > 1.
    # f(1) = 0 (will be adjusted).
    
    K = 2000
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if (j // (i * i)) * i * i == j:
                    mu[j] = 0
    
    # mu[1] = 1
    # We use k from 2 to K
    ks = np.arange(2, K + 1)
    mus = mu[2:K + 1]
    
    # We need to find min over x in [1, 10K] of S(x) = sum_{k=2}^K mu(k) (x/k - floor(x/k))
    # This is sum mu(k) * frac(x/k).
    # x is integer.
    # We can sample x or compute exactly.
    # The function is periodic? No.
    # But for large K, the minimum is likely attained at some x.
    # Let's just use a small LP or simple search if K is small.
    # For K=2000, 10K = 20000.
    # We can compute the sum for all x in 1..20000?
    # Sum is over k=2..2000.
    # Vectorized:
    # x_range = np.arange(1, 10 * K + 1)
    # This might be slow.
    
    # Alternative: The constraint is sum_{k} h(k) (floor(x/k) - x/k) <= 1.
    # Let h(k) = c * mu(k).
    # c * sum mu(k) (floor(x/k) - x/k) <= 1.
    # sum mu(k) (floor(x/k) - x/k) = S_K(x) - x * sum mu(k)/k.
    # Let A(x) = S_K(x) - x * L_K, where L_K = sum_{k=2}^K mu(k)/k.
    # We need c * A(x) <= 1 for all x.
    # If A(x) > 0, c <= 1/A(x).
    # If A(x) < 0, c >= 1/A(x).
    # We want to maximize objective a = -c * sum mu(k) log(k)/k.
    # Let B = sum_{k=2}^K mu(k) log(k)/k.
    # a = -c * B.
    # If B is negative (which it is, approx -1), then -B is positive.
    # So we want c to be as large positive as possible?
    # If c is positive, a is positive.
    # Constraint: c <= 1 / max(A(x)) if max(A(x)) > 0.
    # Also c >= 1 / min(A(x)) if min(A(x)) < 0.
    # So c is bounded by [1/min(A), 1/max(A)] (careful with signs).
    # Actually, if min(A) < 0 < max(A), then c must be in [1/max(A), 1/min(A)]?
    # No. c * A(x) <= 1.
    # If A(x) > 0, c <= 1/A(x).
    # If A(x) < 0, c >= 1/A(x).
    # So c <= min_{A>0} (1/A) and c >= max_{A<0} (1/A).
    # We want to maximize c (to maximize a = -c B, since B < 0).
    # So we pick c = min_{A>0} (1/A(x)).
    # We must check if this c satisfies the lower bound.
    
    # Let's compute A(x) for x in 1..10K.
    # K=2000, 10K=20000.
    # A(x) = sum_{k=2}^K mu(k) floor(x/k) - x * L_K.
    # We can compute this efficiently.
    
    L_K = np.sum(mus / ks)
    
    # Compute sum_{k=2}^K mu(k) floor(x/k) for all x in 1..10K
    # This is a convolution-like sum.
    # floor(x/k) changes value at multiples of k.
    # We can use a difference array or just loop if K is small.
    # For K=2000, 10K=20000.
    # Loop x from 1 to 20000.
    # Inner sum over k=2..2000.
    # 20000 * 2000 = 40 million operations. Might be slow in Python.
    # Use NumPy.
    
    # Let's try a smaller K first to be safe, e.g., K=500.
    K = 500
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if (j // (i * i)) * i * i == j:
                    mu[j] = 0
    
    ks = np.arange(2, K + 1)
    mus = mu[2:K + 1]
    L_K = np.sum(mus / ks)
    
    max_x = 10 * K
    # Compute S_K(x) = sum_{k=2}^K mu(k) floor(x/k)
    # We can do this by iterating k and adding mu(k) to a range.
    # floor(x/k) = j for x in [j*k, (j+1)*k - 1].
    # This is still O(K log K) or similar.
    
    # Simpler: Just compute for a subset of x or use the property that the min/max is likely at small x or specific points.
    # Actually, for the Mobius function, A(x) is related to the Mertens function.
    # Let's just compute A(x) for x=1..max_x using a loop in Python with NumPy acceleration for the inner sum?
    # No, vectorize over x.
    # x_vals = np.arange(1, max_x + 1)
    # For each k, floor(x_vals / k) is a vector.
    # Sum mu(k) * floor(x_vals / k).
    # This is K vectors of length max_x.
    # K=500, max_x=5000. 500 * 5000 = 2.5M ops. Very fast.
    
    x_vals = np.arange(1, max_x + 1)
    S_K = np.zeros(max_x, dtype=np.float64)
    for k, m in zip(ks, mus):
        S_K += m * (x_vals // k)
        
    A = S_K - x_vals * L_K
    
    # Find c
    # c * A(x) <= 1
    # If A(x) > 0, c <= 1/A(x)
    # If A(x) < 0, c >= 1/A(x)
    
    pos_A = A[A > 0]
    neg_A = A[A < 0]
    
    if len(pos_A) > 0:
        c_max = np.min(1.0 / pos_A)
    else:
        c_max = np.inf
        
    if len(neg_A) > 0:
        c_min = np.max(1.0 / neg_A)
    else:
        c_min = -np.inf
        
    if c_min > c_max:
        # Infeasible? Or just pick a safe value
        # This shouldn't happen if 0 is in the range of A or if we can scale.
        # If infeasible, return seed.
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        
    # We want to maximize c.
    c = c_max
    
    # Check if c is valid
    if c * np.max(A) > 1.0001:
         # Should not happen
         c = 1.0 / np.max(A) * 0.99
        
    # Construct f
    f_vals = c * mus
    domain = ks
    values = f_vals
    
    # Add 1 to domain?
    # The evaluator handles f(1).
    # We can return domain without 1, or with 1 and value 0.
    # "domain holds distinct positive integers".
    # If 1 is not in domain, f(1) is 0 before adjustment.
    # So we can just return k>1.
    
    return domain, values

# EVOLVE-BLOCK-END
