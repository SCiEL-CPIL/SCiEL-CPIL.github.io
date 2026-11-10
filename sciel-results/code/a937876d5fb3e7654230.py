import numpy as np

def prime_number_theorem_partial_function():
    """
    Construct a partial function f to maximize the objective.
    """
    K = 10000  # Maximum key
    
    # 1. Define negative part
    neg_keys = np.array([2, 3], dtype=np.int64)
    neg_vals = np.array([-10.0, -10.0], dtype=np.float64)
    
    # Calculate required positive sum
    req_sum = np.sum(-neg_vals / neg_keys) # 5.0 + 3.333... = 8.333...
    
    # 2. Define positive part
    # We want to fill req_sum using f(j) <= 10 for j in [4, K]
    # To minimize objective penalty sum(f(j)*log(j)/j), we prefer large j.
    # So we iterate j from K down to 4.
    
    pos_keys = []
    pos_vals = []
    current_sum = 0.0
    
    # Iterate from K down to 4
    # Using a loop might be slow if K is large, but K=10000 is fine.
    # We can vectorize the "fill" logic.
    
    # Approximate sum of 10/j from J_min to K is 10 * log(K/J_min).
    # We need 10 * log(K/J_min) >= req_sum.
    # log(K/J_min) >= req_sum / 10.
    # K/J_min >= exp(req_sum / 10).
    # J_min <= K / exp(req_sum / 10).
    
    exp_factor = np.exp(req_sum / 10.0)
    J_min_approx = int(K / exp_factor)
    
    # Ensure J_min is at least 4
    J_min = max(4, J_min_approx)
    
    # Now we have candidates j in [J_min, K].
    # We will assign f(j)=10 to most of them, and adjust the smallest one (J_min) to match the sum exactly.
    
    # Sum of 10/j for j in [J_min+1, K]
    # We can just compute the sum of 10/j for j in [J_min, K] and see how much we overshoot.
    # Let S_full = sum_{j=J_min}^K 10/j.
    # If S_full > req_sum, we can reduce f(J_min) to (req_sum - sum_{j=J_min+1}^K 10/j).
    # But f(J_min) must be >= 0.
    # Also f(J_min) <= 10.
    
    # Let's just build the arrays.
    # To be efficient, we only include keys where f(j) != 0.
    
    # We'll use a dictionary or just filter later.
    # Let's create arrays of size K+1 to be safe, then filter.
    f = np.zeros(K + 1, dtype=np.float64)
    
    # Set negative values
    f[neg_keys] = neg_vals
    
    # Set positive values
    # Start from K down to J_min
    # We want sum_{j=J_min}^K f(j)/j = req_sum.
    # Let's set f(j) = 10 for j > J_min.
    # And f(J_min) = req_sum - sum_{j=J_min+1}^K 10/j.
    
    if J_min < K:
        # Sum of 10/j for j in [J_min+1, K]
        # We can compute this sum.
        # For K=10000, this is a loop of ~10000, which is fast.
        sum_rest = 0.0
        for j in range(J_min + 1, K + 1):
            sum_rest += 10.0 / j
            f[j] = 10.0
            
        f[J_min] = req_sum - sum_rest
        
        # If f[J_min] < 0, we overshot. We need to remove some keys.
        # This means J_min was too small (too many keys).
        # We should increase J_min.
        # But for now, let's assume it works or we can scale.
        if f[J_min] < 0:
            # This case is handled by scaling down or adjusting.
            # For simplicity, if it's negative, we just set it to 0 and scale the whole thing?
            # No, that breaks the sum constraint.
            # Better: Find the correct J_min.
            # Let's just use a binary search or linear scan to find the right J_min.
            pass # We'll handle this by starting with a larger J_min if needed.
            
    # To be robust, let's just pick a J_min such that the sum is close.
    # Actually, let's just use the largest possible J_min that allows the sum to be reached.
    # i.e., smallest set of keys.
    # We want to minimize the number of keys to keep the array sparse?
    # No, the cost is m * 10K. Smaller m is better.
    # So we want the fewest number of positive keys.
    # This means we want the largest possible f(j), which is 10.
    # And we want to use the largest possible j to minimize the count?
    # No, using large j means 1/j is small, so we need MORE keys to reach the sum.
    # Using small j means 1/j is large, so we need FEWER keys.
    # BUT, small j has larger log(j)/j penalty.
    # Trade-off:
    # 1. Fewer keys -> faster evaluation.
    # 2. Larger j -> lower objective penalty.
    
    # Let's check the penalty.
    # Penalty for key j is 10 * log(j)/j.
    # For j=100, penalty = 10 * 4.6/100 = 0.46.
    # For j=1000, penalty = 10 * 6.9/1000 = 0.069.
    # So using large j is much better for the objective.
    # Even if we need more keys, the total penalty is lower.
    # And m (number of keys) only affects time, not score.
    # So we should use the LARGEST possible j.
    
    # So, we start from K and go down until we reach the required sum.
    
    f = np.zeros(K + 1, dtype=np.float64)
    f[neg_keys] = neg_vals
    
    current_sum = 0.0
    last_key = -1
    for j in range(K, 3, -1):
        if current_sum + 10.0 / j >= req_sum:
            # This is the last key we need
            f[j] = req_sum - current_sum
            last_key = j
            break
        else:
            f[j] = 10.0
            current_sum += 10.0 / j
            last_key = j
            
    if last_key == -1:
        # Should not happen if K is large enough
        pass

    # Extract domain and values
    mask = f != 0
    domain = np.where(mask)[0]
    values = f[mask]
    
    # Verify constraints
    # 1. Sum constraint
    s = np.sum(values / domain)
    # It should be 0.
    
    # 2. M(x) <= 1.0001
    # Compute M(x) for x in [1, 10K]
    # M(x) = sum_k f(k) * floor(x/k)
    # We can vectorize this.
    # x = 1..10K
    # k = domain
    # floor(x/k) is a matrix of size (10K, m).
    # This might be memory intensive if m is large.
    # m is approx 10K / 10 = K? No.
    # Sum 10/j from J to K is 10 log(K/J).
    # We need sum 8.33.
    # 10 log(K/J) = 8.33 => log(K/J) = 0.833 => K/J = 2.3 => J = K/2.3.
    # So we use keys from K/2.3 to K.
    # Number of keys is approx K/2.3.
    # For K=10000, m ~ 4300.
    # Matrix size 100000 x 4300 = 430 million floats.
    # 430M * 8 bytes = 3.4 GB. Too much.
    
    # We need to compute M(x) efficiently.
    # M(x) = sum_{k in domain} f(k) * floor(x/k).
    # We can iterate over k and add f(k) * floor(x/k) to M(x).
    # This is O(m * 10K) operations.
    # 4300 * 100000 = 430 million operations.
    # In Python, this is too slow (600s limit).
    # We need to use NumPy vectorization or reduce K.
    
    # Let's reduce K to 1000.
    # Then 10K = 10000.
    # m ~ 1000/2.3 = 430.
    # 430 * 10000 = 4.3 million operations.
    # This is fast in NumPy.
    
    # Let's restart with K=1000.
    
    K = 1000
    f = np.zeros(K + 1, dtype=np.float64)
    neg_keys = np.array([2, 3], dtype=np.int64)
    neg_vals = np.array([-10.0, -10.0], dtype=np.float64)
    f[neg_keys] = neg_vals
    req_sum = np.sum(-neg_vals / neg_keys)
    
    current_sum = 0.0
    for j in range(K, 3, -1):
        if current_sum + 10.0 / j >= req_sum:
            f[j] = req_sum - current_sum
            break
        else:
            f[j] = 10.0
            current_sum += 10.0 / j
            
    mask = f != 0
    domain = np.where(mask)[0]
    values = f[mask]
    
    # Compute M(x)
    # x from 1 to 10K
    x_max = 10 * K
    x_range = np.arange(1, x_max + 1)
    
    # M(x) = sum_k f(k) * floor(x/k)
    # We can do this by iterating over k (m is small)
    M = np.zeros(x_max, dtype=np.float64)
    
    for k, val in zip(domain, values):
        # floor(x_range / k)
        M += val * (x_range // k)
        
    max_M = np.max(M)
    
    if max_M > 1.0001:
        # Scale down
        scale = 1.0001 / max_M
        values = values * scale
        # Re-check?
        # M will be scaled by same factor, so max_M becomes 1.0001.
        
    return domain, values

