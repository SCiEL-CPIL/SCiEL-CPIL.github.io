# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the metric a = -sum_k f(k) * log(k) / k subject to:
    - M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K]
    - After clipping to [-10,10] and normalization (f(1) -= S where S = sum f(k)/k), sum f(k)/k = 0
    
    Key insight: Use f(k) = -10 for the largest k values (highest log(k)/k ratio per unit 1/k budget),
    saturating at the clipping limit. The constraint M(x) <= 1.0001 is easily satisfied since
    M(x) = f(1) + sum_{k|x, k>1} f(k) <= f(1) = 10*sum_{k in D} 1/k <= 1.0001.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    K = 200000
    
    # We want to maximize sum f(k)*log(k)/k with f(k) = -10 for k in D
    # So we want to maximize sum log(k) for k in D subject to sum 1/k <= 0.10001
    # (since f(1) = 10*sum 1/k must be <= 1.0001)
    
    # Sort k from K down to 2, greedily add k with largest log(k) per unit 1/k = k*log(k)
    # Actually, since f(k) = -10 is fixed, we want to maximize sum log(k) subject to sum 1/k <= budget
    # The optimal is to pick k with largest log(k) per 1/k = k*log(k)
    
    # But k*log(k) is increasing for k >= 2, so we just want the largest k's
    # Wait, that's not right. log(k) is the value, 1/k is the cost.
    # Ratio is log(k) / (1/k) = k*log(k), which is increasing.
    # So greedily pick largest k's first.
    
    budget = 0.10001  # sum 1/k <= this for f(1) = 10*sum 1/k <= 1.0001
    
    # Greedy: start from k=K down to 2, add k if budget allows
    domain_list = []
    remaining_budget = budget
    
    for k in range(K, 1, -1):
        if k * remaining_budget >= 1.0:  # 1/k <= remaining_budget
            domain_list.append(k)
            remaining_budget -= 1.0 / k
        else:
            # Can't fit this k, but maybe smaller ones? Since k*log(k) is increasing,
            # and 1/k is decreasing, once we can't fit k, we also can't fit any larger k.
            # But we're going downward, so we stop here.
            # Actually no - we're going from large to small. If we can't fit k, we can't fit any larger.
            # But we might be able to fit smaller k's. However, since k*log(k) is increasing,
            # smaller k's have lower value per cost. So it's optimal to stop.
            break
    
    # Add 1 to domain with value 0 (it will be adjusted by normalization)
    domain_list.append(1)
    
    domain = np.array(sorted(domain_list), dtype=np.int64)
    values = np.zeros(len(domain), dtype=np.float64)
    
    # Set f(k) = -10 for k > 1
    for i, k in enumerate(domain):
        if k > 1:
            values[i] = -10.0
    
    return domain, values


# EVOLVE-BLOCK-END
