import numpy as np
import time
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    
    # We will use simulated annealing to find a good set A.
    # x is a binary vector of length 129.
    
    def get_cost(x):
        # x is numpy array of 0s and 1s
        # Convolution x * x gives ordered pair counts
        conv = np.convolve(x, x, mode='full')
        # conv[k] is number of pairs (i, j) with i+j=k
        # We care about k in [128, 256]
        
        # Convert ordered to unordered
        # R(k) = conv[k] / 2 if k is odd
        # R(k) = (conv[k] + x[k/2]) / 2 if k is even
        
        # We only need to check range 128 to 256
        # Note: conv length is 2*n - 1 = 257. Indices 0 to 256.
        
        penalty = 0.0
        sq_sum = 0.0
        
        # Vectorized calculation for the range
        # Indices 128 to 256
        targets = np.arange(target_min, target_max + 1)
        conv_vals = conv[targets]
        
        # Determine parity
        even_mask = (targets % 2 == 0)
        
        # For even targets, we need x[target/2]
        # target/2 will be integer
        mid_indices = targets[even_mask] // 2
        mid_vals = x[mid_indices]
        
        # Initialize R
        R = np.zeros(len(targets))
        
        # Odd targets
        odd_mask = ~even_mask
        R[odd_mask] = conv_vals[odd_mask] / 2.0
        
        # Even targets
        # conv_vals[even_mask] includes 2*sum_{a<b} + x[m]^2
        # Unordered sum is sum_{a<b} + x[m]
        # conv_val = 2 * sum_{a<b} + x[m] (since x[m]^2 = x[m])
        # sum_{a<b} = (conv_val - x[m]) / 2
        # R = sum_{a<b} + x[m] = (conv_val - x[m]) / 2 + x[m] = (conv_val + x[m]) / 2
        R[even_mask] = (conv_vals[even_mask] + mid_vals) / 2.0
        
        # Check coverage: R[T] must be >= 1
        # Since R is calculated from integers, it should be integer or .5?
        # Wait.
        # If k is odd, conv[k] is even. R is integer.
        # If k is even, conv[k] + x[k/2] is even?
        # conv[k] = sum x[i]x[k-i].
        # If k is even, i=k/2 is the only self-pair.
        # Other pairs come in pairs (i, k-i) and (k-i, i).
        # So conv[k] = 2 * (sum_{i < k/2} x[i]x[k-i]) + x[k/2].
        # conv[k] + x[k/2] = 2 * (sum_{i < k/2} x[i]x[k-i]) + 2*x[k/2].
        # This is always even. So R is always an integer.
        
        # Penalty for missing coverage
        missing = np.sum(np.maximum(0, 1 - R))
        penalty = missing * 10000.0
        
        # Cost to minimize peak: use sum of squares
        sq_sum = np.sum(R**2)
        
        return penalty + sq_sum

    # Initial random set
    # Try to start with a reasonably sized set
    # Random size between 30 and 60
    size = random.randint(30, 60)
    indices = random.sample(range(n), size)
    x = np.zeros(n, dtype=np.int8)
    x[indices] = 1
    
    current_cost = get_cost(x)
    
    # Simulated Annealing parameters
    start_temp = 100.0
    end_temp = 0.01
    steps = 5000
    # Adjust steps based on time limit, but 5000 is fast
    # np.convolve on 129 elements is very fast.
    
    for i in range(steps):
        # Linear cooling
        temp = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Pick a random move: flip a bit
        idx = random.randint(0, n - 1)
        x[idx] = 1 - x[idx]
        
        new_cost = get_cost(x)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_cost = new_cost
        else:
            # Revert
            x[idx] = 1 - x[idx]
            
    # Convert back to list of indices
    result = np.nonzero(x)[0].tolist()
    
    # Ensure non-empty
    if not result:
        return [64]
        
    return result
