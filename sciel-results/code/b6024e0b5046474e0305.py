import time
import math

_start_time = None

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > 295:
        return 0.0

    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    # Primary: Overall balance
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # Soft-max of absolute deviations
    abs_devs = [abs(c0 - target), abs(c1 - target), abs(c2 - target)]
    max_abs_dev = max(abs_devs)
    alpha = 2.0
    exp_vals = [math.exp(alpha * (d - max_abs_dev)) for d in abs_devs]
    sum_exp = sum(exp_vals)
    soft_max_dev = max_abs_dev + math.log(sum_exp) / alpha
    dev_combined = dev + 0.5 * soft_max_dev**2
    
    # Secondary: Balance of non-zeros (1s vs 2s)
    # If c0 is fixed, we want c1 and c2 to be as close as possible.
    c_nonzero = c1 + c2
    if c_nonzero > 0:
        target_nonzero = c_nonzero / 2.0
        dev_nonzero = (c1 - target_nonzero)**2 + (c2 - target_nonzero)**2
    else:
        dev_nonzero = 0.0
        
    # Tertiary: Structure (Transitions)
    transitions = 0
    for i in range(1, n):
        if vector[i] != vector[i-1]:
            transitions += 1
    trans_norm = transitions / max(n - 1, 1)
    trans_penalty = -0.1 * trans_norm
    
    # Quaternary: Autocorrelation (3-AP risk)
    autocorr = 0.0
    num_shifts = min(n // 3, 20)
    if num_shifts > 0:
        for d in range(1, num_shifts + 1):
            matches = 0
            for i in range(n - 2 * d):
                if vector[i] == vector[i + d]:
                    matches += 1
            autocorr += matches / max(n - 2 * d, 1)
        autocorr /= num_shifts
    autocorr_penalty = -0.15 * autocorr
    
    # Tie-breaker
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    h_norm = h / 0x7FFFFFFF

    # Combine
    # Weighting:
    # 1. Overall balance (strongest)
    # 2. Non-zero balance (medium)
    # 3. Structure (weak)
    return -dev_combined - 0.2 * dev_nonzero + trans_penalty + autocorr_penalty + 1e-9 * h_norm
