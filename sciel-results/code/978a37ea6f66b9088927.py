import time
import math

def _squarefree_kernel(n):
    """Return frozenset of primes with odd exponent in factorization of n."""
    if n <= 1:
        return frozenset()
    result = frozenset()
    f = n
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result = result | frozenset([d])
        d += 1 if d == 2 else 2
    if f > 1:
        result = result | frozenset([f])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    target = frozenset([2, 3, 5, 7])
    
    min_val = 211
    max_possible = 10210
    
    # Precompute kernels
    kernels = {}
    for n in range(min_val, max_possible + 1):
        kernels[n] = _squarefree_kernel(n)
    
    # DP with backtracking: for each kernel state, store the list of numbers used
    # To keep memory manageable, only store one path per kernel state
    best_for_kernel = {frozenset(): []}
    
    for current_max in range(min_val, max_possible + 1):
        if time.time() > deadline:
            break
        k = kernels[current_max]
        # Snapshot current state to avoid using current_max multiple times
        current_states = dict(best_for_kernel)
        for kernel_state, numbers in current_states.items():
            new_kernel = kernel_state ^ k
            new_numbers = numbers + [current_max]
            if new_kernel not in best_for_kernel:
                best_for_kernel[new_kernel] = new_numbers
            # Check if we hit target
            if new_kernel == target:
                # Found a solution
                if time.time() > deadline:
                    break
                return new_numbers
    
    # Fallback
    return [840]
