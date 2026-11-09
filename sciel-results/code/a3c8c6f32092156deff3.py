"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    def is_valid(subset):
        """Check if all reciprocal subset sums of subset are distinct using integer arithmetic."""
        n = len(subset)
        if n == 0:
            return True
        # Compute LCM of all elements to scale reciprocals to integers
        # 1/x = lcm / x, so we work with integers lcm/x
        scale = 1
        for x in subset:
            scale = lcm(scale, x)
        # Now check if all subset sums of {scale/x for x in subset} are distinct
        weights = [scale // x for x in subset]
        # Compute all 2^n subset sums
        num_sums = 1 << n
        sums = [0] * num_sums
        for mask in range(1, num_sums):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + weights[bit_idx]
        # Check all distinct
        return len(set(sums)) == num_sums
    
    def greedy_build(order, current_set, deadline):
        """Try to extend current_set by adding elements from order if valid."""
        result = list(current_set)
        for candidate in order:
            if candidate in result:
                continue
            trial = result + [candidate]
            n = len(trial)
            if n <= 13:  # 2^13 = 8192 sums, manageable with integers
                if time.time() > deadline:
                    break
                if is_valid(trial):
                    result = trial
        return result
    
    best_result = []
    
    # Try multiple random orderings
    all_elements = list(range(1, 23))
    random.seed(42)
    
    # First: try ascending order (baseline)
    result = greedy_build(all_elements, [], deadline)
    if len(result) > len(best_result):
        best_result = result
    
    # Try descending order
    result = greedy_build(all_elements[::-1], [], deadline)
    if len(result) > len(best_result):
        best_result = result
    
    # Try many random permutations
    attempts = 0
    while time.time() < deadline - 5:
        random_order = all_elements[:]
        random.shuffle(random_order)
        result = greedy_build(random_order, [], deadline)
        if len(result) > len(best_result):
            best_result = result
            attempts += 1
        attempts += 1
        if attempts > 200:
            break
    
    # Local improvement: try to swap elements to increase size
    if best_result and len(best_result) < 13:
        current = list(best_result)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            in_set = set(current)
            not_in_set = [x for x in all_elements if x not in in_set]
            for i in range(len(current)):
                for j in not_in_set:
                    trial = current[:i] + current[i+1:] + [j]
                    n = len(trial)
                    if n <= 13:
                        if time.time() > deadline - 3:
                            break
                        if is_valid(trial):
                            # Check if this actually helps (we're keeping same size, but maybe enables further growth)
                            # For now, just accept swaps that maintain validity
                            pass  # Swaps don't increase size, skip for now
                if time.time() > deadline - 3:
                    break
    
    # Try to grow from best_result by adding one more element
    if best_result and time.time() < deadline - 2:
        in_set = set(best_result)
        for candidate in all_elements:
            if candidate not in in_set:
                trial = best_result + [candidate]
                n = len(trial)
                if n <= 13:
                    if is_valid(trial):
                        best_result = trial
                        break
    
    if best_result:
        return best_result
    
    # Fallback: return a known valid small set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
