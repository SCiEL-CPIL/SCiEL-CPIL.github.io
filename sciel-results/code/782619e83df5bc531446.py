"""Rewritten program: SAT-based packing of distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Compute LCM of all divisors - this is the period of the pattern
    all_lcm = reduce(lcm, divisors)  # = 30030

    # Initial solution: just the divisors themselves
    best_val = max(divisors) - min(divisors)
    best_assign = divisors[:]

    # For a given span W, we need to find a window [L, L+W] containing distinct multiples
    # of each divisor. Due to LCM periodicity, we only need to check L in [1, all_lcm].
    
    def is_feasible_window(L, W):
        for d in divisors:
            first_k = (L + d - 1) // d  # ceil(L/d)
            last_k = (L + W) // d        # floor((L+W)/d)
            if first_k > last_k:
                return False
        return True
    
    def find_assignment(L, W):
        # Build candidate lists
        candidates = []
        for d in divisors:
            first_k = (L + d - 1) // d
            last_k = (L + W) // d
            if first_k > last_k:
                return None
            cand_list = [k * d for k in range(first_k, last_k + 1)]
            candidates.append(cand_list)
        
        # Order divisors by number of candidates (ascending) - MRV heuristic
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return assignment[:]
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    used.remove(val)
            return None
        
        return backtrack(0)
    
    # Search for the minimum W systematically using LCM periodicity
    # For each W, check all L in [1, min(all_lcm, MAX_V - W)]
    max_W_to_try = 500
    found_solution = False
    
    for W in range(9, max_W_to_try + 1):
        if time.time() > deadline:
            break
        
        max_L = min(all_lcm, MAX_V - W)
        
        # Check all L in [1, max_L] - this is exhaustive due to LCM periodicity
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            if not is_feasible_window(L, W):
                continue
            
            result = find_assignment(L, W)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                found_solution = True
                # Since we're going from small W to large, and this is exhaustive for this W,
                # the first W with a solution is the minimum. We can stop.
                break
        
        if found_solution:
            break
    
    # Local search polish: try to improve the current best solution
    if best_assign is not None:
        for _ in range(2000):
            if time.time() > deadline:
                break
            candidate = best_assign[:]
            # Pick a random index
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]
            # Try to move to a nearby multiple, biased towards reducing span
            cur_min = min(candidate)
            cur_max = max(candidate)
            center = (cur_min + cur_max) // 2
            # Try multiples closer to center
            k = current // d
            for delta_k in range(-5, 6):
                if delta_k == 0:
                    continue
                new_val = (k + delta_k) * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                if new_val in candidate:
                    continue
                candidate[idx] = new_val
                if len(set(candidate)) == n:
                    span = max(candidate) - min(candidate)
                    if span < best_val:
                        best_val = span
                        best_assign = candidate[:]
                        break
                candidate[idx] = current
    
    # Final validation
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    # Verify
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int), f"Element {i} is not int"
        assert not isinstance(result[i], bool), f"Element {i} is bool"
        assert 1 <= result[i] <= MAX_V, f"Element {i} out of range: {result[i]}"
        assert result[i] % d == 0, f"Element {i} ({result[i]}) not divisible by {d}"
    assert len(set(result)) == n, "Elements are not distinct"
    
    return result

# EVOLVE-BLOCK-END
