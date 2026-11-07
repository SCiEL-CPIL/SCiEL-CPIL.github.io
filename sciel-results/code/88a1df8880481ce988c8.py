import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin for 90s limit
    
    divisors = DIVISORS
    n = N
    max_val = MAX_VAL
    
    # Precompute LCM of all divisors
    period = reduce(_lcm, divisors)
    
    # Strategy: 
    # 1) Sliding window over the sorted list of all valid multiples
    # 2) For each window, check if it contains at least one multiple of each divisor
    # 3) Use backtracking to find a valid assignment (distinct values)
    # 4) Return the assignment with minimal span
    
    def can_assign(L, R):
        """Check if we can assign distinct multiples to each divisor in [L, R]."""
        # Compute candidates for each divisor
        cands = []
        for d in divisors:
            k = (L + d - 1) // d
            vals = []
            v = k * d
            while v <= R:
                vals.append(v)
                v += d
            cands.append(vals)
        
        # If any divisor has no candidates, return False
        if any(len(c) == 0 for c in cands):
            return False, None
        
        # Backtracking with ordering by fewest candidates first
        order = sorted(range(n), key=lambda i: len(cands[i]))
        
        used = set()
        assign = [0] * n
        
        def bt(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in cands[i]:
                if v not in used:
                    used.add(v)
                    assign[i] = v
                    if bt(idx + 1):
                        return True
                    used.remove(v)
                    assign[i] = 0
            return False
        
        if bt(0):
            return True, assign
        return False, None
    
    # Build the sorted list of all multiples of any divisor up to max_val
    # This is the "universe" of candidate values
    all_multiples = set()
    for d in divisors:
        for k in range(1, max_val // d + 1):
            all_multiples.add(k * d)
    
    sorted_multiples = sorted(all_multiples)
    total = len(sorted_multiples)
    
    # For each position in sorted_multiples, we want to know which divisors it's a multiple of
    # Precompute: for each value, which divisors divide it
    # But that's too expensive. Instead, use the sliding window approach:
    # For a window [sorted_multiples[i], sorted_multiples[j]], 
    # check if for each divisor d, there's at least one multiple in the window.
    
    # To make this efficient, we can use a two-pointer / sliding window:
    # For each left pointer i, find the smallest right pointer j such that
    # the window [sorted_multiples[i], sorted_multiples[j]] contains at least
    # one multiple of each divisor.
    
    # But checking "contains at least one multiple of each divisor" is still O(n) per window.
    # With total ~ 1e6 * (1/6 + 1/10 + ...) ~ 1e6 * 0.6 ~ 600k values,
    # and two pointers, this is O(total * n) ~ 6e6 operations, which is fine.
    
    # Actually, let's be smarter: use a frequency counter.
    # For each value in the window, we can determine which divisors it satisfies.
    # But a value can be a multiple of multiple divisors.
    
    # Let's use the following approach:
    # For each value v in sorted_multiples, precompute the set of divisors it's a multiple of.
    # Then use a sliding window with a count array.
    
    # Precompute divisor masks for each value
    # divisor_masks[i] is a bitmask where bit d is set if sorted_multiples[i] is a multiple of divisors[d]
    divisor_masks = []
    for v in sorted_multiples:
        mask = 0
        for d_idx, d in enumerate(divisors):
            if v % d == 0:
                mask |= (1 << d_idx)
        divisor_masks.append(mask)
    
    # Now sliding window:
    # count[d] = number of values in current window that are multiples of divisors[d]
    # We want all count[d] >= 1
    
    best_span = max_val - 1
    best_assign = None
    
    left = 0
    counts = [0] * n
    all_covered = 0  # number of divisors with count >= 1
    
    for right in range(total):
        # Add sorted_multiples[right]
        mask = divisor_masks[right]
        for d_idx in range(n):
            if mask & (1 << d_idx):
                if counts[d_idx] == 0:
                    all_covered += 1
                counts[d_idx] += 1
        
        # Shrink from left while all_covered == n
        while all_covered == n and left <= right:
            # Check if this window is valid
            L = sorted_multiples[left]
            R = sorted_multiples[right]
            span = R - L
            
            if span < best_span:
                # Try to find an assignment
                feasible, assign = can_assign(L, R)
                if feasible:
                    best_span = span
                    best_assign = assign
                    # Don't break, keep shrinking to find even smaller spans
                    # But we need to be careful: shrinking might invalidate
                    # Actually, we should record the best and continue
            
            # Remove sorted_multiples[left]
            mask = divisor_masks[left]
            for d_idx in range(n):
                if mask & (1 << d_idx):
                    counts[d_idx] -= 1
                    if counts[d_idx] == 0:
                        all_covered -= 1
            left += 1
        
        # Time check
        if time.time() > deadline:
            break
    
    # If we found a good assignment, return it
    if best_assign is not None:
        # Validate
        for i, d in enumerate(divisors):
            if best_assign[i] % d != 0:
                raise ValueError(f"Invalid: {best_assign[i]} not divisible by {d}")
            if best_assign[i] < 1 or best_assign[i] > max_val:
                raise ValueError(f"Invalid: {best_assign[i]} out of range")
        if len(set(best_assign)) != n:
            raise ValueError("Invalid: not all distinct")
        return [int(x) for x in best_assign]
    else:
        # Fallback: find any valid assignment
        # Use a greedy approach: for each divisor, pick the smallest unused multiple
        used = set()
        result = []
        for d in divisors:
            v = d
            while v in used or v > max_val:
                v += d
            result.append(int(v))
            used.add(v)
        return result

# EVOLVE-BLOCK-END
