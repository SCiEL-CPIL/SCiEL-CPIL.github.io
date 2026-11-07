import random
import itertools

def construct():
    def is_valid(S):
        n = len(S)
        sums = set()
        # Use bitmask to generate subset sums
        # N is small (<= 12), so 2^N is small
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            if s in sums:
                return False
            sums.add(s)
        
        # Check divisibility
        # Create a set for fast lookup
        sum_set = sums
        max_sum = max(sum_set)
        for a in sum_set:
            multiple = 2 * a
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += a
        return True

    # Try sizes from 11 down to 5
    for target_size in range(11, 4, -1):
        # Narrow interval strategy
        # We want numbers to be large to reduce the chance of small sums dividing large sums?
        # Actually, if numbers are large, sums are large.
        # Let's try sampling from [100, 256]
        # Or even narrower: [200, 256]
        
        attempts = 500
        for _ in range(attempts):
            # Sample distinct integers
            # Bias towards larger numbers
            # Let's pick from [150, 256]
            S = random.sample(range(150, 257), target_size)
            if is_valid(S):
                return S
        
        # If not found in narrow range, try wider range [50, 256]
        for _ in range(500):
            S = random.sample(range(50, 257), target_size)
            if is_valid(S):
                return S
                
    # Fallback
    return [31, 32, 34]
