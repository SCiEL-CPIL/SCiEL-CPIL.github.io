import random
import itertools

def construct():
    def is_valid(S):
        n = len(S)
        sums = set()
        # Generate all non-empty subset sums
        # Bitmask iteration
        for mask in range(1, 1 << n):
            s = 0
            # Optimization: use built-in sum over generator?
            # sum(S[i] for i in range(n) if mask & (1 << i))
            # This is slower than manual loop in pure python?
            # Let's use the generator for clarity, it's C-optimized sum
            s = sum(S[i] for i in range(n) if mask & (1 << i))
            if s in sums:
                return False
            sums.add(s)
        
        # Check divisibility
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        for s in sum_list:
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sums:
                    return False
                multiple += s
        return True

    # Try to find a valid set of size N
    # Using interval [128, 192)
    candidates = list(range(128, 192))
    
    # We will try sizes from 14 down to 10
    # Increase attempts for smaller sizes if larger ones fail quickly
    for target_size in range(14, 9, -1):
        # Number of attempts: more for larger N? No, larger N is harder.
        # But if N=14 is possible, we want it.
        # Let's do 2000 attempts for N=14, 5000 for N=13, etc.
        attempts = 2000 + (14 - target_size) * 1000
        for _ in range(attempts):
            S = random.sample(candidates, target_size)
            if is_valid(S):
                return S
                
    # If all fail, fallback to a smaller known valid set or random search in wider range
    # Try a few random sets from 1..256
    for target_size in range(10, 4, -1):
        for _ in range(1000):
            S = random.sample(range(1, 257), target_size)
            if is_valid(S):
                return S

    return [31, 32, 34]
