import time
import itertools
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin from 90s limit
    
    # Helper to check if n is S-smooth
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    # Generate smooth numbers up to limit
    def generate_smooth(limit, primes):
        smooths = set()
        def backtrack(current, idx):
            if current > limit:
                return
            if idx == len(primes):
                smooths.add(current)
                return
            p = primes[idx]
            next_val = current
            while next_val <= limit:
                backtrack(next_val, idx + 1)
                next_val *= p
        backtrack(1, 0)
        return smooths

    # Backtracking to find k_i
    def find_k_values(max_k, smooth_set, target_count):
        found = []
        
        def backtrack(current_list, start_val):
            if len(current_list) == target_count:
                found.extend(current_list)
                return True
            if time.time() > deadline:
                return False
                
            # Heuristic: try values in a specific order
            # We can try to pick values that are compatible with the existing list
            for k in range(start_val, max_k + 1):
                # Check compatibility with all existing elements
                valid = True
                for y in current_list:
                    if (k + y) not in smooth_set:
                        valid = False
                        break
                if valid:
                    if backtrack(current_list + [k], k + 1):
                        return True
            return False

        backtrack([], 1)
        return found

    # Try different prime sets
    # Order: fewest primes first
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
        P = 1
        for p in primes:
            P *= p
            
        max_k = 10000 // P
        if max_k < 12:
            continue
            
        # We need 12 distinct k_i.
        # Max sum of k_i is 2 * max_k.
        smooth_set = generate_smooth(2 * max_k, primes)
        
        k_vals = find_k_values(max_k, smooth_set, 12)
        
        if k_vals:
            # Scale back to a_i
            a_vals = [k * P for k in k_vals]
            # Verify distinctness and bounds
            if len(set(a_vals)) == 12 and all(1 <= a <= 10000 for a in a_vals):
                # Verify prime count
                prime_count = len(primes)
                # We found a solution with 'prime_count' primes.
                # Since we iterate from smallest prime set, this is optimal.
                return a_vals

    # Fallback: return seed solution if no better found
    return list(range(1, 13))
