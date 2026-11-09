import time
import math

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    
    # List of small primes to check
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def generate_smooth_numbers(limit, primes_list):
        """Generate all numbers <= limit whose prime factors are in primes_list."""
        smooth_set = {1}
        queue = [1]
        while queue:
            curr = queue.pop(0)
            for p in primes_list:
                next_val = curr * p
                if next_val > limit:
                    break
                if next_val not in smooth_set:
                    smooth_set.add(next_val)
                    queue.append(next_val)
        return smooth_set

    def generate_smooth_bool_array(limit, primes_list):
        """Generate a boolean array where arr[i] is True if i is smooth (prime factors in primes_list)."""
        arr = [False] * (limit + 1)
        arr[1] = True
        queue = [1]
        while queue:
            curr = queue.pop(0)
            for p in primes_list:
                next_val = curr * p
                if next_val > limit:
                    break
                if not arr[next_val]:
                    arr[next_val] = True
                    queue.append(next_val)
        return arr

    for p in primes:
        if time.time() > deadline:
            break
            
        # Find index of p in primes
        idx = primes.index(p)
        allowed_primes = primes[:idx+1]
        
        # Use boolean array for faster lookup
        smooth_arr = generate_smooth_bool_array(N, allowed_primes)
        
        # Check for pair: scan for smallest a where both a and N-a are smooth
        found_pair = None
        for a in range(1, N):
            b = N - a
            if smooth_arr[a] and smooth_arr[b]:
                found_pair = [a, b]
                break
        
        if found_pair:
            return found_pair

    # Fallback: Return seed result if no small prime pair found
    # This should not happen for N=123457 with these primes
    return [1, 123456]
