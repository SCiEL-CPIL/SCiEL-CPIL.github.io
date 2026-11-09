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
        smooth = set([1])
        # We can do this by iterative multiplication
        # Start with 1, multiply by each prime, add if <= limit
        # Repeat until no new numbers are added
        # Or use nested loops based on the count of primes
        # Given small limit, iterative expansion is fine but might be slow if not careful.
        # Better: recursive generation or BFS.
        
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

    for p in primes:
        if time.time() > deadline:
            break
            
        # Generate p-smooth numbers
        # Note: p-smooth means prime factors <= p.
        # So for p=5, primes are [2, 3, 5].
        # For p=7, primes are [2, 3, 5, 7].
        
        # We need the list of primes <= p
        # Since 'primes' is sorted, we can take primes[:idx+1]
        # But we are iterating 'p' from the list 'primes'.
        # Let's just find the index.
        # Actually, we can just pass the list of primes up to p.
        
        # Optimization: If p=2, primes_list=[2]
        # If p=3, primes_list=[2,3]
        # etc.
        
        # Let's refine generate_smooth_numbers to take a list of allowed primes.
        
        # Find index of p in primes
        idx = primes.index(p)
        allowed_primes = primes[:idx+1]
        
        smooth_set = generate_smooth_numbers(N, allowed_primes)
        
        # Check for pair
        found_pair = None
        # To minimize max(P(a), P(b)), any pair of p-smooth numbers works.
        # The max prime factor will be <= p.
        # We just need to find ONE such pair.
        # To be deterministic, we can pick the smallest a.
        
        for a in sorted(smooth_set):
            b = N - a
            if b in smooth_set:
                # Ensure a, b are positive
                if a > 0 and b > 0:
                    found_pair = [a, b]
                    break
        
        if found_pair:
            return found_pair

    # Fallback: Return seed result if no small prime pair found
    # This should not happen for N=123457 with these primes
    return [1, 123456]
