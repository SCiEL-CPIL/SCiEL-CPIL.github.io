import time

def construct():
    S = 123457
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Primes to check
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    # Generate smooth numbers for a given prime limit
    def get_smooth_numbers(N, prime_list):
        nums = {1}
        for p in prime_list:
            new_nums = set()
            for x in nums:
                y = x
                while y <= N:
                    new_nums.add(y)
                    y *= p
            nums = new_nums
        return nums

    # We need to check primes in increasing order.
    # For p in primes:
    #   Generate set of p-smooth numbers
    #   Check if any a in set has S-a in set
    
    # Optimization: We can generate cumulatively or just regenerate.
    # Regeneration is fast for small prime counts.
    
    for i in range(len(primes)):
        if time.time() > deadline:
            break
        
        p = primes[i]
        # Generate all numbers composed of primes up to p
        smooths = get_smooth_numbers(S, primes[:i+1])
        
        # Check for valid pair
        found = False
        for a in smooths:
            b = S - a
            if b in smooths:
                # Found optimal pair for this prime limit
                # Since we iterate primes in increasing order, the first found is minimal max prime factor
                # Note: P(a) <= p and P(b) <= p.
                # Is it possible that P(a) < p and P(b) = p? Yes.
                # Is it possible that P(a) = p and P(b) < p? Yes.
                # The max is <= p.
                # Could there be a solution with max < p?
                # If so, it would have been found in a previous iteration (with smaller prime limit).
                # So this is the minimal max prime factor.
                return [a, b]
        
        # If not found, continue to next prime
    
    # Fallback: Return the seed solution or a reasonable one
    # Seed was [1, 123456], max prime 89.
    # If we didn't find better, return that.
    # But we should try to find something better if we ran out of primes list?
    # The list goes up to 113.
    # P(123456) = 89.
    # So if we didn't find anything with prime <= 89, then 89 is the best we can do with this method?
    # Wait, if we didn't find a pair for p=89, it means there is no pair with max prime factor <= 89?
    # But [1, 123456] has max prime factor 89.
    # 1 is 89-smooth. 123456 is 89-smooth (primes 2,3,29,89).
    # So it MUST be found in the loop for p=89.
    
    return [1, 123456]
