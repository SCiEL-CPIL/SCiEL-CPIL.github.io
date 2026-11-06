import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 85

    # Generate primes up to target (or enough to cover the search)
    # We only need primes up to the optimal B.
    # We can generate primes on the fly or precompute a small list.
    # Primes up to 1000 is safe for B-smooth search if optimal B is small.
    # If optimal B is large, we might need more.
    # But if B is large, the set of smooth numbers grows, but we stop as soon as we find a pair.
    
    # Simple prime generator
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # We will iterate B through primes: 2, 3, 5, 7, ...
    # Start with B=2
    b_val = 2
    # List of primes to use as B
    # We can just check integers and filter primes, or generate primes.
    # Let's generate primes up to target.
    # Sieve of Eratosthenes for primes only is fast.
    # But we are trying to avoid full sieve of lpf.
    # A prime sieve is O(N log log N) but only stores booleans.
    # 123457 booleans is tiny.
    
    # Actually, we can just iterate candidate B and check primality.
    # B=2, 3, 5, 7, 11...
    
    def get_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit that are smooth over the given primes.
        primes should be sorted.
        """
        smooth = {1}
        for p in primes:
            new_smooth = set(smooth)
            # For each existing smooth number, multiply by p repeatedly
            # We need to be careful to use the *previous* set for expansion?
            # No, standard way:
            # new_smooth starts as copy of smooth.
            # For each x in smooth (original), add x*p, x*p^2...
            # But x*p might be > limit.
            # Also, we need to include numbers formed by *new* primes?
            # The set of B-smooth numbers is the closure under multiplication of primes in B.
            # Iterating primes one by one and expanding the set works.
            # However, the expansion `for x in smooth` uses the set *before* adding p?
            # Yes, if we do:
            # old_smooth = smooth
            # smooth = set(old_smooth)
            # for x in old_smooth:
            #    y = x * p
            #    while y <= limit:
            #       smooth.add(y)
            #       y *= p
            # This ensures we generate all combinations.
            
            old_smooth = smooth
            smooth = set(old_smooth)
            for x in old_smooth:
                y = x * p
                while y <= limit:
                    smooth.add(y)
                    y *= p
        return smooth

    # Iterate B
    # We need a list of primes.
    # Let's generate primes up to target using a simple sieve (boolean array).
    # This is different from the LPF sieve.
    is_prime_arr = [True] * (target + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(target)) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, target + 1, i):
                is_prime_arr[j] = False
    
    primes = [i for i in range(2, target + 1) if is_prime_arr[i]]
    
    current_primes = []
    
    for p in primes:
        if time.time() > deadline:
            break
            
        current_primes.append(p)
        
        # Generate smooth numbers for current set of primes
        smooth_set = get_smooth_numbers(current_primes, target)
        
        # Check for pair
        found = False
        # To ensure a < b or just any a
        # We can iterate a in smooth_set
        # If target - a is in smooth_set, we found a pair.
        # Note: a and target-a must be positive.
        # Since a >= 1, target-a <= target-1.
        # Since smooth_set contains 1, and max is target, all are positive.
        
        for a in smooth_set:
            b = target - a
            if b in smooth_set:
                # Found optimal pair for this B
                # Since we iterate B in increasing order, this is the global minimum
                return [a, b]
                
    # If loop finishes without finding (should not happen as B=target includes all)
    # Fallback to simple sieve logic if timeout or failure
    # But B will reach target, so smooth_set will be all numbers 1..target.
    # Then a=1, b=target-1 will be found.
    
    # Safety fallback
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    half = target // 2 + 1
    for a in range(1, half):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            if best_max_p == 3:
                break
    return [best_a, best_b]
