import time
import math

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Fallback: a=1, b=N-1. P(1)=1, P(N-1) is whatever it is.
    best_a = 1
    best_b = N - 1

    # We want to find the smallest K such that there exist a, b with a+b=N
    # and both a and b are K-smooth (i.e., all prime factors <= K).
    # This is equivalent to minimizing max(P(a), P(b)).
    #
    # Strategy:
    # 1. Generate primes in increasing order.
    # 2. Maintain a set of K-smooth numbers <= N as we add primes one by one.
    # 3. After adding a prime p, check if there's a pair (a, b) in the set
    #    such that a + b = N.
    # 4. If found, return immediately. This p is the optimal K (since we
    #    iterate primes in increasing order, the first p where a valid pair
    #    exists gives the minimal max prime factor).
    #
    # Why this works:
    # - If a and b are both K-smooth, then max(P(a), P(b)) <= K.
    # - The first prime p (in increasing order) for which a valid pair exists
    #   means that no pair exists with all prime factors < p, so the optimal
    #   max prime factor is exactly p (or possibly smaller, but we've already
    #   checked all smaller primes and found no pair).
    # - Actually, we need to be careful: the optimal max prime factor might be
    #   a prime that we haven't added yet? No, because we're adding primes in
    #   increasing order. If a pair exists with max prime factor = q, then when
    #   we add q (and all primes < q), the pair will be in the smooth set.
    #   So the first time we find a valid pair, that prime is the optimal K.

    # Generate primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]

    # Start with the set of 1-smooth numbers: just {1}
    smooth = {1}

    for p in primes:
        if time.time() > deadline:
            break

        # Add all new numbers that are p-smooth (i.e., formed by multiplying
        # existing smooth numbers by powers of p)
        # We need to be careful: we should only multiply by p^k where the
        # result is <= N.
        #
        # To do this efficiently, we iterate over the current smooth set and
        # for each element x, compute x * p, x * p^2, ... until > N.
        # But we must not modify smooth while iterating. So we collect new
        # numbers in a separate set, then add them to smooth.
        #
        # However, there's a subtlety: if we add x*p to smooth, then later
        # when we process another element y, we might generate y*p which
        # could equal x*p^2? No, that's fine. The issue is that we want to
        # generate ALL p-smooth numbers, which includes numbers with multiple
        # factors of p.
        #
        # The standard approach: for each existing smooth number x, multiply
        # by p repeatedly: x*p, x*p^2, ... and add all results <= N.
        # But we must do this based on the smooth set BEFORE adding new p-multiples.
        # Otherwise, we might miss some combinations.
        #
        # Actually, the correct way: the set of p-smooth numbers is the closure
        # of {1} under multiplication by primes <= p. So when we add p, the new
        # p-smooth numbers are all numbers of the form x * p^k where x is
        # (p-1)-smooth and k >= 1, x * p^k <= N.
        #
        # So we take a snapshot of the current smooth set (which is (p-1)-smooth),
        # and for each x in that snapshot, generate x*p, x*p^2, ... <= N.
        
        old_smooth = list(smooth)
        new_numbers = set()
        for x in old_smooth:
            val = x
            while val * p <= N:
                val *= p
                new_numbers.add(val)
        
        smooth.update(new_numbers)

        # Check if there's a valid pair
        # For each a in smooth, check if b = N - a is in smooth
        # We can restrict a to be <= N/2 to avoid duplicate checks, but
        # set lookup is O(1), so it's fine to just iterate over all.
        #
        # Optimization: if smooth is large, this could be slow. But for
        # small primes, smooth is small. Let's just do it.
        
        found = False
        for a in smooth:
            b = N - a
            if b in smooth and b > 0:
                # Found a valid pair with max prime factor <= p
                # Since we're iterating primes in increasing order, this p
                # is the smallest possible max prime factor.
                best_a = a
                best_b = b
                found = True
                break
        
        if found:
            return [best_a, best_b]

    # If we didn't find a pair (shouldn't happen since a=1, b=N-1 is always
    # a valid pair, and N-1 will eventually be in the smooth set when we add
    # all its prime factors), return the fallback.
    return [best_a, best_b]
