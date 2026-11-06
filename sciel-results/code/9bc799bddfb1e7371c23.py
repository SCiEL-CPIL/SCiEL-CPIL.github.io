import math

def construct():
    N = 123457
    
    def generate_smooth_numbers(n, primes):
        """Generate all primes-smooth numbers up to n."""
        smooths = {1}
        queue = [1]
        while queue:
            x = queue.pop(0)
            for p in primes:
                y = x * p
                if y <= n:
                    if y not in smooths:
                        smooths.add(y)
                        queue.append(y)
        return smooths

    # Try increasing sets of small primes
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    ]
    
    best_pair = None
    best_val = float('inf')
    
    for primes in prime_sets:
        smooths = generate_smooth_numbers(N, primes)
        # Check for a + b = N with a, b in smooths
        found = False
        for a in smooths:
            b = N - a
            if b in smooths:
                # Calculate max prime factor
                # Since a and b are p-smooth, their LPF is at most max(primes)
                # We can just return this pair as it's very good
                # To be precise, we should calculate LPF to ensure it's minimal among found?
                # Actually, if we find a pair with primes up to 5, LPF is 5.
                # If we find with primes up to 7, LPF is 7.
                # 5 is better than 7. So we should return the first one found?
                # Yes, because we iterate primes in increasing order.
                # The first valid pair found will have the smallest possible LPF
                # bounded by the max prime in the current set.
                # However, it's possible that a pair exists in [2,3,5] but we missed it?
                # No, generate_smooth_numbers generates ALL such numbers.
                # So if a pair exists with LPF <= 5, we find it in the first iteration.
                # If not, we try LPF <= 7, etc.
                # So the first found pair is optimal within the smooth number framework.
                # Is it globally optimal?
                # The global optimum is min over all a of max(P(a), P(N-a)).
                # If the global optimum has P(a)=101, P(b)=103, then max is 103.
                # If we find a smooth pair with max prime 37, that's 37.
                # 37 < 103, so smooth pair is better.
                # So we just return the first pair found.
                return [a, b]
                
    # Fallback if no smooth pair found up to 37
    # This is unlikely but possible.
    # We can do a quick scan of a subset or random sampling.
    # Or just return [1, N-1] which has P(1)=1, P(N-1) maybe large.
    # Better: pick a random a and check?
    # Or use a simple heuristic: try a few values near N/2.
    
    # Let's do a small local search around N/2 for a "reasonably" smooth number
    # Or just return [1, N-1] as a valid candidate, though not optimal.
    # But we can do better. Let's try a few random numbers.
    import random
    for _ in range(1000):
        a = random.randint(1, N-1)
        b = N - a
        # Quick check if a or b is divisible by small primes only?
        # This is getting complicated.
        # Let's just return a deterministic fallback.
        # The task says "Minimize". If we can't find a good one, we return a valid one.
        # But the smooth search up to 37 covers a lot.
        # Numbers with LPF <= 37 are quite dense?
        # Count of 37-smooth numbers up to 123457.
        # It's likely we find a pair.
        
    # If we get here, return a default
    return [1, N-1]
