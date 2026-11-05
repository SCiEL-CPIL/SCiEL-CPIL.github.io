import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, with P(1)=1.
    
    Strategy:
    1. Use incremental B-smooth number generation to find the smallest B
       such that two B-smooth numbers sum to 123457. This directly minimizes
       max(P(a), P(b)).
    2. The smooth set grows as we add more primes, so we check each level.
    3. Deadline guard for safety.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 75  # seconds, safety margin before 90s kill

    # First ~50 primes (enough to cover any reasonable B for N=123457)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
              109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
              173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
              233, 239, 241, 251]

    def generate_smooth(primes_subset, limit):
        """Generate all numbers <= limit whose prime factors are in primes_subset."""
        smooths = {1}
        for p in primes_subset:
            new_smooths = set()
            for s in smooths:
                v = s
                while v * p <= limit:
                    v *= p
                    new_smooths.add(v)
            smooths.update(new_smooths)
        return smooths

    best_result = [1, TARGET - 1]
    best_score = TARGET  # worst case upper bound

    # Iterate through increasing sets of primes
    for k in range(1, len(primes) + 1):
        if time.time() - start_time > DEADLINE:
            break

        current_primes = primes[:k]
        smooth_set = generate_smooth(current_primes, TARGET)

        # Check if any a in smooth_set has TARGET-a also in smooth_set
        found = False
        for a in smooth_set:
            b = TARGET - a
            if b > 0 and b in smooth_set:
                # Found a valid pair with max prime factor <= current_primes[-1]
                # This is optimal for this k, and since we go in order, it's globally optimal
                return [a, b]
            # Also track the best we've seen so far as a fallback
            # (though if we find any pair at level k, that's the answer)

        # If no pair found at this level, continue to next level
        # Update best_result with the best pair found so far (for fallback)
        # Actually, we can't easily get the score without checking all pairs,
        # but we know any pair at level k has max prime factor <= primes[k-1]

    # Fallback: if we didn't find a pair via smooth numbers (shouldn't happen for N=123457),
    # use a sieve-based approach to find the best pair
    if time.time() - start_time < DEADLINE:
        # Precompute largest prime factor for all numbers up to TARGET
        lpf = [0] * (TARGET + 1)
        lpf[1] = 1

        for i in range(2, TARGET + 1):
            if lpf[i] == 0:  # i is prime
                for j in range(i, TARGET + 1, i):
                    lpf[j] = i

        best_a = 1
        best_b = TARGET - 1
        best_score = max(lpf[best_a], lpf[best_b])

        half_target = TARGET // 2
        for a in range(2, half_target + 1):
            if time.time() - start_time > DEADLINE:
                break
            b = TARGET - a
            score = max(lpf[a], lpf[b])
            if score < best_score:
                best_score = score
                best_a = a
                best_b = b

        return [best_a, best_b]

    # Absolute fallback
    return best_result
