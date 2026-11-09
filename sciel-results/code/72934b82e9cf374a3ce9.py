import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    N = 123457

    def generate_smooth_nested(primes_list, limit):
        """Generate all primes_list-smooth numbers <= limit via nested loops."""
        smooths = set([1])
        for p in primes_list:
            new_smooths = set()
            for s in smooths:
                v = s
                while v <= limit:
                    new_smooths.add(v)
                    v *= p
            smooths = new_smooths
        return smooths

    # Fallback: exhaustive LPF scan (guaranteed correct, used if smooth approach times out)
    def exhaustive_fallback(deadline):
        # Sieve for largest prime factor
        lpf = [1] * (N + 1)
        for i in range(2, N + 1):
            if lpf[i] == 1:
                for j in range(i, N + 1, i):
                    lpf[j] = i
        best_max_pf = float('inf')
        best_a = 1
        best_b = N - 1
        for a in range(1, N // 2 + 1):
            if time.time() > deadline:
                break
            b = N - a
            m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
            if m < best_max_pf:
                best_max_pf = m
                best_a = a
                best_b = b
                if m <= 3:
                    break
        return [best_a, best_b]

    # Primary strategy: try increasing K (prime bound)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    for i in range(len(primes)):
        if time.time() - start_time > DEADLINE - 10:
            # Not enough time left for smooth approach; fall back to exhaustive
            return exhaustive_fallback(start_time + DEADLINE)

        current_primes = primes[:i + 1]
        max_p = current_primes[-1]

        smooths = generate_smooth_nested(current_primes, N)

        # Check if any a in smooths has b = N - a also in smooths
        # We iterate over the sorted smooths to find the first valid pair.
        # Since we are increasing K, the first K that yields a valid pair is optimal.
        # Within that K, any pair is valid. The "value" is max(P(a), P(b)).
        # Since both are K-smooth, max(P(a), P(b)) <= K.
        # It might be strictly less than K, but that would have been found in a previous iteration.
        # So the first found pair for the minimal K is the answer.
        
        found = False
        best_pair = None
        for a in sorted(smooths):
            b = N - a
            if b in smooths and b >= 1:
                found = True
                best_pair = (a, b)
                break

        if found:
            a, b = best_pair
            return [a, b]

    # If we exhaust all primes without finding a pair (shouldn't happen for N=123457),
    # fall back to exhaustive scan
    if time.time() - start_time < DEADLINE - 5:
        return exhaustive_fallback(start_time + DEADLINE)

    # Last resort: return a valid (if not optimal) decomposition
    return [1, N - 1]
