import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    N = 123457

    # Primes in increasing order — we will test K = each prime
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    def generate_smooth(primes_list, limit):
        """Generate all primes_list-smooth numbers <= limit via DFS."""
        smooths = set()
        n_primes = len(primes_list)

        def gen(idx, val):
            if val > limit:
                return
            smooths.add(val)
            if idx == n_primes:
                return
            p = primes_list[idx]
            # Use p: multiply by p repeatedly (staying with same idx)
            nv = val
            while nv * p <= limit:
                nv *= p
                gen(idx, nv)
            # Skip p: move to next prime
            gen(idx + 1, val)

        gen(0, 1)
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
    for i in range(len(primes)):
        if time.time() - start_time > DEADLINE - 10:
            # Not enough time left for smooth approach; fall back to exhaustive
            return exhaustive_fallback(start_time + DEADLINE)

        current_primes = primes[:i + 1]
        max_p = current_primes[-1]

        smooths = generate_smooth(current_primes, N)

        # Check if any a in smooths has b = N - a also in smooths
        found = False
        best_pair = None
        for a in smooths:
            b = N - a
            if b in smooths and b >= 1:
                found = True
                # Track the pair that minimizes the actual max(P(a),P(b))
                # Since all are max_p-smooth, max(P(a),P(b)) <= max_p
                # But we want the actual minimum; for now take first found
                # (all pairs at this K have the same upper bound max_p)
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
