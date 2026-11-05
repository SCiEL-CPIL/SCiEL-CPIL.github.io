import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    min_n = 10000
    max_n = 1000000

    # Primes to try in increasing order
    primes_list = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def generate_smooth_numbers(primes, lo, hi):
        """Generate all P-smooth numbers in [lo, hi] using DFS."""
        result = set()
        n_primes = len(primes)

        def dfs(idx, current):
            if current > hi:
                return
            if current >= lo:
                result.add(current)
            if idx >= n_primes:
                return
            p = primes[idx]
            # Multiply by p^k for k >= 0
            val = current
            while val <= hi:
                dfs(idx + 1, val)
                if val > hi // p:
                    break
                val *= p

        dfs(0, 1)
        return sorted(result)

    # Try increasing P values
    for P in primes_list:
        if time.time() > deadline:
            break

        # Generate P-smooth numbers in [min_n, max_n]
        smooth = generate_smooth_numbers([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97][:P], min_n, max_n)

        # Actually, let's just use primes up to P
        primes_up_to_P = [p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97] if p <= P]
        smooth = generate_smooth_numbers(primes_up_to_P, min_n, max_n)

        # Check for consecutive pairs
        for i in range(len(smooth) - 1):
            if smooth[i + 1] - smooth[i] == 1:
                return smooth[i]

    # Fallback: if no pair found with smooth generation, do a linear scan
    # This should rarely be reached
    if time.time() < deadline - 2.0:
        N = max_n
        lpf = [0] * (N + 1)
        for i in range(2, N + 1):
            if lpf[i] == 0:
                for j in range(i, N + 1, i):
                    lpf[j] = i

        best_n = min_n
        best_val = max(lpf[min_n], lpf[min_n + 1])
        for n in range(min_n + 1, N):
            val = max(lpf[n], lpf[n + 1])
            if val < best_val:
                best_val = val
                best_n = n
            if time.time() > deadline:
                break
        return best_n

    # Absolute fallback
    return min_n
