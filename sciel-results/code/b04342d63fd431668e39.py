import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s wall-clock kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    n_primes = len(primes)

    best_n = 1
    best_score = -1

    def tau_perp_from_divisors(divisors):
        """Calculate score from a sorted list of divisors."""
        divisors.sort()
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                score += 1
        return score

    def dfs(idx, max_exp, current_n, current_tau, divisors):
        nonlocal best_n, best_score

        if time.time() - start_time > TIME_LIMIT:
            return

        if idx >= n_primes:
            # Leaf node: evaluate
            if current_n > 1:
                score = tau_perp_from_divisors(list(divisors))
                if score > best_score:
                    best_score = score
                    best_n = current_n
            return

        p = primes[idx]

        # Upper bound pruning: even if all remaining primes are included with max exp,
        # the total tau is bounded. tau_perp <= tau - 1.
        # Max possible tau from here: current_tau * (max_exp+1)^(n_primes - idx)
        # But that's very loose. Better: just check if current_tau - 1 could beat best.
        # Actually, tau can only grow, so the min final tau is current_tau (if we stop).
        # But we must use all primes in the path... no, we can stop at any point.
        # The key insight: more distinct primes = more divisors = more adjacent pairs.
        # So we should strongly prefer using all 15 primes.

        # Determine max exponent for this prime
        limit = max_val // current_n
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1

        e_max = min(e_max_possible, max_exp)

        if e_max < 1:
            # Can't include this prime, skip it (but still evaluate at leaf)
            dfs(idx + 1, max_exp, current_n, current_tau, divisors)
            return

        # Try exponents. Since more distinct primes is better, we want to include this prime.
        # Try e=1 first (to include the prime with minimal cost), then higher exponents.
        # Actually, let's try all valid exponents, but prioritize those that keep
        # room for more distinct primes.
        
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)

            # Build new divisors
            current_divs = list(divisors)
            factor = 1
            new_divs = list(divisors)
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    new_divs.append(d * factor)

            # Prune: even with best case for remaining primes, can we beat best_score?
            # Loose bound: final_tau >= next_tau, so tau_perp <= final_tau - 1
            # But final_tau could be much larger. Use: if next_tau - 1 <= best_score
            # AND we can't add more primes, prune. But we can always add more primes.
            # Better bound: the minimum additional divisors from remaining primes is 1 each.
            # So min final tau = next_tau * 2^(n_primes - idx - 1) if we can include all remaining.
            # But that might exceed max_val. Use a simpler check.
            
            # Simple effective bound: if next_tau is already very large, it's promising.
            # Just proceed and let the leaf evaluation decide.
            
            dfs(idx + 1, e, next_n, next_tau, new_divs)

            if time.time() - start_time > TIME_LIMIT:
                return

        # Also try skipping this prime (e=0)
        # This allows exploring configurations with fewer distinct primes
        # But based on findings, this is less likely to be optimal
        if time.time() - start_time <= TIME_LIMIT:
            dfs(idx + 1, max_exp, current_n, current_tau, divisors)

    # Start DFS
    dfs(0, 60, 1, 1, [1])

    # Safety: if no valid n found (shouldn't happen), return a known good value
    if best_n <= 1:
        # Product of first 15 primes with exponent 1
        n = 1
        for p in primes:
            n *= p
        if n <= max_val:
            best_n = n
        else:
            best_n = 2  # fallback

    return best_n


if __name__ == "__main__":
    n = construct()
    print(f"Best n: {n}")
