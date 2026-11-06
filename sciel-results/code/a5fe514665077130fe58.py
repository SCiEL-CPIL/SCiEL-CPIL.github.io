import math
import time
from functools import lru_cache

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s

    def check_time():
        return time.time() - start_time < TIME_LIMIT - 5

    # Primes up to ~200 (we'll try to include as many as possible)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
              113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
              181, 191, 193, 197, 199]

    max_val = 10**18

    def compute_tau_perp(divs_sorted):
        """Count adjacent coprime pairs in a sorted list of divisors."""
        if len(divs_sorted) < 2:
            return 0
        score = 0
        for i in range(len(divs_sorted) - 1):
            if math.gcd(divs_sorted[i], divs_sorted[i + 1]) == 1:
                score += 1
        return score

    def merge_divisors(existing_divs, p, exponent):
        """
        Given existing sorted divisors of current n, and a new prime p with exponent e,
        produce the sorted divisors of n * p^e.
        """
        # Generate all powers of p: 1, p, p^2, ..., p^e
        powers = [1]
        for _ in range(exponent):
            powers.append(powers[-1] * p)

        # Cross-multiply and sort
        new_divs = []
        for d in existing_divs:
            for pw in powers:
                new_divs.append(d * pw)
        new_divs.sort()
        return new_divs

    def marginal_gain(existing_divs, p, exponent, current_tau_perp):
        """
        Compute the increase in tau_perp when multiplying by p^exponent.
        Returns the new tau_perp value (not the delta).
        """
        new_divs = merge_divisors(existing_divs, p, exponent)
        return compute_tau_perp(new_divs)

    # Greedy construction:
    # Start with n=1, divisors=[1], tau_perp=0
    # At each step, try adding p^e for each remaining prime and small exponents,
    # pick the one that gives the best marginal gain in tau_perp while keeping n <= max_val.
    # Stop when no further addition improves the score or time is up.

    current_n = 1
    current_divs = [1]
    current_score = 0
    used_primes = set()

    # We'll do multiple rounds: in each round, try to add one prime power.
    # To avoid getting stuck, we allow trying different primes and exponents.
    # Key insight: adding a new prime p (not already in n) creates many new coprime adjacencies.
    # Adding a higher exponent of an existing prime has diminishing returns.

    # Strategy: prioritize adding NEW primes first (they create the most coprime pairs),
    # then consider increasing exponents of existing primes.

    # Phase 1: Add primes one by one, choosing the one that maximizes marginal gain.
    # For each candidate prime p not yet used, try exponent 1 first (since p^1 is cheapest).
    # If n * p <= max_val, consider it.

    while check_time():
        best_gain = -1
        best_choice = None  # (p, exponent, new_divs, new_score)

        for p in primes:
            if p in used_primes:
                continue
            # Try exponent 1
            if current_n * p <= max_val:
                new_divs = merge_divisors(current_divs, p, 1)
                new_score = compute_tau_perp(new_divs)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (p, 1, new_divs, new_score)

        # Also try increasing exponent of existing primes
        for p in list(used_primes):
            # Find current exponent of p in current_n
            temp = current_n
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            # Try exp + 1
            if current_n * p <= max_val:
                new_divs = merge_divisors(current_divs, p, 1)  # multiply by p^1
                new_score = compute_tau_perp(new_divs)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (p, 1, new_divs, new_score)

        if best_choice is None or best_gain <= 0:
            break

        p, exp, new_divs, new_score = best_choice
        current_n *= p ** exp
        current_divs = new_divs
        current_score = new_score
        used_primes.add(p)

        # If we've used many primes, the marginal gains will be small.
        # Continue until no improvement.

    # Phase 2: Try to improve by swapping primes or adjusting exponents.
    # This is a local search on the current best candidate.
    if check_time():
        # Try replacing one prime with another unused prime
        for p_remove in list(used_primes):
            if not check_time():
                break
            # Compute n without p_remove
            temp = current_n
            while temp % p_remove == 0:
                temp //= p_remove
            n_without = temp
            # Divisors of n_without
            # Rebuild divisors for n_without
            divs_without = [1]
            temp2 = n_without
            for p2 in primes:
                if temp2 % p2 == 0:
                    count = 0
                    while temp2 % p2 == 0:
                        temp2 //= p2
                        count += 1
                    cur_size = len(divs_without)
                    factor = 1
                    for _ in range(count):
                        factor *= p2
                        for i in range(cur_size):
                            divs_without.append(divs_without[i] * factor)
            divs_without.sort()
            score_without = compute_tau_perp(divs_without)

            # Try adding a different prime
            for p_add in primes:
                if p_add in used_primes and p_add != p_remove:
                    continue
                if p_add == p_remove:
                    continue
                if n_without * p_add <= max_val:
                    new_divs = merge_divisors(divs_without, p_add, 1)
                    new_score = compute_tau_perp(new_divs)
                    if new_score > current_score:
                        current_n = n_without * p_add
                        current_divs = new_divs
                        current_score = new_score
                        used_primes.discard(p_remove)
                        used_primes.add(p_add)
                        break

    # Phase 3: Try to increase exponents of existing primes if beneficial
    if check_time():
        for p in list(used_primes):
            if not check_time():
                break
            # Current exponent
            temp = current_n
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            # Try increasing by 1
            if current_n * p <= max_val:
                new_divs = merge_divisors(current_divs, p, 1)
                new_score = compute_tau_perp(new_divs)
                if new_score > current_score:
                    current_n *= p
                    current_divs = new_divs
                    current_score = new_score

    # Final verification
    if current_n < 2:
        current_n = 2

    return current_n
