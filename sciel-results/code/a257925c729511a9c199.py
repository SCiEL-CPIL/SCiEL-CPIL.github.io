"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82.0  # safety margin before 90s limit
    TARGET = 123457

    # Pre-computed primes up to a reasonable bound
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_all = sieve(2000)

    def largest_prime_factor(n):
        """Compute the largest prime factor of n via trial division."""
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp

    def generate_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are all in primes.
        Returns a dict: number -> largest prime factor (among the primes used).
        Uses iterative multiplication approach.
        """
        if not primes:
            return {1: 1}
        
        smooth = {1: 1}
        for p in primes:
            new_smooth = {}
            # Copy existing entries
            for x, lpf in smooth.items():
                new_smooth[x] = lpf
            # Multiply existing by powers of p
            for x, lpf in list(smooth.items()):
                val = x
                while val * p <= limit:
                    val *= p
                    new_lpf = max(lpf, p)
                    if val not in new_smooth or new_smooth[val] < new_lpf:
                        new_smooth[val] = new_lpf
            smooth = new_smooth
        return smooth

    # Initial best: [1, TARGET-1]
    best_result = [1, TARGET - 1]
    best_score = largest_prime_factor(TARGET - 1)  # P(123456)

    # Try increasing smoothness bounds: use first k primes
    # We'll try k from 1 up to len(primes_all)
    for k in range(1, len(primes_all) + 1):
        if time.time() > deadline - 5:
            break

        primes_subset = primes_all[:k]
        max_p = primes_subset[-1]

        # If max_p is already >= best_score, we might still find a better pair
        # where both have smaller LPF, but the bound is the max prime allowed.
        # Actually, if max_p >= best_score, the smooth numbers could have LPF up to max_p.
        # But we want to minimize max(P(a), P(b)).
        # If we restrict to primes <= max_p, then P(a) <= max_p and P(b) <= max_p.
        # So the score is at most max_p.
        # If max_p < best_score, then any valid pair found here will improve best_score.
        # If max_p >= best_score, we might still find a pair with score < best_score?
        # No, because if a is max_p-smooth, P(a) <= max_p. But P(a) could be smaller.
        # However, we are iterating k in increasing order.
        # At step k, we allow primes up to primes_all[k-1].
        # The score of any pair found is max(P(a), P(b)) where P(a), P(b) <= primes_all[k-1].
        # So the score is at most primes_all[k-1].
        # If primes_all[k-1] < best_score, we might improve.
        # If primes_all[k-1] >= best_score, we might still find a pair with score < best_score
        # only if both a and b have LPF < best_score. But since we're allowing primes up to
        # primes_all[k-1] which is >= best_score, the smooth numbers could include those with
        # LPF = best_score or higher. But we're looking for min max(P(a),P(b)).
        # Actually, the key insight: if we find a pair at step k where both are smooth w.r.t
        # primes <= primes_all[k-1], then max(P(a),P(b)) <= primes_all[k-1].
        # Since we go in increasing k, the first time we find a pair, its score is <= primes_all[k-1].
        # But there might be a better pair at a later k? No, later k allows larger primes,
        # so the score could be larger. But we want to minimize, so we want the smallest possible
        # max(P(a), P(b)).
        #
        # Actually, the correct approach: for each k, generate smooth numbers with primes <= p_k.
        # Check if any two sum to TARGET. If so, the score is at most p_k.
        # Since we're going in increasing k, the first k where we find a pair gives us a score
        # of at most p_k. But is it the minimum? Not necessarily, because at a later k, we might
        # find a pair where both have smaller LPF. But that would have been found at an earlier k
        # if their LPFs were smaller. So the first k where we find a pair gives the optimal score
        # equal to the minimum p_k such that a pair exists.
        #
        # Wait, that's not quite right. Suppose at k=3 (primes 2,3,5), we find a pair with
        # P(a)=3, P(b)=5, so score=5. At k=4 (primes 2,3,5,7), we might find a pair with
        # P(a)=2, P(b)=7, score=7, which is worse. Or P(a)=3, P(b)=3, score=3, which is better.
        # But if P(a)=3 and P(b)=3, then both are 3-smooth, so they would have been found at k=2
        # (primes 2,3). So the first k where a pair is found gives the optimal score.
        #
        # Therefore, we can stop as soon as we find a pair!
        # But wait, we need to make sure we check all pairs at that k, not just the first one.
        # Because the first pair we find might have score p_k, but another pair at the same k
        # might have score < p_k. But since all smooth numbers at step k have LPF <= p_k,
        # the score is max(P(a), P(b)) <= p_k. If we find a pair, the score is at most p_k.
        # But could it be less? Yes. So we should check all pairs at step k and take the min.
        #
        # Actually, let's just check all pairs at each k and update best_score.
        # And we can stop early if best_score is very small (e.g., 2 or 3).

        smooth = generate_smooth_numbers(primes_subset, TARGET)
        
        if len(smooth) < 2:
            continue

        # Check all pairs
        found_at_this_level = False
        min_score_at_level = float('inf')
        best_pair_at_level = None

        for a, pa in smooth.items():
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth:
                pb = smooth[b]
                score = max(pa, pb)
                if score < min_score_at_level:
                    min_score_at_level = score
                    best_pair_at_level = (a, b)
                found_at_this_level = True

        if found_at_this_level:
            if min_score_at_level < best_score:
                best_score = min_score_at_level
                best_result = list(best_pair_at_level)
            
            # If we found a pair, the optimal score is at most max_p.
            # Since we're going in increasing k, and we found a pair, 
            # the minimum possible score is at most max_p.
            # But could there be a better score at a later k? 
            # No, because later k allows larger primes, so the score could only be larger or equal.
            # Wait, no. At later k, we allow more primes, so we might find a pair where
            # both numbers have smaller LPF. But if both have smaller LPF, they would have
            # been smooth at an earlier k. So the first k where we find a pair gives the
            # optimal score.
            #
            # Therefore, we can break as soon as we find a pair.
            break

    # If we didn't find a pair via smooth numbers, try a more targeted search
    if best_score == largest_prime_factor(TARGET - 1):
        # Try to find a better pair by checking specific candidates
        # For each possible largest prime factor p from 2 to best_score-1,
        # check if there exist a, b with a+b=TARGET, P(a)<=p, P(b)<=p.
        # This is equivalent to the smooth number search, but let's try a different approach.
        
        # Alternative: iterate over possible values of a and compute P(a), P(TARGET-a)
        # But this is too slow for all a.
        
        # Instead, let's try to find a pair where both are highly composite with small factors.
        # Try a = 2^i * 3^j * 5^k * ... for small exponents.
        
        # For now, just return the best we have.
        pass

    return best_result

# EVOLVE-BLOCK-END
