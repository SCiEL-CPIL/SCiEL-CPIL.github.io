"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd as _gcd

_MAX_DEN = 10000000


def _construct():
    """
    Find the minimum number of distinct denominators in [5, 10^7]
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 85.0

    # k=7 is impossible (max sum of 7 smallest: 1/5+...+1/11 < 1)
    # k=8: max sum of 8 smallest > 1, so feasible in principle
    for k in range(8, 257):
        if time.time() > deadline:
            break
        sol = _search_k(k, deadline)
        if sol is not None:
            return sol

    # Fallback: guaranteed valid representation
    return _fallback()


def _search_k(k, deadline):
    """
    Backtracking search for exactly k distinct denominators >= 5
    whose reciprocal sum is 1, using pure integer arithmetic with strong pruning.
    """
    chosen = []
    time_check_counter = [0]

    def _should_stop():
        time_check_counter[0] += 1
        if time_check_counter[0] % 1000 == 0:
            return time.time() > deadline
        return False

    def rec(remaining, last, num, den):
        """
        Find `remaining` distinct denominators > last such that
        their reciprocal sum equals num/den (reduced fraction).
        num/den must be in reduced form.
        """
        if _should_stop():
            return False

        if remaining == 0:
            return num == 0

        if remaining == 1:
            if num == 0:
                return False
            if den % num == 0:
                d = den // num
                if d > last and d >= 5 and d <= _MAX_DEN:
                    chosen.append(d)
                    return True
            return False

        # remaining >= 2
        # Next denominator d must satisfy:
        #   d > last, d >= 5
        #   1/d <= num/den (actually < for strict, but we need sum of remaining = num/den)
        #   So d > den/num
        #   Also, sum of `remaining` terms starting from d_low must be >= num/den

        d_low = max(last + 1, 5)
        if num > 0:
            # d must satisfy 1/d < num/den => d > den/num
            d_low = max(d_low, den // num + 1)

        # Upper bound on d: we need 1/d <= num/den => d >= den/num
        # But also, the remaining (remaining-1) terms after d must be able to
        # contribute at least 0, so 1/d <= num/den => d >= den/num
        # And we need enough room: (remaining-1) terms each at most 1/(d+1)
        # So we need num/den - 1/d >= 0, and (remaining-1)/(d+1) >= num/den - 1/d
        
        # d_high: from 1/d <= num/den => d >= den/num (already in d_low)
        # Also from the sum constraint: remaining/d_low >= num/den gives a lower bound on feasibility
        # For d_high: we need (remaining-1) terms to cover num/den - 1/d
        # Each of those terms is at most 1/(d+1), so (remaining-1)/(d+1) >= num/den - 1/d
        # This gives us an upper bound on d.
        
        # Simpler: d_high from the fact that 1/d must be <= num/den
        # Actually d_low already ensures 1/d < num/den
        # d_high: we need the remaining sum to be achievable
        # The minimum possible sum of (remaining-1) terms with denominators > d is (remaining-1)/(d+remaining-1) approx
        # But more precisely, max sum of (remaining-1) terms > d is sum_{i=1}^{remaining-1} 1/(d+i)
        # For pruning, use: (remaining-1)/(d+1) as upper bound for remaining terms
        
        # Let's compute d_high from: 1/d + (remaining-1)/(d+1) >= num/den
        # This is a necessary condition for feasibility
        # d_high is where this becomes false
        
        # Actually, let's just use a generous d_high and rely on the inner pruning
        d_high = min(_MAX_DEN, (remaining * den) // num) if num > 0 else _MAX_DEN

        if d_low > d_high:
            return False

        # Pruning: max possible sum of `remaining` terms with denominators >= d_low
        # is sum_{i=0}^{remaining-1} 1/(d_low+i)
        # Approximate: remaining/d_low - remaining*(remaining-1)/(2*d_low^2)
        # For safety, use remaining/d_low as upper bound (loose but fast)
        if remaining * den < num * d_low:
            return False

        for d in range(d_low, d_high + 1):
            if _should_stop():
                return False

            # 1/d must be strictly less than num/den (since we need positive remainder)
            # i.e., den < num * d
            if not (den < num * d):
                continue

            new_num = num * d - den
            new_den = den * d

            if new_num <= 0:
                continue

            g = _gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            # Pruning: (remaining-1) terms each <= 1/(d+1)
            # Need (remaining-1)/(d+1) >= new_num/new_den
            # i.e., (remaining-1) * new_den >= new_num * (d+1)
            if (remaining - 1) * new_den < new_num * (d + 1):
                break

            # Additional pruning: minimum sum of (remaining-1) terms > d
            # is approximately (remaining-1)/(d+remaining-1)
            # Need new_num/new_den <= (remaining-1)/d approximately
            # Actually the min sum is sum_{i=1}^{remaining-1} 1/(d+i) >= (remaining-1)/(d+remaining-1)
            # So we need new_num/new_den <= (remaining-1)/(d+1) (already checked above as upper bound)
            # And new_num/new_den >= (remaining-1)/(d+remaining-1) (lower bound check)
            # Skip lower bound for speed

            chosen.append(d)
            if rec(remaining - 1, d, new_num, new_den):
                return True
            chosen.pop()

        return False

    if rec(k, 4, 1, 1):
        return chosen[:]
    return None


def _fallback():
    """
    Guaranteed-valid baseline using subset DP over divisors of 2520.
    """
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")


def construct():
    return _construct()
# EVOLVE-BLOCK-END
