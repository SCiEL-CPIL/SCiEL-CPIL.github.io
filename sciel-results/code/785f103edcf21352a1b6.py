import time
from math import gcd
from fractions import Fraction
from functools import lru_cache

def construct():
    deadline = time.time() + 145  # safety margin under 150s kill

    N = 512
    # Precompute prime factors for each number 1..511
    # For fast coprime checking: two numbers are coprime iff they share no prime factor
    prime_factors = {}
    for a in range(1, N):
        n = a
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        prime_factors[a] = factors

    # Precompute conflict: conflict[a] is a set of prime factors of a
    # Two numbers a, b are coprime iff prime_factors[a] & prime_factors[b] == empty set

    def is_coprime_to_all(a, selected_factors_union):
        """Check if a is coprime to all numbers whose prime factors are in selected_factors_union."""
        return prime_factors[a].isdisjoint(selected_factors_union)

    def score_float(lst):
        return sum(1.0 / (N - a) for a in lst)

    def greedy(order, exclude=None):
        """Greedy selection in given order, optionally excluding a set of elements."""
        S = []
        union_factors = set()
        for a in order:
            if exclude is not None and a in exclude:
                continue
            if prime_factors[a].isdisjoint(union_factors):
                S.append(a)
                union_factors |= prime_factors[a]
        return S

    # Initial greedy in descending order
    best = greedy(range(511, 0, -1))
    best_score = score_float(best)

    # Local search phase 1: drop one chosen element, then re-greedy descending
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() >= deadline:
                break
            S2 = greedy(range(511, 0, -1), exclude={x})
            s2 = score_float(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    # Local search phase 2: try adding any element that fits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_set = set(best)
        union_factors = set()
        for a in best:
            union_factors |= prime_factors[a]
        for a in range(511, 0, -1):
            if a in best_set:
                continue
            if prime_factors[a].isdisjoint(union_factors):
                best.append(a)
                best_set.add(a)
                union_factors |= prime_factors[a]
                best_score += 1.0 / (N - a)
                improved = True
                break

    # Local search phase 3: swap-based - replace one element with another
    # Precompute factor unions for faster swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_set = set(best)
        # Precompute factor union for each element removed
        for i, x in enumerate(list(best)):
            if time.time() >= deadline:
                break
            # Compute union of all factors except x
            rest_union = set()
            for k, a in enumerate(best):
                if k != i:
                    rest_union |= prime_factors[a]
            # Try adding y not in best that is coprime to rest_union
            best_y = None
            best_gain = 0.0
            for y in range(511, 0, -1):
                if y in best_set:
                    continue
                if prime_factors[y].isdisjoint(rest_union):
                    gain = 1.0 / (N - y) - 1.0 / (N - x)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_y = y
            if best_y is not None:
                best = [a for k, a in enumerate(best) if k != i] + [best_y]
                best_score += best_gain
                best_set = set(best)
                improved = True
                break

    # Local search phase 4: more aggressive - drop two elements, re-greedy
    # Limit to most promising pairs (those with small weights)
    if time.time() < deadline:
        n = len(best)
        # Sort by weight (1/(512-a)) ascending - try removing low-weight elements first
        sorted_indices = sorted(range(n), key=lambda i: 1.0/(N - best[i]))
        pairs_tried = 0
        max_pairs = 50  # Limit number of pair removals to stay within time budget
        for ii in range(min(n, 10)):
            if time.time() >= deadline or pairs_tried >= max_pairs:
                break
            for jj in range(ii+1, min(n, 10)):
                if time.time() >= deadline or pairs_tried >= max_pairs:
                    break
                x, y = best[sorted_indices[ii]], best[sorted_indices[jj]]
                S2 = greedy(range(511, 0, -1), exclude={x, y})
                s2 = score_float(S2)
                pairs_tried += 1
                if s2 > best_score + 1e-12:
                    best = S2
                    best_score = s2
                    break
            if time.time() >= deadline:
                break

    # Final: try to add any remaining coprime elements
    best_set = set(best)
    union_factors = set()
    for a in best:
        union_factors |= prime_factors[a]
    for a in range(511, 0, -1):
        if a not in best_set and prime_factors[a].isdisjoint(union_factors):
            best.append(a)
            best_set.add(a)
            union_factors |= prime_factors[a]

    # Additional polish: try single-element swaps one more time
    if time.time() < deadline:
        for i, x in enumerate(list(best)):
            if time.time() >= deadline:
                break
            rest_union = set()
            for k, a in enumerate(best):
                if k != i:
                    rest_union |= prime_factors[a]
            best_y = None
            best_gain = 0.0
            for y in range(511, 0, -1):
                if y in best_set:
                    continue
                if prime_factors[y].isdisjoint(rest_union):
                    gain = 1.0 / (N - y) - 1.0 / (N - x)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_y = y
            if best_y is not None:
                best = [a for k, a in enumerate(best) if k != i] + [best_y]
                best_score += best_gain
                best_set = set(best)
                # Recursively check for more improvements
                break

    return best
