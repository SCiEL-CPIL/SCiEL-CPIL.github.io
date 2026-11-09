import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    N = 512

    # Precompute prime factors for each number in 1..511
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors for all candidates
    pf = [set() for _ in range(N)]
    for i in range(1, N):
        pf[i] = prime_factors(i)

    # Precompute 1/(512-a) for each a
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (512.0 - a)

    def is_coprime(a, used_primes):
        """Check if a is coprime to all numbers whose prime factors are in used_primes."""
        return not (pf[a] & used_primes)

    def greedy_select(candidates, used_primes):
        """Greedy select from candidates, must be coprime to used_primes and each other."""
        selected = []
        new_used = set(used_primes)
        for a in candidates:
            if is_coprime(a, new_used):
                selected.append(a)
                new_used |= pf[a]
        return selected

    # Multiple greedy initializations with different orderings
    best_S = None
    best_score = -1.0

    orderings = [
        list(range(511, 0, -1)),  # descending (high weights first)
        list(range(1, 512)),       # ascending
        list(range(511, 0, -2)) + list(range(510, 0, -2)),  # odd first, then even
        list(range(510, 0, -2)) + list(range(511, 0, -2)),  # even first, then odd
        sorted(range(1, N), key=lambda x: -weights[x]),  # by weight descending
    ]

    for ordering in orderings:
        S = []
        used = set()
        for a in ordering:
            if is_coprime(a, used):
                S.append(a)
                used |= pf[a]
        s = sum(weights[a] for a in S)
        if s > best_score:
            best_score = s
            best_S = S[:]

    current_S = best_S[:]
    current_score = best_score
    current_used = set()
    for a in current_S:
        current_used |= pf[a]

    improved = True
    all_candidates = list(range(511, 0, -1))
    current_set = set(current_S)

    while improved and time.time() < deadline:
        improved = False

        # Try removing each element and re-greedy-filling
        for a in list(current_S):
            if time.time() >= deadline:
                break
            bg = [b for b in current_S if b != a]
            bg_used = set()
            for b in bg:
                bg_used |= pf[b]

            valid_candidates = [c for c in all_candidates
                               if c not in current_set and is_coprime(c, bg_used)]
            B = greedy_select(valid_candidates, bg_used)

            if B:
                new_score = current_score - weights[a] + sum(weights[b] for b in B)
                if new_score > current_score + 1e-9:
                    current_S = bg + B
                    current_score = new_score
                    current_used = bg_used
                    for b in B:
                        current_used |= pf[b]
                    current_set = set(current_S)
                    improved = True
                    break

        # Try adding a single element
        if not improved and time.time() < deadline:
            for c in all_candidates:
                if c in current_set:
                    continue
                if is_coprime(c, current_used):
                    new_score = current_score + weights[c]
                    if new_score > current_score + 1e-9:
                        current_S.append(c)
                        current_used |= pf[c]
                        current_score = new_score
                        current_set.add(c)
                        improved = True
                        break

        # Try swap: remove a, add c
        if not improved and time.time() < deadline:
            for a in list(current_S):
                if time.time() >= deadline:
                    break
                bg = [b for b in current_S if b != a]
                bg_used = set()
                for b in bg:
                    bg_used |= pf[b]
                for c in all_candidates:
                    if c in current_set:
                        continue
                    if is_coprime(c, bg_used):
                        new_score = current_score - weights[a] + weights[c]
                        if new_score > current_score + 1e-9:
                            current_S = bg + [c]
                            current_score = new_score
                            current_used = bg_used | pf[c]
                            current_set = set(current_S)
                            improved = True
                            break
                if improved:
                    break

    return current_S
