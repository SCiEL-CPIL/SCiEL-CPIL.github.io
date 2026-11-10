"""Optimized pairwise coprime selection with multi-swap local search and SA."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd, isqrt
from fractions import Fraction

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 145

    N = 512

    # Precompute values (as float for speed) and prime factors
    values_f = [0.0] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N

    for a in range(1, N):
        values_f[a] = 1.0 / (N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)

    # Build prime -> list of numbers that have this prime as factor
    prime_to_numbers = {}
    for a in range(1, N):
        for p in prime_factors[a]:
            if p not in prime_to_numbers:
                prime_to_numbers[p] = []
            prime_to_numbers[p].append(a)

    # Initialize with all primes (good starting point)
    selected = set()
    for a in range(1, N):
        if is_prime_arr[a]:
            selected.add(a)

    # Build conflict tracking: for each selected number, which primes it has
    selected_primes = {}  # prime -> set of selected numbers with this prime
    for b in selected:
        for p in prime_factors[b]:
            if p not in selected_primes:
                selected_primes[p] = set()
            selected_primes[p].add(b)

    def get_conflicts(candidate, selected_set):
        """Get set of selected numbers that conflict with candidate."""
        conflicts = set()
        for p in prime_factors[candidate]:
            if p in selected_primes:
                conflicts.update(selected_primes[p])
        conflicts.discard(candidate)
        return conflicts

    def apply_move(add_list, remove_list):
        for b in remove_list:
            for p in prime_factors[b]:
                if p in selected_primes:
                    selected_primes[p].discard(b)
                    if not selected_primes[p]:
                        del selected_primes[p]
        for c in add_list:
            for p in prime_factors[c]:
                if p not in selected_primes:
                    selected_primes[p] = set()
                selected_primes[p].add(c)
        selected.difference_update(remove_list)
        selected.update(add_list)

    def score_float():
        return sum(values_f[a] for a in selected)

    current_score = score_float()

    # Phase 1: Greedy 1-for-k with best-improvement
    # Repeatedly find the best single swap (add 1, remove k conflicting)
    improved = True
    while improved:
        if time.time() > deadline - 50:
            break
        improved = False
        best_gain = 0.0
        best_add = None
        best_remove = None

        for c in range(1, N):
            if c in selected:
                continue
            conflicts = get_conflicts(c, selected)
            conflict_cost = sum(values_f[b] for b in conflicts)
            gain = values_f[c] - conflict_cost
            if gain > best_gain:
                best_gain = gain
                best_add = [c]
                best_remove = list(conflicts)

        if best_add is not None:
            apply_move(best_add, best_remove)
            current_score += best_gain
            improved = True

    # Phase 2: 2-for-k best-improvement
    if time.time() < deadline - 35:
        # Pre-sort candidates by value for efficiency
        candidates = [a for a in range(1, N) if a not in selected and a > 1]
        candidates.sort(key=lambda x: values_f[x], reverse=True)
        top_cands = candidates[:60]

        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            best_gain = 0.0
            best_add = None
            best_remove = None

            for i in range(len(top_cands)):
                c1 = top_cands[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_cands)):
                    c2 = top_cands[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue

                    conflicts = set()
                    for c in (c1, c2):
                        for p in prime_factors[c]:
                            if p in selected_primes:
                                conflicts.update(selected_primes[p])
                    conflicts.discard(c1)
                    conflicts.discard(c2)

                    conflict_cost = sum(values_f[b] for b in conflicts)
                    gain = values_f[c1] + values_f[c2] - conflict_cost

                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflicts)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
                # Refresh candidates
                candidates = [a for a in range(1, N) if a not in selected and a > 1]
                candidates.sort(key=lambda x: values_f[x], reverse=True)
                top_cands = candidates[:60]

    # Phase 3: 3-for-k best-improvement (limited)
    if time.time() < deadline - 20:
        candidates = [a for a in range(1, N) if a not in selected and a > 1]
        candidates.sort(key=lambda x: values_f[x], reverse=True)
        top_cands = candidates[:40]

        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            best_gain = 0.0
            best_add = None
            best_remove = None

            for i in range(len(top_cands)):
                c1 = top_cands[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_cands)):
                    c2 = top_cands[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_cands)):
                        c3 = top_cands[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue

                        conflicts = set()
                        for c in (c1, c2, c3):
                            for p in prime_factors[c]:
                                if p in selected_primes:
                                    conflicts.update(selected_primes[p])
                        conflicts.discard(c1)
                        conflicts.discard(c2)
                        conflicts.discard(c3)

                        conflict_cost = sum(values_f[b] for b in conflicts)
                        gain = values_f[c1] + values_f[c2] + values_f[c3] - conflict_cost

                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflicts)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
                candidates = [a for a in range(1, N) if a not in selected and a > 1]
                candidates.sort(key=lambda x: values_f[x], reverse=True)
                top_cands = candidates[:40]

    # Phase 4: Simulated Annealing with multi-swap moves
    if time.time() < deadline - 5:
        T = 0.03
        T_min = 0.0005
        cooling = 0.997
        sa_deadline = deadline - 2

        # Precompute for faster SA
        all_numbers = list(range(1, N))

        while T > T_min and time.time() < sa_deadline:
            # Random move: pick 1-2 to add, remove conflicting
            move_type = random.random()
            add_list = []

            if move_type < 0.7:
                # Single add
                a_in = random.choice(all_numbers)
                if a_in not in selected:
                    add_list = [a_in]
            elif move_type < 0.9:
                # Double add
                a1 = random.choice(all_numbers)
                a2 = random.choice(all_numbers)
                if a1 != a2 and a1 not in selected and a2 not in selected and gcd(a1, a2) == 1:
                    add_list = [a1, a2]
            else:
                # Triple add
                a1 = random.choice(all_numbers)
                a2 = random.choice(all_numbers)
                a3 = random.choice(all_numbers)
                if len(set([a1, a2, a3])) == 3 and all(a not in selected for a in (a1, a2, a3)):
                    if gcd(a1, a2) == 1 and gcd(a1, a3) == 1 and gcd(a2, a3) == 1:
                        add_list = [a1, a2, a3]

            if not add_list:
                T *= cooling
                continue

            # Find conflicts
            conflicts = set()
            for c in add_list:
                for p in prime_factors[c]:
                    if p in selected_primes:
                        conflicts.update(selected_primes[p])
            conflicts.difference_update(add_list)

            old_score = current_score
            apply_move(add_list, list(conflicts))
            new_score = score_float()
            delta = new_score - old_score

            if delta > 0:
                current_score = new_score
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    apply_move(list(conflicts), add_list)
                    current_score = old_score

            T *= cooling

    # Final cleanup: 1-for-k best-improvement
    if time.time() < deadline - 3:
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            best_gain = 0.0
            best_add = None
            best_remove = None

            for c in range(1, N):
                if c in selected:
                    continue
                conflicts = get_conflicts(c, selected)
                conflict_cost = sum(values_f[b] for b in conflicts)
                gain = values_f[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_add = [c]
                    best_remove = list(conflicts)

            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True

    # Final verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")

    return result
# EVOLVE-BLOCK-END
