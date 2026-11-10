import math
import random
import time
from itertools import combinations
from fractions import Fraction

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Uses exact rational arithmetic via Fraction for correctness.
    Employs incremental subset-sum tracking for efficient greedy construction.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    def check_dissociated(indices):
        """Check if set of indices has all distinct reciprocal subset sums."""
        k = len(indices)
        if k == 0:
            return True
        vals = [Fraction(1, i + 1) for i in indices]
        seen = {Fraction(0)}
        current_sums = [Fraction(0)]
        for v in vals:
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)
        return True

    def check_can_add(existing_sums_set, existing_sums_list, new_val):
        """
        Check if adding new_val to a set with existing subset sums maintains
        all distinct sums. Returns (True, new_sums_set, new_sums_list) if valid,
        (False, None, None) otherwise.
        """
        new_sums_set = set(existing_sums_set)
        new_sums_list = list(existing_sums_list)
        for s in existing_sums_list:
            ns = s + new_val
            if ns in new_sums_set:
                return (False, None, None)
            new_sums_set.add(ns)
            new_sums_list.append(ns)
        return (True, new_sums_set, new_sums_list)

    best_set = None

    def greedy_build(order, use_fraction=False):
        """
        Build up a dissociated set by greedily adding elements in the given order.
        Uses incremental subset-sum tracking for efficiency.
        """
        nonlocal best_set
        if time.time() > deadline:
            return None

        if use_fraction:
            sums_set = {Fraction(0)}
            sums_list = [Fraction(0)]
            S = []
            for idx in order:
                if time.time() > deadline:
                    break
                v = Fraction(1, idx + 1)
                ok, ns_set, ns_list = check_can_add(sums_set, sums_list, v)
                if ok:
                    S.append(idx)
                    sums_set = ns_set
                    sums_list = ns_list
        else:
            # Use LCM-based integer arithmetic for speed
            # LCM of 1..22
            def lcm(a, b):
                return abs(a * b) // math.gcd(a, b)
            L = 1
            for i in range(1, 23):
                L = lcm(L, i)
            A = [L // (i + 1) for i in range(22)]
            
            sums_set = {0}
            sums_list = [0]
            S = []
            for idx in order:
                if time.time() > deadline:
                    break
                v = A[idx]
                ok, ns_set, ns_list = check_can_add(sums_set, sums_list, v)
                if ok:
                    S.append(idx)
                    sums_set = ns_set
                    sums_list = ns_list

        if S and (best_set is None or len(S) > len(best_set)):
            best_set = list(S)
        return S if S else None

    # Strategy 1: Greedy build with various orderings
    # Ascending
    greedy_build(list(range(22)))
    # Descending
    greedy_build(list(range(21, -1, -1)))
    # Primes first (higher independence)
    primes = [i for i in range(22) if i + 1 in (2, 3, 5, 7, 11, 13, 17, 19)]
    non_primes = [i for i in range(22) if i not in primes]
    greedy_build(primes + non_primes)
    # Random orderings
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(22))
        random.shuffle(order)
        greedy_build(order)

    # Strategy 2: Local search - try to improve current best by swapping elements
    if best_set and time.time() < deadline - 10:
        current = list(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing one element and adding a different one
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                for j in range(22):
                    if j in current:
                        continue
                    candidate = current[:i] + current[i+1:] + [j]
                    if check_dissociated(candidate):
                        current = candidate
                        improved = True
                        break
                if improved:
                    break

        if current and len(current) > len(best_set):
            best_set = list(current)

    # Strategy 3: Randomized greedy with restarts, trying to exceed current best
    if time.time() < deadline - 15:
        target_size = len(best_set) + 1 if best_set else 1
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            order = list(range(22))
            random.shuffle(order)
            result = greedy_build(order)
            if result and len(result) >= target_size:
                if best_set is None or len(result) > len(best_set):
                    best_set = list(result)
                    target_size = len(result) + 1

    # Final safety
    if not best_set:
        best_set = [0]

    return [i + 1 for i in sorted(best_set)]

# EVOLVE-BLOCK-END
