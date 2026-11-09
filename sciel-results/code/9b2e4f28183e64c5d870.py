import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 75  # Safety margin before 90s kill

    n = 256
    candidates = list(range(2, n + 1))

    # Precompute factorizations for all numbers 2..n
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors

    facs = {}
    for x in range(2, n + 1):
        facs[x] = factorize(x)

    # Precompute conflict counts: for each a, how many pairs (b,c) with b<c, b!=a, c!=a
    # satisfy a | b*c. This is used as a heuristic for ordering.
    # a | b*c iff for all primes p^k || a, v_p(b) + v_p(c) >= k
    # Equivalent: a/gcd(a,b) | c
    conflict_counts = [0] * (n + 1)
    for a in range(2, n + 1):
        cnt = 0
        for b in range(2, n + 1):
            if b == a:
                continue
            g = gcd(a, b)
            m = a // g  # need m | c
            if m == 1:
                # all c in [2,n] except a and b work
                cnt += (n - 1) - 1  # n-2
            else:
                # count multiples of m in [2, n]
                count_c = n // m
                # subtract if a is a multiple of m
                if a % m == 0:
                    count_c -= 1
                # subtract if b is a multiple of m
                if b % m == 0:
                    count_c -= 1
                if count_c < 0:
                    count_c = 0
                cnt += count_c
        conflict_counts[a] = cnt

    if time.time() > deadline:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    # Fast can_add function
    def can_add_fast(candidate, current_list):
        """Check if candidate can be added to current_list without violating constraints."""
        m = len(current_list)
        # Check 1: candidate does not divide b*c for any distinct b,c in current_list
        for i in range(m):
            for j in range(i + 1, m):
                if (current_list[i] * current_list[j]) % candidate == 0:
                    return False
        # Check 2: for each a in current_list, a does not divide candidate * x for any x != a
        for i in range(m):
            a = current_list[i]
            for j in range(m):
                if j == i:
                    continue
                if (candidate * current_list[j]) % a == 0:
                    return False
        return True

    def is_valid_set(s_list):
        """Verify that the set satisfies the constraint."""
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            for j in range(m):
                if j == i:
                    continue
                for k in range(j + 1, m):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Generate multiple orderings
    random.seed(42)
    orderings = [
        sorted(candidates, key=lambda x: (conflict_counts[x], x)),  # low conflict first
        sorted(candidates, key=lambda x: (-conflict_counts[x], x)),  # high conflict first
        sorted(candidates, reverse=True),  # descending
        sorted(candidates),  # ascending
    ]
    for _ in range(30):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    best_set = set()
    best_size = 0

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        current = set()
        current_list = []
        for c in ordering:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)

    # Local search: greedy fill
    best_list = list(best_set)
    not_in_best = [x for x in candidates if x not in best_set]
    improved = True
    while improved:
        improved = False
        still_not_in = []
        for c in not_in_best:
            if time.time() > deadline - 10:
                still_not_in.append(c)
                continue
            if can_add_fast(c, best_list):
                best_set.add(c)
                best_list.append(c)
                improved = True
                best_size = len(best_set)
            else:
                still_not_in.append(c)
        not_in_best = still_not_in
        if time.time() > deadline - 10:
            break

    # Local search: 1-for-1 swaps
    if time.time() < deadline - 15:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            for c in candidates:
                if c in test_set:
                    continue
                if can_add_fast(c, test_list):
                    best_set = test_set | {c}
                    best_list = test_list + [c]
                    break

    # Local search: 1-for-2 swaps (remove one, add two)
    if time.time() < deadline - 10:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            elem_to_remove = best_list[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            test_list = [x for x in best_list if x != elem_to_remove]
            candidates_to_add = [x for x in candidates if x not in test_set]
            added = []
            for c in candidates_to_add:
                if time.time() > deadline - 10:
                    break
                if len(added) >= 2:
                    break
                if can_add_fast(c, test_list + added):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
                    best_list = list(best_set)

    # Second round of greedy fill
    if time.time() < deadline - 10:
        best_list = list(best_set)
        not_in_best = [x for x in candidates if x not in best_set]
        improved = True
        while improved:
            improved = False
            still_not_in = []
            for c in not_in_best:
                if time.time() > deadline - 5:
                    still_not_in.append(c)
                    continue
                if can_add_fast(c, best_list):
                    best_set.add(c)
                    best_list.append(c)
                    improved = True
                    best_size = len(best_set)
                else:
                    still_not_in.append(c)
            not_in_best = still_not_in
            if time.time() > deadline - 5:
                break

    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback: rebuild from scratch
        current = set()
        current_list = []
        for c in candidates:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        result = sorted(current)

    # Ensure we return a list of Python ints
    return [int(x) for x in result]
