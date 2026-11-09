import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s kill

    n = 256
    candidates = list(range(2, n + 1))

    # Precompute factorizations
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

    # Precompute conflict counts for ordering
    conflict_counts = [0] * (n + 1)
    for a in range(2, n + 1):
        cnt = 0
        for b in range(2, n + 1):
            if b == a:
                continue
            g = gcd(a, b)
            m = a // g
            if m == 1:
                cnt += (n - 2)
            else:
                count_c = n // m
                if a % m == 0:
                    count_c -= 1
                if b % m == 0:
                    count_c -= 1
                if count_c < 0:
                    count_c = 0
                cnt += count_c
        conflict_counts[a] = cnt

    if time.time() > deadline - 20:
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    def can_add(candidate, current_list):
        k = len(current_list)
        for i in range(k):
            for j in range(i + 1, k):
                if (current_list[i] * current_list[j]) % candidate == 0:
                    return False
        for i in range(k):
            a = current_list[i]
            for j in range(k):
                if j == i:
                    continue
                if (candidate * current_list[j]) % a == 0:
                    return False
        return True

    best_set = set()
    best_size = 0

    sorted_by_conflict = sorted(range(2, n + 1), key=lambda x: (conflict_counts[x], x))
    sorted_desc = list(range(n, 1, -1))
    sorted_asc = list(range(2, n + 1))

    orderings = [sorted_by_conflict, sorted_desc, sorted_asc]

    import random
    random.seed(123)
    for _ in range(20):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline - 15:
            break
        current_list = []
        for c in ordering:
            if can_add(c, current_list):
                current_list.append(c)
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = set(current_list)

    best_list = list(best_set)
    improved = True
    while improved:
        improved = False
        for c in candidates:
            if time.time() > deadline - 10:
                break
            if c in best_set:
                continue
            if can_add(c, best_list):
                best_set.add(c)
                best_list.append(c)
                improved = True
                best_size = len(best_set)
        if time.time() > deadline - 10:
            break

    if time.time() < deadline - 12:
        best_list = list(best_set)
        for _ in range(3):
            improved = False
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
                    if can_add(c, test_list):
                        best_set = test_set | {c}
                        best_list = test_list + [c]
                        improved = True
                        break
                if time.time() > deadline - 10:
                    break
            if not improved:
                break

    if time.time() < deadline - 10:
        best_list = list(best_set)
        for _ in range(2):
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 8:
                    break
                elem_to_remove = best_list[i]
                test_set = set(best_set)
                test_set.discard(elem_to_remove)
                test_list = [x for x in best_list if x != elem_to_remove]
                candidates_to_add = [x for x in candidates if x not in test_set]
                added = []
                for c in candidates_to_add:
                    if time.time() > deadline - 8:
                        break
                    if len(added) >= 2:
                        break
                    if can_add(c, test_list + added):
                        added.append(c)
                if len(added) >= 2:
                    new_set = test_set | set(added)
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_size = len(best_set)
                        best_list = list(best_set)
                        improved = True
            if not improved:
                break

    if time.time() < deadline - 5:
        best_list = list(best_set)
        improved = True
        while improved:
            improved = False
            for c in candidates:
                if time.time() > deadline - 3:
                    break
                if c in best_set:
                    continue
                if can_add(c, best_list):
                    best_set.add(c)
                    best_list.append(c)
                    improved = True
                    best_size = len(best_set)
            if time.time() > deadline - 3:
                break

    result = sorted(best_set)
    valid = True
    for i in range(len(result)):
        a = result[i]
        for j in range(len(result)):
            b = result[j]
            if i == j:
                continue
            for k in range(len(result)):
                c = result[k]
                if i == k or j == k:
                    continue
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        current_list = []
        for c in sorted_by_conflict:
            if can_add(c, current_list):
                current_list.append(c)
        result = sorted(current_list)

    return [int(x) for x in result]
