"""Combined program: precomputed incompatibility graph + diversified greedy + multi-swap local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    n = 256
    candidates = list(range(2, n + 1))
    n_total = len(candidates)

    # Precompute: for each pair (b,c), which a values divide b*c
    # divides_by_pair[b][c] = set of a in [2,256] such that a | b*c, a != b, a != c
    divides_by_pair = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for b in range(2, n + 1):
        for c in range(b + 1, n + 1):
            prod = b * c
            for a in range(2, n + 1):
                if a == b or a == c:
                    continue
                if prod % a == 0:
                    divides_by_pair[b][c].add(a)
                    divides_by_pair[c][b].add(a)

    # Precompute: for each (a, x), the set of y values such that a | x*y
    # forbidden_y[a][x] = {y in [2,256] : y != a, y != x, a | x*y}
    forbidden_y = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for a in range(2, n + 1):
        for x in range(2, n + 1):
            if a == x:
                continue
            prod = x  # we need a | x*y, so y must satisfy a | x*y
            # a | x*y means (a/gcd(a,x)) | y
            from math import gcd
            g = gcd(a, x)
            needed = a // g
            # y must be a multiple of needed, and y != a, y != x
            y = needed
            while y <= n:
                if y != a and y != x:
                    forbidden_y[a][x].add(y)
                y += needed

    def can_add_fast(candidate, current_list):
        """Fast check using precomputed tables."""
        n_cur = len(current_list)
        
        # Case 1: candidate ∤ b*c for all distinct b,c in current_list
        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                b = current_list[i]
                c = current_list[j]
                if candidate in divides_by_pair[b][c]:
                    return False
        
        # Case 2: for each a in current_list, a ∤ candidate * y for all y in current_list, y != a
        for i in range(n_cur):
            a = current_list[i]
            forb = forbidden_y[a][candidate]
            # Check if any y in current_list (y != a) is in forb
            for j in range(n_cur):
                if j == i:
                    continue
                if current_list[j] in forb:
                    return False
        return True

    def is_valid_set(s):
        """Verify the set satisfies the constraint."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Diversified greedy with multiple orderings
    best_set = set()
    best_size = 0

    random.seed(42)
    orderings = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        list(candidates),
    ]
    for _ in range(60):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline - 15:
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

    # Greedy fill: add any remaining candidates that fit
    best_list = list(best_set)
    not_in_best = [x for x in candidates if x not in best_set]
    improved = True
    while improved:
        improved = False
        still_not_in = []
        for c in not_in_best:
            if time.time() > deadline - 15:
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

    # 1-for-1 swap
    if time.time() < deadline - 20:
        best_list = list(best_set)
        for i in range(len(best_list)):
            if time.time() > deadline - 15:
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

    # 1-for-2 swap
    if time.time() < deadline - 15:
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

    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback: build valid set from scratch
        current = set()
        current_list = []
        for c in candidates:
            if can_add_fast(c, current_list):
                current.add(c)
                current_list.append(c)
        result = sorted(current)

    return [int(x) for x in result]
