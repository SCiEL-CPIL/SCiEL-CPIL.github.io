"""Rewritten program for equal-lcm-triple avoidance on [1,128]."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a // gcd(a, b) * b

def _v2(n):
    c = 0
    while n % 2 == 0:
        c += 1
        n //= 2
    return c

def _validate_set(S):
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            lab = _lcm(a, b)
            for k in range(j + 1, n):
                c = S[k]
                if lab == _lcm(a, c) == _lcm(b, c):
                    return False
    return True

def _can_add(S_list, x, lcm_cache):
    for i in range(len(S_list)):
        a = S_list[i]
        key_ax = (a, x) if a < x else (x, a)
        lab = lcm_cache[key_ax]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            key_ab = (a, b) if a < b else (b, a)
            key_bx = (b, x) if b < x else (x, b)
            if lab == lcm_cache[key_ab] == lcm_cache[key_bx]:
                return False
    return True

def _greedy_order(evens, lcm_cache, odds_set, deadline):
    """Greedy add evens to odds base."""
    S_list = list(odds_set)
    S_set = set(S_list)
    for x in evens:
        if time.time() > deadline:
            break
        if _can_add(S_list, x, lcm_cache):
            S_list.append(x)
            S_set.add(x)
    return S_list

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    N = 128
    all_nums = list(range(1, N + 1))
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]

    # Precompute LCM cache
    lcm_cache = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_cache[(a, b)] = _lcm(a, b)

    # Try multiple greedy orderings
    best_list = list(odds)
    best_size = len(odds)

    orderings = [
        sorted(evens, key=lambda x: (_v2(x), x)),
        sorted(evens, key=lambda x: (x,)),
        sorted(evens, key=lambda x: (-_v2(x), x)),
        sorted(evens, key=lambda x: (x % 4, _v2(x), x)),
        sorted(evens, key=lambda x: (x % 8, _v2(x), x)),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        S_list = _greedy_order(ordering, lcm_cache, set(odds), deadline)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_list = list(S_list)

    # Local search: 1-swap improvement
    def local_search_1swap(current_list, current_set, deadline):
        nonlocal best_size, best_list
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return current_list, current_set
            cur_size = len(current_list)
            for i in range(cur_size):
                if time.time() > deadline:
                    return current_list, current_set
                a = current_list[i]
                temp_list = [y for j, y in enumerate(current_list) if j != i]
                for x in all_nums:
                    if x in current_set:
                        continue
                    # Check if x can be added to temp_list
                    ok = _can_add(temp_list, x, lcm_cache)
                    if ok:
                        new_list = temp_list + [x]
                        new_set = current_set - {a} | {x}
                        if len(new_list) > best_size:
                            best_size = len(new_list)
                            best_list = list(new_list)
                        current_list = new_list
                        current_set = new_set
                        improved = True
                        break
                if improved:
                    break
        return current_list, current_set

    S_list = best_list
    S_set = set(S_list)
    S_list, S_set = local_search_1swap(S_list, S_set, deadline)

    # Try 2-swap if time permits
    if time.time() < deadline - 10:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            cur_size = len(S_list)
            for i in range(cur_size):
                if time.time() > deadline:
                    break
                a = S_list[i]
                for j in range(i + 1, cur_size):
                    if time.time() > deadline:
                        break
                    b = S_list[j]
                    temp_list = [z for k, z in enumerate(S_list) if k != i and k != j]
                    for x in all_nums:
                        if x in S_set:
                            continue
                        if not _can_add(temp_list, x, lcm_cache):
                            continue
                        temp2_list = temp_list + [x]
                        temp2_set = S_set - {a, b} | {x}
                        for y in all_nums:
                            if y in temp2_set or y == x:
                                continue
                            if _can_add(temp2_list, y, lcm_cache):
                                new_list = temp2_list + [y]
                                new_set = temp2_set | {y}
                                if len(new_list) > best_size:
                                    best_size = len(new_list)
                                    best_list = list(new_list)
                                S_list = new_list
                                S_set = new_set
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final validation
    result = best_list
    if not _validate_set(result):
        result = list(odds)
    return result
# EVOLVE-BLOCK-END
