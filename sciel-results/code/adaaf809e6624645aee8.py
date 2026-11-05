import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    # Base set: 171..256 (86 elements). For a in 171..256, 3a > 512 >= max(b+c),
    # so no a divides b+c for any b,c > a in [1,256]. Internally conflict-free.
    base = list(range(171, 257))

    def is_valid_addition(S_list, x):
        """S_list sorted ascending, x not in S_list. Return True if x can be added."""
        m = len(S_list)
        # Binary search: lo = first index with S_list[lo] > x
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) // 2
            if S_list[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        # Elements S_list[lo:] are > x; S_list[:lo] are < x (since x not in S_list)
        # x as 'a': no pair (b,c) with x < b <= c and (b+c) % x == 0
        for i in range(lo, m):
            for j in range(i, m):
                if (S_list[i] + S_list[j]) % x == 0:
                    return False
        # x as 'b' (middle): for a < x and c > x: (x+c) % a == 0; also a | 2x
        for i in range(lo):
            a = S_list[i]
            if (2 * x) % a == 0:
                return False
            for j in range(lo, m):
                if (x + S_list[j]) % a == 0:
                    return False
        # x as 'c' (largest): for a < b < x: (b+x) % a == 0
        for i in range(lo):
            a = S_list[i]
            for j in range(i + 1, lo):
                if (S_list[j] + x) % a == 0:
                    return False
        return True

    def full_valid(S_list):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def try_order(S_start, order):
        S = list(S_start)
        S.sort()
        S_set = set(S)
        for x in order:
            if x in S_set:
                continue
            if is_valid_addition(S, x):
                S.append(x)
                S.sort()
                S_set.add(x)
        return S

    best_result = list(base)
    best_size = len(best_result)
    smaller = list(range(1, 171))

    # Deterministic orders
    for order in (smaller[::-1], smaller):
        S = try_order(base, order)
        if len(S) > best_size:
            best_size = len(S)
            best_result = S

    # Random restarts
    rng = random.Random(12345)
    while time.time() < deadline - 15.0:
        order = smaller[:]
        rng.shuffle(order)
        S = try_order(base, order)
        if len(S) > best_size:
            best_size = len(S)
            best_result = S

    # Local search: 1-swap
    def local_search(S_list, max_iter=200):
        S = list(S_list)
        S.sort()
        S_set = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 4.0:
            improved = False
            iters += 1
            m = len(S)
            for i in range(m):
                if time.time() > deadline - 4.0:
                    break
                removed = S[i]
                S_no = [x for x in S if x != removed]
                S_no.sort()
                S_no_set = set(S_no)
                for x in range(1, n + 1):
                    if x in S_no_set:
                        continue
                    if is_valid_addition(S_no, x):
                        S_new = S_no + [x]
                        S_new.sort()
                        S_new_set = set(S_new)
                        fill_order = [y for y in range(1, n + 1) if y not in S_new_set]
                        rng.shuffle(fill_order)
                        S_filled = S_new
                        for y in fill_order:
                            if time.time() > deadline - 4.0:
                                break
                            if y in S_new_set:
                                continue
                            if is_valid_addition(S_filled, y):
                                S_filled.append(y)
                                S_filled.sort()
                                S_new_set.add(y)
                        if len(S_filled) > len(S):
                            S = S_filled
                            S_set = set(S)
                            improved = True
                            break
                if improved:
                    break
        return S

    if time.time() < deadline - 6.0:
        S_ls = local_search(best_result, max_iter=200)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_result = S_ls

    best_result = sorted(set(best_result))
    if not full_valid(best_result) or not best_result:
        best_result = list(base)

    return best_result
