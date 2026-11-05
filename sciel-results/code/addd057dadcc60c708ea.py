import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    def is_valid_set(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does NOT divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_full(S, x):
        """Full check: x can be added to S satisfying a<b<=c, a ∤ (b+c)."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        # Case 1: x is 'a'. Need x < b <= c for b, c in S.
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        # Case 2: x is 'b' or 'c'. For each a < x in S, check pairs (b,c) where b<=c and b+c divisible by a
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            # x acts as b: need a < x <= c, c in S
            for j in range(m):
                c = S_list[j]
                if c >= x:
                    if (x + c) % a == 0:
                        return False
            # x acts as c: need a < b <= x, b in S
            for j in range(m):
                b = S_list[j]
                if a < b <= x:
                    if (b + x) % a == 0:
                        return False
        return True

    def can_add_fast(S_set, S_sorted, x):
        """Faster check using precomputed sorted list."""
        if x in S_set:
            return False
        m = len(S_sorted)
        # Case 1: x is 'a'. Need x < b <= c for b, c in S.
        for i in range(m):
            b = S_sorted[i]
            if b > x:
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_sorted[j]
                    if (b + c) % x == 0:
                        return False
        # Case 2: x is 'b' or 'c'
        for i in range(m):
            a = S_sorted[i]
            if a >= x:
                break
            for j in range(m):
                c = S_sorted[j]
                if c >= x:
                    if (x + c) % a == 0:
                        return False
            for j in range(m):
                b = S_sorted[j]
                if a < b <= x:
                    if (b + x) % a == 0:
                        return False
        return True

    def local_search_swap(S, time_limit):
        """Try to improve S by swapping elements to increase size."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                S_sorted = sorted(S)
                # Try to add any element not in S
                for x in range(1, n + 1):
                    if x not in S:
                        if can_add_fast(S, S_sorted, x):
                            S.add(x)
                            improved = True
                            break
                if not improved:
                    S.add(a)  # put it back
        return S

    def greedy_extend(S, time_limit, order=None):
        """Greedily add elements to S in given order (or natural order)."""
        S = set(S)
        if order is None:
            order = list(range(1, n + 1))
        S_sorted = sorted(S)
        for x in order:
            if time.time() > time_limit:
                break
            if x not in S:
                if can_add_fast(S, S_sorted, x):
                    S.add(x)
                    S_sorted.append(x)
                    S_sorted.sort()
        return S

    best_size = 0
    best_S = set()

    # Strategy 1: Start with the known-good upper tail [171, 256] and extend downward
    if time.time() < deadline - 5.0:
        S = set(range(171, n + 1))
        # Try to add elements from 1 to 170 in various orders
        for attempt in range(10):
            if time.time() > deadline - 3.0:
                break
            rng = random.Random(attempt * 1000 + 42)
            candidates = list(range(1, 171))
            rng.shuffle(candidates)
            S_temp = set(S)
            S_temp = greedy_extend(S_temp, deadline - 3.0, order=candidates)
            # Local search
            if time.time() < deadline - 5.0:
                S_temp = local_search_swap(S_temp, deadline - 3.0)
            if len(S_temp) > best_size:
                best_size = len(S_temp)
                best_S = set(S_temp)

    # Strategy 2: Try different upper tail starting points
    for tail_start in [165, 170, 175, 180, 185, 190, 195, 200]:
        if time.time() > deadline - 5.0:
            break
        S = set(range(tail_start, n + 1))
        # Verify validity
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, sorted(valid_S), x):
                    valid_S.add(x)
            S = valid_S

        for attempt in range(5):
            if time.time() > deadline - 3.0:
                break
            rng = random.Random(tail_start * 1000 + attempt)
            candidates = list(range(1, tail_start))
            rng.shuffle(candidates)
            S_temp = set(S)
            S_temp = greedy_extend(S_temp, deadline - 3.0, order=candidates)
            if time.time() < deadline - 5.0:
                S_temp = local_search_swap(S_temp, deadline - 3.0)
            if len(S_temp) > best_size:
                best_size = len(S_temp)
                best_S = set(S_temp)

    # Strategy 3: Descending greedy from various starting points
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(restart * 2000 + 777)
        S = set()
        # Start with a random subset of upper elements
        upper_size = rng.randint(50, 80)
        upper_start = n - upper_size + 1
        S = set(range(upper_start, n + 1))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, sorted(valid_S), x):
                    valid_S.add(x)
            S = valid_S

        # Descending greedy
        S_temp = set(S)
        S_sorted = sorted(S_temp)
        for x in range(n - 1, 0, -1):
            if time.time() > deadline - 3.0:
                break
            if x not in S_temp:
                if can_add_fast(S_temp, S_sorted, x):
                    S_temp.add(x)
                    S_sorted.append(x)
                    S_sorted.sort()

        if time.time() < deadline - 5.0:
            S_temp = local_search_swap(S_temp, deadline - 3.0)

        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)

    # Strategy 4: Random restarts with greedy fill
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(restart * 3000 + 99)
        S = set(rng.sample(range(1, n + 1), rng.randint(30, 60)))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, sorted(valid_S), x):
                    valid_S.add(x)
            S = valid_S

        S_temp = set(S)
        candidates = [x for x in range(1, n + 1) if x not in S_temp]
        rng.shuffle(candidates)
        S_temp = greedy_extend(S_temp, deadline - 3.0, order=candidates)

        if time.time() < deadline - 5.0:
            S_temp = local_search_swap(S_temp, deadline - 3.0)

        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)

    # Strategy 5: Odd numbers base
    if time.time() < deadline - 5.0:
        S = set(range(1, n + 1, 2))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, sorted(valid_S), x):
                    valid_S.add(x)
            S = valid_S

        S_temp = set(S)
        candidates = list(range(2, n + 1, 2))
        rng = random.Random(12345)
        rng.shuffle(candidates)
        S_temp = greedy_extend(S_temp, deadline - 3.0, order=candidates)

        if time.time() < deadline - 5.0:
            S_temp = local_search_swap(S_temp, deadline - 3.0)

        if len(S_temp) > best_size:
            best_size = len(S_temp)
            best_S = set(S_temp)

    # Final polish: try to add any remaining elements to best_S
    if time.time() < deadline - 2.0 and best_size > 0:
        S_final = set(best_S)
        S_sorted = sorted(S_final)
        for x in range(1, n + 1):
            if time.time() > deadline - 1.0:
                break
            if x not in S_final:
                if can_add_fast(S_final, S_sorted, x):
                    S_final.add(x)
                    S_sorted.append(x)
                    S_sorted.sort()
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final

    if best_size == 0:
        return list(range(129, 257))

    if not is_valid_set(best_S):
        return sorted(set(range(171, 257)))

    return sorted(best_S)
