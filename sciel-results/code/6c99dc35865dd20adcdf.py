import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    # Precompute all 4-term arithmetic progressions
    aps_containing = [[] for _ in range(N + 1)]

    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    def make_bool_array(S):
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        return arr

    def is_valid_add(arr, x):
        for (y, z, w) in aps_containing[x]:
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def is_valid_swap(arr, i, j):
        for (y, z, w) in aps_containing[j]:
            if i == y or i == z or i == w:
                continue
            if arr[y] and arr[z] and arr[w]:
                return False
        return True

    def is_valid_double_swap(arr, i1, i2, j1, j2):
        """Check if removing i1,i2 and adding j1,j2 keeps S valid."""
        arr[i1] = False
        arr[i2] = False
        ok1 = True
        for (y, z, w) in aps_containing[j1]:
            if arr[y] and arr[z] and arr[w]:
                ok1 = False
                break
        if ok1:
            ok2 = True
            for (y, z, w) in aps_containing[j2]:
                if arr[y] and arr[z] and arr[w]:
                    ok2 = False
                    break
            valid = ok2
        else:
            valid = False
        arr[i1] = True
        arr[i2] = True
        return valid

    def local_search(S):
        S = set(S)
        arr = make_bool_array(S)

        while time.time() - start_time < time_limit:
            # Try to add elements
            candidates = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates)

            added = False
            for x in candidates:
                if is_valid_add(arr, x):
                    S.add(x)
                    arr[x] = True
                    added = True
                    break

            if added:
                continue

            # If no element could be added, try 1-exchange swaps
            swapped = False
            elements_in_S = [x for x in range(1, N + 1) if arr[x]]
            random.shuffle(elements_in_S)
            candidates_out = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates_out)

            for i in elements_in_S:
                if time.time() - start_time > time_limit:
                    break
                for j in candidates_out:
                    if is_valid_swap(arr, i, j):
                        S.remove(i)
                        arr[i] = False
                        S.add(j)
                        arr[j] = True
                        swapped = True
                        break
                if swapped:
                    break

            if swapped:
                continue

            # Try 2-exchange: remove 2 from S, add 2 not in S
            double_swapped = False
            elements_in_S2 = [x for x in range(1, N + 1) if arr[x]]
            random.shuffle(elements_in_S2)
            candidates_out2 = [x for x in range(1, N + 1) if not arr[x]]
            random.shuffle(candidates_out2)

            # Limit the search space for 2-exchange to keep it fast
            max_i1 = min(len(elements_in_S2), 35)
            max_i2_window = 8
            max_j1 = min(len(candidates_out2), 35)
            max_j2_window = 8

            for idx_i1 in range(max_i1):
                if time.time() - start_time > time_limit:
                    break
                i1 = elements_in_S2[idx_i1]
                for idx_i2 in range(idx_i1 + 1, min(idx_i1 + max_i2_window, len(elements_in_S2))):
                    i2 = elements_in_S2[idx_i2]
                    for idx_j1 in range(max_j1):
                        if time.time() - start_time > time_limit:
                            break
                        j1 = candidates_out2[idx_j1]
                        for idx_j2 in range(idx_j1 + 1, min(idx_j1 + max_j2_window, len(candidates_out2))):
                            j2 = candidates_out2[idx_j2]
                            if is_valid_double_swap(arr, i1, i2, j1, j2):
                                S.discard(i1)
                                S.discard(i2)
                                arr[i1] = False
                                arr[i2] = False
                                S.add(j1)
                                S.add(j2)
                                arr[j1] = True
                                arr[j2] = True
                                double_swapped = True
                                break
                        if double_swapped:
                            break
                    if double_swapped:
                        break
                if double_swapped:
                    break

            if not double_swapped:
                break

        return S

    # Strategy: Multiple restarts with randomization
    best_S = set()

    # Start 1: Greedy (sorted order)
    S_curr = set()
    arr_curr = [False] * (N + 1)
    for x in range(1, N + 1):
        if is_valid_add(arr_curr, x):
            S_curr.add(x)
            arr_curr[x] = True
    best_S = local_search(S_curr)

    # Start 2: Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 3: Another Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 4: Greedy with different randomization
    if time.time() - start_time < time_limit - 5:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 5: If time remains, try another
    if time.time() - start_time < time_limit - 5:
        S_curr = set()
        arr_curr = [False] * (N + 1)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(arr_curr, x):
                S_curr.add(x)
                arr_curr[x] = True
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    return sorted(best_S)
