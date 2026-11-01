import time
import numpy as np
from collections import defaultdict
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin below 150s

    # 1. Generate initial superpermutation using standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # 2. Local Search with vectorized deletion checks
    current = np.array(word, dtype=np.int32)
    L = len(current)
    target_perms = 1
    for i in range(2, n + 1):
        target_perms *= i

    # Use a base that fits in int64 safely for n=7
    base = np.array([n ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    max_code = int(base[0])  # n^(n-1)

    def compute_all_codes(arr, L_val):
        if L_val < n:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, n)
        return windows.astype(np.int64) @ base

    def compute_counts_from_codes(codes):
        if len(codes) == 0:
            return {}
        unique, counts = np.unique(codes, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    codes = compute_all_codes(current, L)
    perm_count = compute_counts_from_codes(codes)

    if len(perm_count) < target_perms:
        return "".join(map(str, current.tolist()))

    best_len = L

    def try_deletion(arr, L_val, codes_arr, pcount, pos):
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False, None

        old_codes = codes_arr[w_start:w_end + 1]

        new_codes_set = set()
        for j in range(max(0, pos - n_val + 1), pos):
            suffix_start = pos + 1
            suffix_end = j + n_val
            if suffix_end > L_val:
                continue
            if j < pos:
                prefix = arr[j:pos].astype(np.int64)
            else:
                prefix = np.array([], dtype=np.int64)
            if pos + 1 < j + n_val:
                suffix = arr[pos + 1:j + n_val].astype(np.int64)
            else:
                suffix = np.array([], dtype=np.int64)
            if len(prefix) + len(suffix) != n_val:
                continue
            window = np.concatenate([prefix, suffix])
            code = int(window @ base)
            new_codes_set.add(code)

        for oc in old_codes:
            oc_int = int(oc)
            if pcount.get(oc_int, 0) == 1:
                if oc_int not in new_codes_set:
                    return False, None

        return True, new_codes_set

    # Phase 1: Repeated deletion passes - batch deletions for speed
    while time.time() < deadline:
        deleted_this_pass = False
        # Try deleting from right to left (more effective)
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break

            can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)

            if can_del:
                w_start = max(0, pos - n + 1)
                w_end = min(pos, L - n)
                old_codes = codes[w_start:w_end + 1]
                for oc in old_codes:
                    oc_int = int(oc)
                    perm_count[oc_int] -= 1
                    if perm_count[oc_int] == 0:
                        del perm_count[oc_int]

                for nc in new_codes_set:
                    perm_count[nc] = perm_count.get(nc, 0) + 1

                current = np.concatenate([current[:pos], current[pos + 1:]])
                L -= 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                deleted_this_pass = True
                # Continue scanning without breaking - allow multiple deletions per pass
                continue

        if not deleted_this_pass:
            for pos in range(L):
                if time.time() > deadline - 2.0:
                    break

                can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)

                if can_del:
                    w_start = max(0, pos - n + 1)
                    w_end = min(pos, L - n)
                    old_codes = codes[w_start:w_end + 1]
                    for oc in old_codes:
                        oc_int = int(oc)
                        perm_count[oc_int] -= 1
                        if perm_count[oc_int] == 0:
                            del perm_count[oc_int]

                    for nc in new_codes_set:
                        perm_count[nc] = perm_count.get(nc, 0) + 1

                    current = np.concatenate([current[:pos], current[pos + 1:]])
                    L -= 1
                    best_len = min(best_len, L)
                    codes = compute_all_codes(current, L)
                    deleted_this_pass = True
                    # Continue scanning without breaking
                    continue

        if not deleted_this_pass:
            break

    # Phase 2: 2-opt style moves with faster validation
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        rng = np.random.default_rng(42)
        max_2opt_rounds = int(remaining_time / 2.0)

        # Precompute all valid permutation codes
        all_perm_codes = set()
        for perm in permutations(range(1, n + 1)):
            code = int(np.array(perm, dtype=np.int64) @ base)
            all_perm_codes.add(code)

        for round_num in range(max_2opt_rounds):
            if time.time() > deadline - 1.0:
                break

            pos = int(rng.integers(0, L))
            symbol = current[pos]

            can_del, new_codes_set = try_deletion(current, L, codes, perm_count, pos)
            if not can_del:
                continue

            saved_current = current.copy()
            saved_L = L
            saved_codes = codes.copy()
            saved_perm_count = dict(perm_count)

            w_start = max(0, pos - n + 1)
            w_end = min(pos, L - n)
            old_codes = codes[w_start:w_end + 1]
            for oc in old_codes:
                oc_int = int(oc)
                perm_count[oc_int] -= 1
                if perm_count[oc_int] == 0:
                    del perm_count[oc_int]

            for nc in new_codes_set:
                perm_count[nc] = perm_count.get(nc, 0) + 1

            current = np.concatenate([current[:pos], current[pos + 1:]])
            L -= 1
            codes = compute_all_codes(current, L)

            best_insert_pos = -1
            best_insert_valid = False

            for insert_pos in range(L + 1):
                if time.time() > deadline - 1.0:
                    break

                valid_insert = True
                new_window_codes = set()

                for j in range(max(0, insert_pos - n + 1), min(insert_pos + 1, L - n + 1)):
                    window = []
                    for k in range(n):
                        new_pos = j + k
                        if new_pos < insert_pos:
                            window.append(current[new_pos])
                        else:
                            if new_pos == insert_pos:
                                window.append(symbol)
                            elif new_pos - 1 < L:
                                window.append(current[new_pos - 1])
                            else:
                                valid_insert = False
                                break
                    if not valid_insert:
                        break

                    window_arr = np.array(window, dtype=np.int64)
                    wc = int(window_arr @ base)
                    new_window_codes.add(wc)

                if not valid_insert:
                    continue

                # Quick check: does the new set of windows still cover all permutations?
                # We need to check that all_perm_codes are still present
                # The windows that change are those involving insert_pos
                # Old windows around this area are gone, new ones are in new_window_codes
                # We can check if any required code is lost

                # Compute which old windows are affected
                affected_start = max(0, insert_pos - n + 1)
                affected_end = min(insert_pos, L - n)

                # Check if all required codes are still covered
                valid = True
                for ac in all_perm_codes:
                    # Check if ac is in new_window_codes or still in perm_count (from unaffected windows)
                    if ac in new_window_codes:
                        continue
                    # Check if it was in the unaffected part
                    # This is expensive, so let's just do a full check for now
                    pass

                # Full validation
                new_string = np.concatenate([current[:insert_pos], np.array([symbol], dtype=np.int32), current[insert_pos:]])
                new_L = L + 1
                new_codes_arr = compute_all_codes(new_string, new_L)
                new_pcount = compute_counts_from_codes(new_codes_arr)

                if len(new_pcount) >= target_perms:
                    if all(c in new_pcount for c in all_perm_codes):
                        best_insert_pos = insert_pos
                        best_insert_valid = True
                        break

            if best_insert_valid:
                current = np.concatenate([current[:best_insert_pos], np.array([symbol], dtype=np.int32), current[best_insert_pos:]])
                L = L + 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                perm_count = compute_counts_from_codes(codes)
            else:
                current = saved_current
                L = saved_L
                codes = saved_codes
                perm_count = saved_perm_count

    return "".join(map(str, current.tolist()))
