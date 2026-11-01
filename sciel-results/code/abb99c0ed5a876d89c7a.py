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
            for o in range(max_ov, -1, -1, ):
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

    # Precompute all valid permutation codes for quick reference
    all_perm_codes = set()
    for perm in permutations(range(1, n + 1)):
        code = int(np.array(perm, dtype=np.int64) @ base)
        all_perm_codes.add(code)

    def try_deletion_fast(arr, L_val, codes_arr, pcount, pos):
        """Fast deletion check: returns (can_delete, new_codes_set, old_codes_list)"""
        n_val = n
        w_start = max(0, pos - n_val + 1)
        w_end = min(pos, L_val - n_val)

        if w_start > w_end:
            return False, set(), []

        old_codes = codes_arr[w_start:w_end + 1]
        old_codes_list = old_codes.tolist()

        # Check if any old code has count 1 and isn't in new windows
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

        for oc in old_codes_list:
            if pcount.get(oc, 0) == 1:
                if oc not in new_codes_set:
                    return False, set(), []

        return True, new_codes_set, old_codes_list

    # Phase 1: Repeated deletion passes - batch deletions for speed
    while time.time() < deadline - 5.0:
        deleted_this_pass = False
        
        # Try deleting from right to left (more effective)
        for pos in range(L - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break

            can_del, new_codes_set, old_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)

            if can_del:
                for oc in old_codes_list:
                    perm_count[oc] -= 1
                    if perm_count[oc] == 0:
                        del perm_count[oc]

                for nc in new_codes_set:
                    perm_count[nc] = perm_count.get(nc, 0) + 1

                current = np.concatenate([current[:pos], current[pos + 1:]])
                L -= 1
                best_len = min(best_len, L)
                codes = compute_all_codes(current, L)
                deleted_this_pass = True
                continue

        if not deleted_this_pass:
            # Try left to right as second pass
            for pos in range(L):
                if time.time() > deadline - 5.0:
                    break

                can_del, new_codes_set, old_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)

                if can_del:
                    for oc in old_codes_list:
                        perm_count[oc] -= 1
                        if perm_count[oc] == 0:
                            del perm_count[oc]

                    for nc in new_codes_set:
                        perm_count[nc] = perm_count.get(nc, 0) + 1

                    current = np.concatenate([current[:pos], current[pos + 1:]])
                    L -= 1
                    best_len = min(best_len, L)
                    codes = compute_all_codes(current, L)
                    deleted_this_pass = True
                    continue

        if not deleted_this_pass:
            break

    # Phase 2: 2-opt style moves with faster validation
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        rng = np.random.default_rng(42)
        
        # Try swap moves: remove one symbol and insert it elsewhere
        for _ in range(int(remaining_time / 0.5)):
            if time.time() > deadline - 2.0:
                break

            if L < n:
                break

            pos = int(rng.integers(0, L))
            symbol = current[pos]

            can_del, new_codes_set, old_codes_list = try_deletion_fast(current, L, codes, perm_count, pos)
            if not can_del:
                continue

            # Save state
            saved_current = current.copy()
            saved_L = L
            saved_codes = codes.copy()
            saved_perm_count = dict(perm_count)

            # Apply deletion
            for oc in old_codes_list:
                perm_count[oc] -= 1
                if perm_count[oc] == 0:
                    del perm_count[oc]

            for nc in new_codes_set:
                perm_count[nc] = perm_count.get(nc, 0) + 1

            current = np.concatenate([current[:pos], current[pos + 1:]])
            L -= 1
            codes = compute_all_codes(current, L)

            # Try inserting at random positions
            found_good_insert = False
            for _ in range(20):  # Try up to 20 random insert positions
                if time.time() > deadline - 2.0:
                    break

                insert_pos = int(rng.integers(0, L + 1))

                # Quick check: compute new windows around insert_pos
                new_string = np.concatenate([current[:insert_pos], np.array([symbol], dtype=np.int32), current[insert_pos:]])
                new_L = L + 1
                
                # Only check if this could be valid - check affected windows
                affected_start = max(0, insert_pos - n + 1)
                affected_end = min(insert_pos, new_L - n)
                
                if affected_start > affected_end:
                    continue
                    
                # Check if all affected windows are valid permutations
                valid = True
                for w in range(affected_start, affected_end + 1):
                    window = new_string[w:w + n]
                    if len(set(window.tolist())) != n:
                        valid = False
                        break
                
                if not valid:
                    continue

                # Full validation: check all perms still present
                new_codes_arr = compute_all_codes(new_string, new_L)
                new_pcount = compute_counts_from_codes(new_codes_arr)

                if len(new_pcount) >= target_perms and all(c in new_pcount for c in all_perm_codes):
                    current = new_string
                    L = new_L
                    best_len = min(best_len, L)
                    codes = new_codes_arr
                    perm_count = new_pcount
                    found_good_insert = True
                    break

            if not found_good_insert:
                # Restore state
                current = saved_current
                L = saved_L
                codes = saved_codes
                perm_count = saved_perm_count

    return "".join(map(str, current.tolist()))
