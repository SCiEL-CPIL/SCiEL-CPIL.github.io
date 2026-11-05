import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    # Precompute 2D neighbor array for vectorized operations
    neighbors_2d = np.empty((N, 2 * n + 1), dtype=np.int32)
    for i in range(N):
        neighbors_2d[i] = neighbors_list[i]

    # Precompute neighbor-of-neighbor structure for 2-swap
    # For each pair of codewords (c1, c2), the union of their neighbor sets
    # This is expensive to precompute fully, so we'll compute on the fly

    def greedy_construct_random(seed=None, diversify=False):
        if seed is not None:
            random.seed(seed)

        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()

        # First pick: random if diversify, else 0
        if diversify:
            first = random.randint(0, N - 1)
        else:
            first = 0
        code.append(first)
        code_set.add(first)
        coverage[neighbors_list[first]] += 1
        uncovered_mask[neighbors_list[first]] = False

        while uncovered_mask.any():
            gains = uncovered_mask[neighbors_2d].sum(axis=1)

            for c in code:
                gains[c] = 0

            max_gain = int(gains.max())
            if max_gain == 0:
                break

            best_candidates = np.where(gains == max_gain)[0]

            if len(best_candidates) > 1:
                # Secondary criterion: prefer candidates with lower overlap
                overlaps = np.zeros(len(best_candidates), dtype=np.int32)
                for i, c in enumerate(best_candidates):
                    overlaps[i] = np.sum(coverage[neighbors_list[c]] > 1)

                min_overlap = overlaps.min()
                low_overlap_candidates = best_candidates[overlaps == min_overlap]

                # More randomness in tie-breaking
                if len(low_overlap_candidates) > 1:
                    if diversify:
                        # Pick from a random subset
                        k = min(len(low_overlap_candidates), random.randint(1, 5))
                        chosen_subset = random.sample(list(low_overlap_candidates), k)
                        chosen = int(random.choice(chosen_subset))
                    else:
                        chosen = int(random.choice(low_overlap_candidates))
                else:
                    chosen = int(low_overlap_candidates[0])
            else:
                chosen = int(best_candidates[0])

            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors_list[chosen]] += 1
            uncovered_mask[neighbors_list[chosen]] = False

        return code, code_set, coverage, uncovered_mask

    # Run multiple greedy constructions
    best_initial_code = None
    best_initial_size = float('inf')
    best_initial_set = None
    best_initial_cov = None
    best_initial_um = None

    num_restarts = 30

    for i in range(num_restarts):
        if time.time() > deadline - 35:
            break
        diversify = (i >= 10)  # More diversification in later restarts
        c, cs, cov, um = greedy_construct_random(seed=i, diversify=diversify)
        if len(c) < best_initial_size:
            best_initial_size = len(c)
            best_initial_code = c[:]
            best_initial_set = cs.copy()
            best_initial_cov = cov.copy()
            best_initial_um = um.copy()

    if best_initial_code is None:
        c, cs, cov, um = greedy_construct_random(seed=0, diversify=False)
        best_initial_code = c
        best_initial_set = cs
        best_initial_cov = cov
        best_initial_um = um

    code = best_initial_code
    code_set = best_initial_set
    coverage = best_initial_cov
    uncovered_mask = best_initial_um

    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            best_c = -1
            best_count = 0

            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])

            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break

            if best_c == -1:
                break

            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])

        return added

    def apply_code_change(removed_list, added_list):
        nonlocal code, code_set, coverage
        for c in removed_list:
            if c in code_set:
                code.remove(c)
                code_set.remove(c)
                coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_remove():
        nonlocal code, code_set, coverage
        removed = False
        for c in list(code):
            if is_redundant(c):
                apply_code_change([c], [])
                removed = True
        return removed

    def kick():
        nonlocal code, code_set, coverage
        if not code:
            return False

        old_size = len(code)

        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            apply_code_change([c_remove], [])
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            apply_code_change([c_remove], added)
            return True
        else:
            return False

    def kick_2_improved():
        nonlocal code, code_set, coverage
        if len(code) < 2:
            return False

        old_size = len(code)

        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)

        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) > 0:
            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)

            new_size = old_size - 2 + len(added)

            if new_size < old_size:
                apply_code_change([c1, c2], added)
                return True
        else:
            apply_code_change([c1, c2], [])
            return True

        return False

    def swap_2_move():
        nonlocal code, code_set, coverage
        if len(code) < 3:
            return False

        old_size = len(code)

        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)

        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            apply_code_change([c1, c2], [])
            return True

        uncovered_set = set(uncovered.tolist())

        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])

        best_c = -1
        best_count = 0

        for c in candidate_pool:
            if c in code_set:
                continue
            cnt = len(neighbors_set[c] & uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_c = c

        if best_c != -1 and best_count == len(uncovered_set):
            apply_code_change([c1, c2], [best_c])
            return True

        return False

    def swap_move():
        nonlocal code, code_set, coverage
        if len(code) < 2:
            return False

        old_size = len(code)

        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            return False

        uncovered_set = set(uncovered.tolist())

        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbors_set[w])

        best_c = -1
        best_count = 0

        for c in candidate_pool:
            if c in code_set:
                continue
            cnt = len(neighbors_set[c] & uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_c = c

        if best_c != -1 and best_count == len(uncovered_set):
            apply_code_change([c_remove], [best_c])
            return True

        return False

    # ILS-style perturbation: remove a random subset and repair
    def ils_perturb_and_repair():
        nonlocal code, code_set, coverage
        if len(code) < 5:
            return False

        old_size = len(code)

        # Remove 2-3 random codewords
        num_remove = random.randint(2, 3)
        to_remove = random.sample(list(code), num_remove)

        temp_coverage = coverage.copy()
        for c in to_remove:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) > 0:
            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)

            new_size = old_size - num_remove + len(added)

            if new_size < old_size:
                apply_code_change(to_remove, added)
                return True
        else:
            apply_code_change(to_remove, [])
            return True

        return False

    # Main Loop
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()

        # 2. Kick (single removal)
        if time.time() > deadline:
            break

        kicks_tried = 0
        success = False
        max_kicks = 40
        while kicks_tried < max_kicks:
            if kick():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break

        # 3. 2-removal kick
        if time.time() > deadline:
            break
        if not success and len(code) > 2:
            kick_2_improved()

        # 4. Swap 2-for-1 move
        if time.time() > deadline:
            break
        if len(code) > 3:
            swap_2_move()

        # 5. ILS perturbation
        if time.time() > deadline:
            break
        if not success:
            ils_perturb_and_repair()

        # 6. Swap moves to escape local optima
        if time.time() < deadline:
            swap_tried = 0
            max_swaps = 15
            while swap_tried < max_swaps:
                if swap_move():
                    changed = True
                    while changed:
                        if time.time() > deadline:
                            break
                        changed = try_remove()
                    swap_tried = 0
                else:
                    swap_tried += 1
                if time.time() > deadline:
                    break

        # 7. Final polishing: systematic single-removal check
        if time.time() > deadline - 5:
            break
        for c in list(code):
            if is_redundant(c):
                apply_code_change([c], [])
            if time.time() > deadline:
                break

    result = [idx_to_tuple(c) for c in code]
    return result
