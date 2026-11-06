"""Optimized small maximal Sidon set constructor for [1,256]."""
import time
import random

N = 256
MAX_SUM = 2 * N  # 512
# We need a bit for each sum value 2..512, index sum-2
NUM_SUMS = MAX_SUM - 1  # sums range from 2 to 512, that's 511 values

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin

    best_size = N + 1
    best_set = None

    # Phase 1: Try multiple random orderings with greedy construction
    # Use bitmask for sum tracking for speed
    for attempt in range(500):
        if time.time() > deadline:
            break

        # Random permutation of [1, N]
        order = list(range(1, N + 1))
        random.shuffle(order)

        selected = []
        sum_mask = 0  # bit i set means sum (i+2) is used

        for x in order:
            # Check if x can be added: all sums x+a for a in selected (a<=x) and 2*x must be unused
            # We need to check sums x+a where a is in selected and a <= x
            # Actually for Sidon: all unordered pairs (a,b) with a<=b must have distinct sums
            # When adding x, we need to check:
            #   - 2*x is not in sum_mask
            #   - x+a is not in sum_mask for each a in selected where a < x
            #     (if a > x, then a+x was already checked when a was added, but we need x+a which is same)
            # Actually, when we add x, the new sums created are x+a for all a in selected (a can be <, =, or > x)
            # but since we're building incrementally, when we add x, all previously added elements a have
            # already had their pair sums with each other. The new sums are x+a for each a in selected,
            # plus 2*x.
            # But we only need to check x+a for a in selected where a <= x? No, we need ALL a in selected.
            # Wait: if a > x, then when a was added, the sum a+x wasn't created yet (x wasn't in set).
            # So we DO need to check x+a for all a in selected.

            # Check 2*x
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue

            # Check x+a for each a in selected
            can_add = True
            for a in selected:
                s = x + a
                idx = s - 2
                if sum_mask & (1 << idx):
                    can_add = False
                    break

            if can_add:
                # Add x
                selected.append(x)
                sum_mask |= (1 << (2 * x - 2))
                for a in selected[:-1]:  # all previously selected
                    sum_mask |= (1 << (x + a - 2))

        # Now check if the set is maximal
        sel_set = set(selected)
        is_maximal = True
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            # Try to add x
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            for a in selected:
                s = x + a
                idx = s - 2
                if sum_mask & (1 << idx):
                    break
            else:
                # x can be added, so not maximal
                is_maximal = False
                break

        if is_maximal:
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                if best_size <= 10:  # very good, stop early
                    break

    # Phase 2: Local search - try to remove elements from best_set and see if still maximal
    if best_set is not None:
        current = best_set[:]
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try removing current[i]
                candidate = current[:i] + current[i+1:]
                if len(candidate) == 0:
                    continue
                # Check if candidate is Sidon
                sums = set()
                valid = True
                for j in range(len(candidate)):
                    a = candidate[j]
                    for k in range(j, len(candidate)):
                        b = candidate[k]
                        s = a + b
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if not valid:
                    continue
                # Check if candidate is maximal
                cand_set = set(candidate)
                is_max = True
                for x in range(1, N + 1):
                    if x in cand_set:
                        continue
                    # Check if x can be added to candidate
                    ok = True
                    if 2 * x in sums:
                        ok = False
                    else:
                        for a in candidate:
                            if x + a in sums:
                                ok = False
                                break
                    if ok:
                        is_max = False
                        break
                if is_max:
                    current = candidate
                    improved = True
                    break
            if not improved:
                break
        if len(current) < best_size:
            best_size = len(current)
            best_set = current

    # Phase 3: More randomized greedy with different heuristics
    for attempt in range(300):
        if time.time() > deadline:
            break

        # Weighted random: prefer smaller numbers (they create smaller sums, potentially blocking more)
        weights = [1.0 / (i + 1) for i in range(N)]
        order = list(range(1, N + 1))
        # Partial shuffle with bias
        for i in range(N):
            j = random.randint(i, N - 1)
            order[i], order[j] = order[j], order[i]

        selected = []
        sum_mask = 0

        for x in order:
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            can_add = True
            for a in selected:
                s = x + a
                idx = s - 2
                if sum_mask & (1 << idx):
                    can_add = False
                    break
            if can_add:
                selected.append(x)
                sum_mask |= (1 << (2 * x - 2))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x + a - 2))

        sel_set = set(selected)
        is_maximal = True
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            for a in selected:
                s = x + a
                idx = s - 2
                if sum_mask & (1 << idx):
                    break
            else:
                is_maximal = False
                break

        if is_maximal:
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                if best_size <= 10:
                    break

    if best_set is None:
        # Fallback: simple greedy
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            can_add = True
            for a in selected:
                s = x + a
                idx = s - 2
                if sum_mask & (1 << idx):
                    can_add = False
                    break
            if can_add:
                selected.append(x)
                sum_mask |= (1 << (2 * x - 2))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x + a - 2))
        return selected

    return best_set
