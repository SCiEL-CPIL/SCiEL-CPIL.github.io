N = 256

import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 28.0

    def greedy_randomized():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        selected = []
        sums = set()

        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if sums.intersection(added_sums):
                continue

            selected.append(x)
            sums.update(added_sums)

        return selected

    def can_add(x, selected, sums):
        """Check if x can be added to selected while maintaining Sidon property."""
        seen = set()
        for a in selected:
            s = x + a
            if s in sums:
                return False
            if s in seen:
                return False
            seen.add(s)
        s2 = 2 * x
        if s2 in sums:
            return False
        if s2 in seen:
            return False
        return True

    def is_maximal_fast(selected, sums):
        """Check inclusion maximality."""
        sel_set = set(selected)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if can_add(x, selected, sums):
                return False
        return True

    def rebuild_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    best_set = None
    best_size = float('inf')

    # Phase 1: Many randomized greedy restarts
    while time.time() < deadline - 12.0:
        current_set = greedy_randomized()
        if current_set and len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    if best_set is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected
        best_size = len(best_set)

    # Phase 2: Aggressive local search
    while time.time() < deadline - 1.0:
        if not best_set:
            break

        sums = rebuild_sums(best_set)
        improved = False

        # Try removing each element
        for idx in range(len(best_set) - 1, -1, -1):
            new_selected = best_set[:idx] + best_set[idx+1:]
            if not new_selected:
                continue
            new_sums = rebuild_sums(new_selected)
            if is_maximal_fast(new_selected, new_sums):
                best_set = new_selected
                best_size = len(new_selected)
                improved = True
                break

        if not improved:
            # Try 2-removal + refill
            if len(best_set) > 4 and random.random() < 0.5:
                for _ in range(5):
                    idxs = random.sample(range(len(best_set)), 2)
                    idxs.sort(reverse=True)
                    new_selected = list(best_set)
                    for idx in idxs:
                        new_selected.pop(idx)
                    sums_new = rebuild_sums(new_selected)
                    for x in random.sample(range(1, N + 1), min(50, N)):
                        if x in new_selected:
                            continue
                        if can_add(x, new_selected, sums_new):
                            new_selected.append(x)
                            sums_new.add(2 * x)
                            for a in new_selected[:-1]:
                                sums_new.add(x + a)
                    if len(new_selected) < best_size:
                        if is_maximal_fast(new_selected, rebuild_sums(new_selected)):
                            best_set = new_selected
                            best_size = len(new_selected)
                            improved = True
                            break
            else:
                current_set = greedy_randomized()
                if current_set and len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set

    # Final verification
    def verify_final(selected):
        sel_set = set(selected)
        if len(sel_set) != len(selected):
            return False
        if any(x < 1 or x > N for x in selected):
            return False
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) != len(added):
                continue
            if not sums.intersection(added):
                return False
        return True

    if not verify_final(best_set):
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    return best_set
