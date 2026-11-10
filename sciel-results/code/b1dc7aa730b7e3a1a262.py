import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit (or 90s if specified, but prompt says 60s in one place, 90s in another. "Each evaluation must finish within 60 s". I will use 55s to be safe.)

    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        reciprocals = [Fraction(1, e) for e in elements]
        sums = [Fraction(0, 1)]
        seen = {Fraction(0, 1)}
        for r in reciprocals:
            new_sums = []
            # Optimization: Check collision before adding all
            # If any s + r is in seen, it's a collision
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                # We add to seen only after checking all? 
                # No, if s1 + r == s2 + r then s1 == s2 which is impossible since sums are distinct.
                # The collision is between new sums and OLD sums, or between new sums themselves?
                # New sums are distinct because s are distinct.
                # So we only need to check if ns is in `seen` (old sums).
                # However, if we add ns to seen immediately, do we check against other new sums?
                # s1 + r == s2 + r => s1 == s2. Since s1 != s2, new sums are distinct.
                # So checking against `seen` (which contains old sums) is sufficient.
                # But wait, `seen` is updated?
                # In the loop, we iterate over `sums` (the old list).
                # We should not add `ns` to `seen` until we have checked all `s` in `sums`?
                # Actually, if we add `ns` to `seen` inside the loop, and later `s' + r` equals `ns`, that would imply `s' = s`, which is false.
                # So it is safe to add to `seen` immediately?
                # Yes, because new sums are distinct from each other.
                # But we must ensure we don't collide with *other* new sums if we were doing a different operation?
                # No, `s + r` is injective for distinct `s`.
                # So `new_sums` are distinct.
                # The only collision is `new_sum in old_sums`.
                # So `if ns in seen: return False` is correct.
                # We can add `ns` to `seen` immediately.
                seen.add(ns)
            sums.extend(new_sums)
        return True

    all_elements = list(range(1, 23))
    best_set = []

    # Define multiple orders to try
    # 1. Descending
    orders = [
        list(range(22, 0, -1)),
        list(range(1, 23)),
        list(range(22, 0, -2)) + list(range(1, 23, 2)),
        list(range(1, 23, 2)) + list(range(22, 0, -2)),
        # Interleaved
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
    ]

    # Random restarts
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        rand_order = all_elements[:]
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for e in order:
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        if len(current) > len(best_set):
            best_set = current[:]
            # If we found a very large set, we might want to stop early or prioritize improving it
            if len(best_set) >= 18:
                # Continue to see if we can get 19+
                pass

    # Local Search: Try to add remaining elements
    if time.time() < deadline:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                candidate = best_set + [e]
                if is_dissociated(candidate):
                    best_set = candidate
                    remaining.remove(e)
                    improved = True

    # Local Search: Swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_len = len(best_set)
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(len(best_set)):
                    if i == j:
                        continue
                    for e in all_elements:
                        if e in best_set:
                            continue
                        # Swap best_set[j] with e
                        candidate = best_set[:]
                        candidate[j] = e
                        if is_dissociated(candidate):
                            # Try to fill with remaining
                            rem = [x for x in all_elements if x not in candidate]
                            improved_candidate = candidate[:]
                            for x in rem:
                                if time.time() > deadline:
                                    break
                                test = improved_candidate + [x]
                                if is_dissociated(test):
                                    improved_candidate = test
                            if len(improved_candidate) > current_len:
                                best_set = improved_candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Final verification
    if not is_dissociated(best_set):
        # Try to repair
        for e in best_set[:]:
            test = [x for x in best_set if x != e]
            if is_dissociated(test):
                best_set = test
                break
        if not is_dissociated(best_set):
            best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
