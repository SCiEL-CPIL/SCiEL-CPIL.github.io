import math
import time
from fractions import Fraction
from itertools import combinations

DENOMS = list(range(2, 49))
N = len(DENOMS)
ONE = Fraction(1, 1)
FLOAT_VALS = [1.0 / d for d in DENOMS]
TIME_LIMIT = 85.0  # Leave 5s buffer before 90s kill


def construct():
    start_time = time.time()

    # Precompute: for each index, the float value
    # We want to find a subset S of indices such that sum(1/DENOMS[i] for i in S) < 1
    # and we want to minimize 1 - sum (i.e., maximize the sum while keeping it < 1)

    # Strategy: Use a targeted search with multiple initializations and local refinement.
    # The key insight is that we want the sum to be as close to 1 as possible from below.
    # 
    # Approach:
    # 1. Try several greedy initializations with different orderings
    # 2. Apply local search (add, remove, swap, 2-swap) to refine
    # 3. Use exact Fraction arithmetic for final verification
    # 4. Also try a more systematic search for small subsets

    best_deficit = 1.0  # Start with deficit = 1 (empty set)
    best_indices = []

    def time_check():
        return time.time() - start_time < TIME_LIMIT

    def refine(selected_set, current_sum):
        """Local search refinement: add, swap, 2-swap moves."""
        nonlocal best_deficit, best_indices
        sel = set(selected_set)
        s = current_sum

        def update_best():
            nonlocal best_deficit, best_indices
            deficit = 1.0 - s
            if deficit < best_deficit and deficit > 0:
                best_deficit = deficit
                best_indices = sorted(sel)

        update_best()

        improved = True
        while improved and time_check():
            improved = False

            # Try adding one element
            best_add = -1
            best_add_def = 1.0 - s
            for j in range(N):
                if j not in sel:
                    new_s = s + FLOAT_VALS[j]
                    if new_s < 1.0:
                        nd = 1.0 - new_s
                        if nd < best_add_def:
                            best_add_def = nd
                            best_add = j
            if best_add != -1:
                sel.add(best_add)
                s += FLOAT_VALS[best_add]
                improved = True
                update_best()
                continue

            # Try swapping: remove a, add b
            best_a = -1
            best_b = -1
            best_swap_def = 1.0 - s
            for a in sel:
                base = s - FLOAT_VALS[a]
                for b in range(N):
                    if b in sel:
                        continue
                    new_s = base + FLOAT_VALS[b]
                    if new_s < 1.0:
                        nd = 1.0 - new_s
                        if nd < best_swap_def:
                            best_swap_def = nd
                            best_a = a
                            best_b = b
            if best_a != -1:
                sel.discard(best_a)
                sel.add(best_b)
                s = s - FLOAT_VALS[best_a] + FLOAT_VALS[best_b]
                improved = True
                update_best()
                continue

            # Try 2-swap: remove a, add b, c (b != c, b,c not in sel)
            best_a = -1
            best_b = -1
            best_c = -1
            best_2swap_def = 1.0 - s
            for a in sel:
                base = s - FLOAT_VALS[a]
                for b in range(N):
                    if b in sel:
                        continue
                    v1 = FLOAT_VALS[b]
                    if base + v1 >= 1.0:
                        continue
                    for c in range(b + 1, N):
                        if c in sel:
                            continue
                        v2 = FLOAT_VALS[c]
                        new_s = base + v1 + v2
                        if new_s < 1.0:
                            nd = 1.0 - new_s
                            if nd < best_2swap_def:
                                best_2swap_def = nd
                                best_a = a
                                best_b = b
                                best_c = c
            if best_a != -1:
                sel.discard(best_a)
                sel.add(best_b)
                sel.add(best_c)
                s = s - FLOAT_VALS[best_a] + FLOAT_VALS[best_b] + FLOAT_VALS[best_c]
                improved = True
                update_best()
                continue

            # Try 1-remove, 2-add is covered above.
            # Try 2-remove, 3-add? Too expensive.
            # Try removing one and adding two where the two are larger than what was removed
            # This is the same as the 2-swap above.

            # Try a "kick" move: remove two, add two
            best_a1 = -1
            best_a2 = -1
            best_b1 = -1
            best_b2 = -1
            best_22_def = 1.0 - s
            sel_list = list(sel)
            for i in range(len(sel_list)):
                a1 = sel_list[i]
                for j in range(i + 1, len(sel_list)):
                    a2 = sel_list[j]
                    base = s - FLOAT_VALS[a1] - FLOAT_VALS[a2]
                    for b1 in range(N):
                        if b1 in sel:
                            continue
                        v1 = FLOAT_VALS[b1]
                        if base + v1 >= 1.0:
                            continue
                        for b2 in range(b1 + 1, N):
                            if b2 in sel:
                                continue
                            v2 = FLOAT_VALS[b2]
                            new_s = base + v1 + v2
                            if new_s < 1.0:
                                nd = 1.0 - new_s
                                if nd < best_22_def:
                                    best_22_def = nd
                                    best_a1 = a1
                                    best_a2 = a2
                                    best_b1 = b1
                                    best_b2 = b2
            if best_a1 != -1:
                sel.discard(best_a1)
                sel.discard(best_a2)
                sel.add(best_b1)
                sel.add(best_b2)
                s = s - FLOAT_VALS[best_a1] - FLOAT_VALS[best_a2] + FLOAT_VALS[best_b1] + FLOAT_VALS[best_b2]
                improved = True
                update_best()
                continue

        return sel, s

    # Multiple greedy initializations
    # Strategy 1: Greedy by largest value that fits
    sel1 = set()
    s1 = 0.0
    for i in range(N):  # Already sorted by value descending (index 0 = 1/2, index 1 = 1/3, ...)
        if s1 + FLOAT_VALS[i] < 1.0:
            sel1.add(i)
            s1 += FLOAT_VALS[i]
    sel1, s1 = refine(sel1, s1)

    # Strategy 2: Start from full set, remove largest until sum < 1
    sel2 = set(range(N))
    s2 = sum(FLOAT_VALS)
    for i in range(N):
        if s2 < 1.0:
            break
        if i in sel2:
            sel2.discard(i)
            s2 -= FLOAT_VALS[i]
    sel2, s2 = refine(sel2, s2)

    # Strategy 3: Greedy but with a different ordering - try smaller denominators first but skip some
    # Actually let's try: sort by value, but use a "skip" heuristic
    sel3 = set()
    s3 = 0.0
    # Try to include 1/2, then see what fits
    for i in range(N):
        if s3 + FLOAT_VALS[i] < 1.0:
            sel3.add(i)
            s3 += FLOAT_VALS[i]
    # This is the same as strategy 1 since indices are already in order of decreasing value.
    # Let's try a different approach: random-ish restarts with deterministic seeds

    # Strategy 4: Try subsets of size 2, 3, 4, 5 systematically for small sizes
    # For size 2: all pairs
    for i in range(N):
        for j in range(i + 1, N):
            s = FLOAT_VALS[i] + FLOAT_VALS[j]
            if s < 1.0:
                d = 1.0 - s
                if d < best_deficit:
                    best_deficit = d
                    best_indices = [i, j]
        if not time_check():
            break

    # For size 3:
    if time_check():
        for i in range(N):
            for j in range(i + 1, N):
                s2 = FLOAT_VALS[i] + FLOAT_VALS[j]
                if s2 >= 1.0:
                    continue
                for k in range(j + 1, N):
                    s = s2 + FLOAT_VALS[k]
                    if s < 1.0:
                        d = 1.0 - s
                        if d < best_deficit:
                            best_deficit = d
                            best_indices = [i, j, k]
            if not time_check():
                break

    # For size 4:
    if time_check():
        for i in range(N):
            for j in range(i + 1, N):
                s2 = FLOAT_VALS[i] + FLOAT_VALS[j]
                if s2 >= 1.0:
                    continue
                for k in range(j + 1, N):
                    s3 = s2 + FLOAT_VALS[k]
                    if s3 >= 1.0:
                        continue
                    for l in range(k + 1, N):
                        s = s3 + FLOAT_VALS[l]
                        if s < 1.0:
                            d = 1.0 - s
                            if d < best_deficit:
                                best_deficit = d
                                best_indices = [i, j, k, l]
                    if not time_check():
                        break
                if not time_check():
                    break
            if not time_check():
                break

    # For size 5:
    if time_check():
        for i in range(N):
            for j in range(i + 1, N):
                s2 = FLOAT_VALS[i] + FLOAT_VALS[j]
                if s2 >= 1.0:
                    continue
                for k in range(j + 1, N):
                    s3 = s2 + FLOAT_VALS[k]
                    if s3 >= 1.0:
                        continue
                    for l in range(k + 1, N):
                        s4 = s3 + FLOAT_VALS[l]
                        if s4 >= 1.0:
                            continue
                        for m in range(l + 1, N):
                            s = s4 + FLOAT_VALS[m]
                            if s < 1.0:
                                d = 1.0 - s
                                if d < best_deficit:
                                    best_deficit = d
                                    best_indices = [i, j, k, l, m]
                        if not time_check():
                            break
                    if not time_check():
                        break
                if not time_check():
                    break
            if not time_check():
                break

    # Now do more targeted local search from the best found so far
    if best_indices and time_check():
        sel_best = set(best_indices)
        s_best = sum(FLOAT_VALS[i] for i in sel_best)
        sel_best, s_best = refine(sel_best, s_best)

    # Try additional random-like restarts with different initial subsets
    # Use deterministic "random" based on simple hash
    if time_check():
        # Try starting with just 1/2 and then greedily adding
        for start_set in [
            {0},  # 1/2
            {1},  # 1/3
            {0, 1},  # 1/2 + 1/3
            {0, 2},  # 1/2 + 1/4
            {0, 3},  # 1/2 + 1/5
            {0, 4},  # 1/2 + 1/6
            {1, 2},  # 1/3 + 1/4
            {2, 3},  # 1/4 + 1/5
            {0, 1, 2},  # 1/2 + 1/3 + 1/4
            {0, 1, 3},  # 1/2 + 1/3 + 1/5
            {0, 2, 3},  # 1/2 + 1/4 + 1/5
            {0, 3, 4},  # 1/2 + 1/5 + 1/6
        ]:
            if not time_check():
                break
            # Only start if the initial sum is < 1
            init_sum = sum(FLOAT_VALS[i] for i in start_set)
            if init_sum < 1.0:
                sel_init = set(start_set)
                s_init = init_sum
                # Greedily add more
                for i in range(N):
                    if i not in sel_init and s_init + FLOAT_VALS[i] < 1.0:
                        sel_init.add(i)
                        s_init += FLOAT_VALS[i]
                sel_init, s_init = refine(sel_init, s_init)

    # Final verification with exact arithmetic
    if best_indices:
        exact_sum = sum(Fraction(1, DENOMS[i]) for i in best_indices)
        if exact_sum >= ONE:
            # Fallback: remove elements until sum < 1
            # Sort by value descending
            items = sorted(best_indices, key=lambda i: FLOAT_VALS[i], reverse=True)
            for i in items:
                if exact_sum < ONE:
                    break
                best_indices.remove(i)
                exact_sum -= Fraction(1, DENOMS[i])
            if not best_indices:
                # Last resort: return a single small fraction
                best_indices = [N - 1]  # 1/48
        # Final check
        exact_sum = sum(Fraction(1, DENOMS[i]) for i in best_indices)
        if exact_sum >= ONE:
            # This shouldn't happen, but just in case
            best_indices = [N - 1]

    return sorted(best_indices)


if __name__ == "__main__":
    idx = construct()
    s = sum(Fraction(1, DENOMS[i]) for i in idx)
    print("indices:", idx)
    print("sum:", s)
    print("deficit:", Fraction(1, 1) - s)
