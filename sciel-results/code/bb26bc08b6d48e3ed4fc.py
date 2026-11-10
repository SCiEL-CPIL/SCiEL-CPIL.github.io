# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z using bitmask-based
    conflict checking, multiple construction heuristics, and intensive local search.
    """
    m = 205
    deadline = time.monotonic() + 610.0  # safety margin before 630s kill

    # --- Precomputation: forbidden differences ---
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)

    # Conflict bitmask: conflict_mask[i] is a Python int with bit j set if i conflicts with j
    # Two elements i, j conflict if (i-j) mod m is in forbidden OR (j-i) mod m is in forbidden
    # Since (i-j) mod m + (j-i) mod m = m (when i != j), and forbidden may not be symmetric,
    # we check both directions.
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Also build a list of conflicts for each node (for iteration)
    conflict_list = [0] * m
    for i in range(m):
        lst = []
        mask = conflict_mask[i]
        j = 0
        while mask:
            if mask & 1:
                lst.append(j)
            mask >>= 1
            j += 1
        conflict_list[i] = lst

    def is_feasible(subset_mask, candidate):
        """Check if adding candidate to subset_mask is feasible (no conflicts)."""
        # candidate must not be in subset
        if subset_mask & (1 << candidate):
            return False
        # candidate must not conflict with any element in subset
        return (conflict_mask[candidate] & subset_mask) == 0

    def subset_size(subset_mask):
        return bin(subset_mask).count('1')

    def mask_to_set(subset_mask):
        result = set()
        i = 0
        mask = subset_mask
        while mask:
            if mask & 1:
                result.add(i)
            mask >>= 1
            i += 1
        return result

    def set_to_mask(s):
        mask = 0
        for x in s:
            mask |= (1 << x)
        return mask

    # --- Strategy 1: Greedy construction with different orderings ---
    def greedy_construct(order):
        """Build a maximal independent set by processing elements in given order."""
        subset = 0
        for cand in order:
            if is_feasible(subset, cand):
                subset |= (1 << cand)
        return subset

    # --- Strategy 2: Local search with bitmask operations ---
    def local_search(initial_mask, rng, max_iters=50000):
        """
        Intensive local search: try adds, swaps, and 2-swaps.
        Use bitmask for O(1) feasibility checks.
        """
        current = initial_mask
        cur_size = subset_size(current)
        best = current
        best_size = cur_size
        no_improve = 0
        max_no_improve = 3000

        # Precompute list of all elements not in current for add attempts
        all_elements = list(range(m))

        for it in range(max_iters):
            if time.monotonic() > deadline:
                break
            if no_improve > max_no_improve:
                break

            improved_this_iter = False

            # Try to add a random element not in current
            if cur_size < m:
                # Find elements not in current
                not_in = [x for x in all_elements if not (current & (1 << x))]
                if not_in:
                    for _ in range(15):
                        cand = rng.choice(not_in)
                        if is_feasible(current, cand):
                            new_mask = current | (1 << cand)
                            new_size = cur_size + 1
                            current = new_mask
                            cur_size = new_size
                            improved_this_iter = True
                            if cur_size > best_size:
                                best = current
                                best_size = cur_size
                                no_improve = 0
                            break

            if improved_this_iter:
                continue

            # Try a swap: remove one element, add another
            if cur_size > 0:
                in_current = [x for x in all_elements if (current & (1 << x))]
                if in_current:
                    to_remove = rng.choice(in_current)
                    temp_mask = current & ~(1 << to_remove)
                    temp_size = cur_size - 1

                    # Try to add a better element
                    not_in = [x for x in all_elements if not (temp_mask & (1 << x))]
                    found_add = False
                    for _ in range(20):
                        cand = rng.choice(not_in)
                        if cand == to_remove:
                            continue
                        if is_feasible(temp_mask, cand):
                            new_mask = temp_mask | (1 << cand)
                            new_size = temp_size + 1
                            current = new_mask
                            cur_size = new_size
                            found_add = True
                            improved_this_iter = True
                            if cur_size > best_size:
                                best = current
                                best_size = cur_size
                                no_improve = 0
                            break

                    if not found_add:
                        # Try 2-opt: remove two, add two (or one)
                        if len(in_current) >= 2:
                            to_remove2 = rng.choice([x for x in in_current if x != to_remove])
                            temp_mask2 = temp_mask & ~(1 << to_remove2)
                            temp_size2 = temp_size - 1

                            not_in2 = [x for x in all_elements if not (temp_mask2 & (1 << x))]
                            found_add2 = False
                            for _ in range(15):
                                cand1 = rng.choice(not_in2)
                                if cand1 == to_remove or cand1 == to_remove2:
                                    continue
                                if is_feasible(temp_mask2, cand1):
                                    temp_mask2 |= (1 << cand1)
                                    temp_size2 += 1
                                    # Try to add one more
                                    not_in3 = [x for x in all_elements if not (temp_mask2 & (1 << x))]
                                    found_add2 = True
                                    for _2 in range(10):
                                        cand2 = rng.choice(not_in3)
                                        if cand2 == to_remove or cand2 == to_remove2 or cand2 == cand1:
                                            continue
                                        if is_feasible(temp_mask2, cand2):
                                            temp_mask2 |= (1 << cand2)
                                            temp_size2 += 1
                                            break
                                    current = temp_mask2
                                    cur_size = temp_size2
                                    if cur_size > best_size:
                                        best = current
                                        best_size = cur_size
                                        no_improve = 0
                                    break
                            if not found_add2:
                                # Undo
                                current = initial_mask if it == 0 else current
                                # Actually we should restore to the state before this swap attempt
                                # This is tricky, so let's just keep the current state
                                pass

            if not improved_this_iter:
                no_improve += 1
            else:
                no_improve = 0

        return best

    # --- Strategy 3: Systematic local search (deterministic) ---
    def systematic_local_search(initial_mask):
        """
        Deterministic local search: try all possible adds and swaps.
        """
        current = initial_mask
        cur_size = subset_size(current)
        best = current
        best_size = cur_size
        improved = True
        all_elements = list(range(m))

        while improved:
            if time.monotonic() > deadline:
                break
            improved = False

            # Try all possible adds
            for cand in all_elements:
                if not (current & (1 << cand)):
                    if is_feasible(current, cand):
                        new_mask = current | (1 << cand)
                        new_size = cur_size + 1
                        current = new_mask
                        cur_size = new_size
                        improved = True
                        if cur_size > best_size:
                            best = current
                            best_size = cur_size
            if improved:
                continue

            # Try all possible swaps
            in_current = [x for x in all_elements if (current & (1 << x))]
            for to_remove in in_current:
                if time.monotonic() > deadline:
                    break
                temp_mask = current & ~(1 << to_remove)
                temp_size = cur_size - 1
                for cand in all_elements:
                    if cand == to_remove:
                        continue
                    if not (temp_mask & (1 << cand)):
                        if is_feasible(temp_mask, cand):
                            new_mask = temp_mask | (1 << cand)
                            new_size = temp_size + 1
                            if new_size > cur_size:
                                current = new_mask
                                cur_size = new_size
                                improved = True
                                if cur_size > best_size:
                                    best = current
                                    best_size = cur_size
                                break
                    if time.monotonic() > deadline:
                        break

        return best

    # --- Main search loop ---
    best_mask = 0
    best_size = 0

    rng = random.Random(12345)

    # Phase 1: Multiple greedy constructions with diverse orderings
    phase1_deadline = deadline - 300.0  # Reserve 300s for later phases

    # Ordering 1: Random
    while time.monotonic() < phase1_deadline:
        order = list(range(m))
        rng.shuffle(order)
        sol = greedy_construct(order)
        if subset_size(sol) > best_size:
            best_mask = sol
            best_size = subset_size(sol)

        # Also try sorted by degree (fewest conflicts first)
        if time.monotonic() < phase1_deadline - 5:
            degrees = [len(conflict_list[i]) for i in range(m)]
            order_deg = sorted(range(m), key=lambda x: degrees[x])
            sol_deg = greedy_construct(order_deg)
            if subset_size(sol_deg) > best_size:
                best_mask = sol_deg
                best_size = subset_size(sol_deg)

        # Ordering by negative degree (most conflicts first)
        if time.monotonic() < phase1_deadline - 5:
            order_neg = sorted(range(m), key=lambda x: -degrees[x])
            sol_neg = greedy_construct(order_neg)
            if subset_size(sol_neg) > best_size:
                best_mask = sol_neg
                best_size = subset_size(sol_neg)

    # Phase 2: Systematic local search on best
    if time.monotonic() < deadline - 100:
        improved = systematic_local_search(best_mask)
        if subset_size(improved) > best_size:
            best_mask = improved
            best_size = subset_size(improved)

    # Phase 3: Randomized local search with multiple restarts
    phase3_deadline = deadline - 50.0
    restart_count = 0
    while time.monotonic() < phase3_deadline:
        # Start from best or a random greedy solution
        if restart_count % 3 == 0:
            start = best_mask
        else:
            order = list(range(m))
            rng.shuffle(order)
            start = greedy_construct(order)

        improved = local_search(start, rng, max_iters=20000)
        if subset_size(improved) > best_size:
            best_mask = improved
            best_size = subset_size(improved)

        restart_count += 1

    # Phase 4: Final systematic local search
    if time.monotonic() < deadline - 5:
        improved = systematic_local_search(best_mask)
        if subset_size(improved) > best_size:
            best_mask = improved
            best_size = subset_size(improved)

    # Ensure we have at least one element
    if best_mask == 0:
        best_mask = 1  # element 0

    # Convert to array
    result_set = mask_to_set(best_mask)
    result = np.array(sorted(result_set), dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
