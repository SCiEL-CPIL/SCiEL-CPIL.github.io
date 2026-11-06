"""Construct a large 4-AP-free subset of [1, 512]."""
import time
import random

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    random.seed(42)

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with a+3d <= N
    # For each element x, store the list of APs (as tuples) that contain x
    # AP index -> set of 4 elements
    ap_by_element = {x: [] for x in range(1, N + 1)}
    all_aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                ap_by_element[x].append(ap_idx)

    # A set is 4-AP-free iff no AP has all 4 elements in the set.
    # We track: for each AP, how many of its 4 elements are in the set.
    # An AP is "bad" if count == 4.

    # Greedy construction: add elements in order of fewest conflicts first
    # (i.e., elements that appear in fewer APs are "safer" to add)
    order = sorted(range(1, N + 1), key=lambda x: len(ap_by_element[x]))

    selected = set()
    ap_count = [0] * len(all_aps)  # how many elements of each AP are selected

    def can_add(x):
        """Check if adding x creates a 4-AP."""
        for ap_idx in ap_by_element[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True

    def add(x):
        selected.add(x)
        for ap_idx in ap_by_element[x]:
            ap_count[ap_idx] += 1

    def remove(x):
        selected.remove(x)
        for ap_idx in ap_by_element[x]:
            ap_count[ap_idx] -= 1

    # Phase 1: Greedy add in conflict-order
    for x in order:
        if can_add(x):
            add(x)

    best_size = len(selected)

    # Phase 2: Local search with swaps
    # Try to swap out an element for an excluded one to increase size or improve potential
    # "Potential": for each excluded element, count how many APs it would conflict with.
    # Lower conflict count = better candidate to add.

    def conflict_count(x):
        """Number of APs where x would complete a 4-AP (i.e., 3 others already selected)."""
        c = 0
        for ap_idx in ap_by_element[x]:
            if ap_count[ap_idx] == 3:
                c += 1
        return c

    # Precompute for each element the number of APs it's in (static)
    ap_degree = {x: len(ap_by_element[x]) for x in range(1, N + 1)}

    # Swap local search
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        # Build list of excluded elements sorted by conflict count (ascending)
        excluded = [x for x in range(1, N + 1) if x not in selected]
        # For each excluded element, compute how many elements we'd need to remove
        # to make it addable. That's the number of "blocking" elements.
        # A simpler approach: try 1-for-1 swaps.
        # For each included element y, for each excluded element x,
        # check if removing y and adding x keeps 4-AP-free and increases size (never, since 1-for-1)
        # Actually 1-for-1 doesn't increase size. We need to find a way to increase size.
        # 
        # Better: find an excluded element x that can be added if we remove a small set of elements.
        # If |blocking set| < 1, we can just add. If |blocking set| == 1, we can swap.
        # If |blocking set| == 2, we remove 2 and add 1, net -1, not helpful.
        #
        # So the key is: find x with conflict_count(x) == 0 (just add) or == 1 (swap).
        #
        # Let's first try to add any x with conflict_count == 0.
        for x in excluded:
            if conflict_count(x) == 0:
                add(x)
                improved = True
                best_size = len(selected)
                # Restart the loop since conflicts changed
                break

        if improved:
            continue

        # Now try swaps: remove one element y, add one element x, net size same.
        # But we want to increase size, so swaps alone won't help unless they enable future additions.
        # Let's try: for each excluded x with conflict_count(x) == 1, find the blocking element y,
        # remove y, add x. This keeps size same but might enable further additions.
        # We do this if it doesn't hurt (and we re-check after).
        
        # More aggressive: try to find a "2-for-3" improvement:
        # Remove 2 elements, add 3 elements. This is complex.
        #
        # Let's do a simple hill-climbing: try random swaps and keep if size increases
        # (impossible for 1-for-1) or if it enables a subsequent add.
        
        # Alternative: use a different greedy order with randomization and restart.
        # Let's try a few random restarts with different orders.
        break

    # Phase 3: Multiple random restarts with greedy + local add
    best_set = set(selected)
    best_size = len(selected)

    for restart in range(200):
        if (time.time() - start_time) > DEADLINE:
            break

        # Random order
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        # Sort by a random perturbation of conflict count
        # Actually, let's try: sort by (conflict_count, random) or just random
        selected2 = set()
        ap_count2 = [0] * len(all_aps)

        def can_add2(x):
            for ap_idx in ap_by_element[x]:
                if ap_count2[ap_idx] == 3:
                    return False
            return True

        def add2(x):
            selected2.add(x)
            for ap_idx in ap_by_element[x]:
                ap_count2[ap_idx] += 1

        for x in elems:
            if can_add2(x):
                add2(x)

        # Local add phase
        changed = True
        while changed and (time.time() - start_time) < DEADLINE:
            changed = False
            for x in range(1, N + 1):
                if x not in selected2:
                    if can_add2(x):
                        add2(x)
                        changed = True
                        break

        if len(selected2) > best_size:
            best_size = len(selected2)
            best_set = set(selected2)

    # Final: try to add more to best_set
    # (already done in local add phase)

    return sorted(best_set)
