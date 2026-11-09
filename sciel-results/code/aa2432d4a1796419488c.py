import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin under 90s kill

    # Fast reachable-sums tracker using a boolean list (index = sum, value = reachable)
    # Only sums 0..200 matter; we cap at 200.

    def can_add(reach, x):
        # Adding x creates sum 200 iff (200 - x) is reachable.
        return not reach[200 - x]

    def add_to(reach, x):
        # Update reachable sums after adding x (x in 1..100).
        # Iterate descending so we don't reuse x within the same step.
        for s in range(200 - x, -1, -1):
            if reach[s]:
                reach[s + x] = True

    def recalc_reachable(S):
        """Return reach array (list of bools length 201) for set S, or None if 200 reachable."""
        reach = [False] * 201
        reach[0] = True
        for x in S:
            # check
            if reach[200 - x]:
                return None
            for s in range(200 - x, -1, -1):
                if reach[s]:
                    reach[s + x] = True
        return reach

    def greedy_init(order):
        selected = set()
        reach = [False] * 201
        reach[0] = True
        for x in order:
            if not reach[200 - x]:
                selected.add(x)
                for s in range(200 - x, -1, -1):
                    if reach[s]:
                        reach[s + x] = True
        return selected, reach

    # Diverse orderings for greedy initialization
    orderings = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 49, -1)) + list(range(49, 0, -1)),
        [x for x in range(1, 101) if x % 2 == 0] + [x for x in range(1, 101) if x % 2 == 1],
        [x for x in range(1, 101) if x % 2 == 1] + [x for x in range(1, 101) if x % 2 == 0],
        [x for x in range(1, 101) if x <= 10] + list(range(100, 10, -1)),
        list(range(100, 90, -1)) + [x for x in range(90, 0, -1) if x % 3 != 0],
        sorted(range(1, 101), key=lambda x: abs(x - 50)),
        [x for x in range(1, 101) if x % 5 in [0, 1]] + [x for x in range(1, 101) if x % 5 not in [0, 1]],
        # Additional orderings
        list(range(75, 101)) + list(range(1, 75)),
        list(range(1, 76)) + list(range(100, 75, -1)),
        [x for x in range(1, 101) if x % 4 in [0, 1]] + [x for x in range(1, 101) if x % 4 in [2, 3]],
        [x for x in range(1, 101) if x % 4 in [2, 3]] + [x for x in range(1, 101) if x % 4 in [0, 1]],
        sorted(range(1, 101), key=lambda x: x % 7) + list(range(100, 0, -1)),
    ]

    best_S = set()
    best_R = None
    best_size = 0

    for order in orderings:
        S, R = greedy_init(order)
        if len(S) > best_size:
            best_S = S
            best_R = R
            best_size = len(S)

    S = best_S
    R = best_R
    missing = set(range(1, 101)) - S

    no_progress_count = 0
    max_no_progress = 60

    while time.time() < deadline:
        # Phase 1: Try to add elements (sorted by how "safe" they are — prefer those where 200-x is NOT reachable)
        added_any = False
        # Sort missing: safe first (200-x not in R), then unsafe
        safe = [x for x in missing if not R[200 - x]]
        if safe:
            # Try adding the largest safe element first (keeps flexibility)
            for x in sorted(safe, reverse=True):
                S.add(x)
                for s in range(200 - x, -1, -1):
                    if R[s]:
                        R[s + x] = True
                missing.discard(x)
                added_any = True
                break

        if added_any:
            no_progress_count = 0
            continue

        # Phase 2: Try single removals to free up space for additions
        removed_any = False
        for y in list(S):
            if time.time() > deadline:
                break
            S_test = S - {y}
            R_test = recalc_reachable(S_test)
            if R_test is not None:
                # Check if removing y allows adding something
                can_add_list = [x for x in missing if not R_test[200 - x]]
                if can_add_list:
                    best_add = max(can_add_list)
                    S = S_test
                    R = R_test
                    S.add(best_add)
                    missing.discard(best_add)
                    for s in range(200 - best_add, -1, -1):
                        if R[s]:
                            R[s + best_add] = True
                    removed_any = True
                    no_progress_count = 0
                    break

        if removed_any:
            continue

        # Phase 3: Try swaps (remove y, add x) where net size stays same or we can then add more
        swap_done = False
        candidates = list(missing)
        if candidates and S:
            random.shuffle(candidates)
            for x in candidates[:min(len(candidates), 20)]:
                if time.time() > deadline:
                    break
                for y in list(S):
                    if time.time() > deadline:
                        break
                    S_test = S - {y}
                    S_test.add(x)
                    R_test = recalc_reachable(S_test)
                    if R_test is not None:
                        # Check if this swap enables further additions
                        extra_adds = 0
                        temp_R = R_test[:]
                        temp_missing = missing - {x}
                        temp_missing.add(y)
                        for z in sorted(temp_missing, reverse=True):
                            if not temp_R[200 - z]:
                                extra_adds += 1
                                for s in range(200 - z, -1, -1):
                                    if temp_R[s]:
                                        temp_R[s + z] = True
                        if extra_adds > 0:
                            S = S_test
                            R = R_test
                            missing = set(range(1, 101)) - S
                            for z in sorted(temp_missing, reverse=True):
                                if z in missing and not R[200 - z]:
                                    S.add(z)
                                    missing.discard(z)
                                    for s in range(200 - z, -1, -1):
                                        if R[s]:
                                            R[s + z] = True
                            swap_done = True
                            no_progress_count = 0
                            break
                if swap_done:
                    break

        if swap_done:
            continue

        # Phase 4: Random restarts if stuck
        no_progress_count += 1
        if no_progress_count >= max_no_progress:
            k = random.randint(2, min(5, len(S)))
            if k > 0 and S:
                to_remove = random.sample(list(S), k)
                S = S - set(to_remove)
                R = recalc_reachable(S)
                if R is None:
                    R = [False] * 201
                    R[0] = True
                    for x in S:
                        for s in range(200 - x, -1, -1):
                            if R[s]:
                                R[s + x] = True
                missing = set(range(1, 101)) - S
                no_progress_count = 0
                # Re-greedy fill
                for x in sorted(missing, reverse=True):
                    if not R[200 - x]:
                        S.add(x)
                        missing.discard(x)
                        for s in range(200 - x, -1, -1):
                            if R[s]:
                                R[s + x] = True

    # Final polish pass: repeated full passes of additions until no more fit
    improved = True
    while improved:
        improved = False
        for x in sorted(missing, reverse=True):
            if not R[200 - x]:
                S.add(x)
                missing.discard(x)
                for s in range(200 - x, -1, -1):
                    if R[s]:
                        R[s + x] = True
                improved = True

    return list(S)
