import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin under 90s kill

    # Use Python integers as bitsets for fast reachability
    # bit i is set iff sum i is reachable
    # We only care about sums 0..200

    def reach_add(reach, x):
        """Return new reach bitset after adding element x, or None if 200 becomes reachable."""
        # Check if adding x creates sum 200: need (200-x) to be reachable
        if (reach >> (200 - x)) & 1:
            return None
        # Shift reach by x and OR, but only keep bits 0..200
        new_reach = reach | ((reach << x) & ((1 << 201) - 1))
        return new_reach

    def recalc_reachable(S):
        """Return reach bitset for set S, or None if 200 is reachable."""
        reach = 1  # bit 0 set
        for x in S:
            reach = reach_add(reach, x)
            if reach is None:
                return None
        return reach

    def greedy_init(order):
        selected = set()
        reach = 1
        for x in order:
            if not ((reach >> (200 - x)) & 1):
                selected.add(x)
                reach = reach | ((reach << x) & ((1 << 201) - 1))
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
        list(range(75, 101)) + list(range(1, 75)),
        list(range(1, 76)) + list(range(100, 75, -1)),
        [x for x in range(1, 101) if x % 4 in [0, 1]] + [x for x in range(1, 101) if x % 4 in [2, 3]],
        [x for x in range(1, 101) if x % 4 in [2, 3]] + [x for x in range(1, 101) if x % 4 in [0, 1]],
        sorted(range(1, 101), key=lambda x: x % 7) + list(range(100, 0, -1)),
        # New: start with large numbers to avoid forming 200
        list(range(100, 65, -1)) + list(range(1, 66)),
        list(range(66, 101)) + list(range(1, 66, -1)),
        # Start with numbers that are "far" from forming 200
        sorted(range(1, 101), key=lambda x: min(x, 200-x)),
        sorted(range(1, 101), key=lambda x: abs(x - 100)),
    ]

    best_S = set()
    best_R = 1
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
    max_no_progress = 40

    def try_add(S, R, x):
        """Try to add x to S. Return (new_S, new_R) or None."""
        if (R >> (200 - x)) & 1:
            return None
        new_R = R | ((R << x) & ((1 << 201) - 1))
        new_S = S | {x}
        return new_S, new_R

    def greedy_fill(S, R, missing_set):
        """Greedily add as many elements as possible from missing_set."""
        for x in sorted(missing_set, reverse=True):
            if (R >> (200 - x)) & 1:
                continue
            R = R | ((R << x) & ((1 << 201) - 1))
            S.add(x)
            missing_set.discard(x)
        return S, R

    while time.time() < deadline:
        # Phase 1: Try to add elements
        added_any = False
        safe = [x for x in missing if not ((R >> (200 - x)) & 1)]
        if safe:
            for x in sorted(safe, reverse=True):
                S.add(x)
                R = R | ((R << x) & ((1 << 201) - 1))
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
                can_add_list = [x for x in missing if not ((R_test >> (200 - x)) & 1)]
                if can_add_list:
                    best_add = max(can_add_list)
                    S = S_test
                    R = R_test
                    S.add(best_add)
                    missing.discard(best_add)
                    R = R | ((R << best_add) & ((1 << 201) - 1))
                    removed_any = True
                    no_progress_count = 0
                    break

        if removed_any:
            continue

        # Phase 3: Try 2-for-1 exchanges (remove 2, add 1) to improve
        swap_done = False
        if len(S) >= 2 and missing:
            # Sample some 2-element removals
            S_list = list(S)
            random.shuffle(S_list)
            num_trials = min(50, len(S_list) * (len(S_list) - 1) // 2)
            for _ in range(num_trials):
                if time.time() > deadline:
                    break
                y1, y2 = random.sample(S_list, 2)
                S_test = S - {y1, y2}
                R_test = recalc_reachable(S_test)
                if R_test is None:
                    continue
                # Try adding one element from missing
                for x in sorted(missing, reverse=True):
                    if (R_test >> (200 - x)) & 1:
                        continue
                    # Check if this swap is beneficial: we removed 2, add 1, net -1
                    # But maybe we can then add more
                    S_new = S_test | {x}
                    R_new = R_test | ((R_test << x) & ((1 << 201) - 1))
                    # Try greedy fill
                    temp_missing = missing - {x} | {y1, y2}
                    S_fill, R_fill = greedy_fill(set(S_new), R_new, set(temp_missing))
                    if len(S_fill) > len(S):
                        S = S_fill
                        R = R_fill
                        missing = set(range(1, 101)) - S
                        swap_done = True
                        no_progress_count = 0
                        break
                if swap_done:
                    break

        if swap_done:
            continue

        # Phase 4: Try 1-for-2 exchanges (remove 1, add 2)
        swap_done2 = False
        if S and len(missing) >= 2:
            S_list = list(S)
            random.shuffle(S_list)
            for y in S_list[:min(20, len(S_list))]:
                if time.time() > deadline:
                    break
                S_test = S - {y}
                R_test = recalc_reachable(S_test)
                if R_test is None:
                    continue
                # Try adding two elements from missing
                missing_list = sorted(missing, reverse=True)
                found = False
                for i in range(min(30, len(missing_list))):
                    x1 = missing_list[i]
                    if (R_test >> (200 - x1)) & 1:
                        continue
                    R_temp = R_test | ((R_test << x1) & ((1 << 201) - 1))
                    for j in range(i+1, min(i+30, len(missing_list))):
                        x2 = missing_list[j]
                        if (R_temp >> (200 - x2)) & 1:
                            continue
                        R_new = R_temp | ((R_temp << x2) & ((1 << 201) - 1))
                        S_new = S_test | {x1, x2}
                        # Try greedy fill
                        temp_missing = missing - {x1, x2} | {y}
                        S_fill, R_fill = greedy_fill(set(S_new), R_new, set(temp_missing))
                        if len(S_fill) > len(S):
                            S = S_fill
                            R = R_fill
                            missing = set(range(1, 101)) - S
                            swap_done2 = True
                            no_progress_count = 0
                            found = True
                            break
                    if found:
                        break
                if swap_done2:
                    break

        if swap_done2:
            continue

        # Phase 5: Random restarts if stuck
        no_progress_count += 1
        if no_progress_count >= max_no_progress:
            # Try a random perturbation
            k = random.randint(1, min(4, len(S)))
            if k > 0 and S:
                to_remove = random.sample(list(S), k)
                S = S - set(to_remove)
                R = recalc_reachable(S)
                if R is None:
                    R = 1
                    for x in S:
                        R = R | ((R << x) & ((1 << 201) - 1))
                missing = set(range(1, 101)) - S
                no_progress_count = 0
                # Re-greedy fill
                S, R = greedy_fill(S, R, missing)
                missing = set(range(1, 101)) - S

    # Final polish pass: repeated full passes of additions until no more fit
    improved = True
    while improved:
        improved = False
        for x in sorted(missing, reverse=True):
            if not ((R >> (200 - x)) & 1):
                S.add(x)
                missing.discard(x)
                R = R | ((R << x) & ((1 << 201) - 1))
                improved = True

    return list(S)
