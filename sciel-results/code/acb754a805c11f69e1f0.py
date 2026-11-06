import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # Leave 25s margin for safety

    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)

    best_solution = None
    best_max = float('inf')

    def find_set_with_max(M):
        """Find a Sidon set with max element exactly M, including 0 and M, realizing diffs 1..30."""
        nonlocal best_solution, best_max

        # Start with {0, M}
        current_set = {0, M}
        used_diffs = set()
        found_mask = 0

        # Add differences from 0 and M
        used_diffs.add(M)
        if M <= 30:
            found_mask |= (1 << M)

        def backtrack(current_set, used_diffs, found_mask):
            nonlocal best_solution, best_max

            if time.time() > deadline:
                return False

            # Check if we have a complete solution
            if (found_mask & target_mask) == target_mask:
                sol = sorted(current_set)
                if max(sol) < best_max:
                    best_max = max(sol)
                    best_solution = sol
                return True

            if len(current_set) >= 32:
                return False

            # Generate candidates based on missing differences
            missing_diffs = target_diffs - {d for d in range(1, 31) if (found_mask >> d) & 1}
            if not missing_diffs:
                return True

            # Candidates: for each missing diff d, try x such that x or x-d or x+d is in current_set
            # Actually, we want x such that |x - y| = d for some y in current_set
            candidates = set()
            for d in missing_diffs:
                for y in current_set:
                    x1 = y + d
                    x2 = y - d
                    if 0 < x1 < M:
                        candidates.add(x1)
                    if 0 < x2 < M:
                        candidates.add(x2)

            # Sort candidates: prefer those that add more missing diffs
            def score_candidate(x):
                count = 0
                for y in current_set:
                    d = abs(x - y)
                    if d in missing_diffs and d not in used_diffs:
                        count += 1
                return -count  # Higher score (more diffs) comes first

            sorted_candidates = sorted(candidates, key=score_candidate)

            for x in sorted_candidates:
                if time.time() > deadline:
                    return False

                if x in current_set:
                    continue

                # Check if x can be added (no duplicate differences)
                valid = True
                new_found_mask = found_mask

                for y in current_set:
                    d = abs(x - y)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    if 1 <= d <= 30:
                        new_found_mask |= (1 << d)

                if not valid:
                    continue

                # Pruning: check if remaining capacity can cover all missing diffs
                new_missing = target_diffs - {d for d in range(1, 31) if (new_found_mask >> d) & 1}
                remaining_slots = 32 - len(current_set) - 1
                # Each new element can add at most len(current_set)+1 new diffs, but realistically
                # let's just check if we have enough elements
                if len(new_missing) > remaining_slots * (len(current_set) + 1):
                    continue

                new_used = set(used_diffs)
                for y in current_set:
                    d = abs(x - y)
                    if d > 0:
                        new_used.add(d)

                new_set = current_set | {x}

                if backtrack(new_set, new_used, new_found_mask):
                    return True

            return False

        return backtrack(current_set, used_diffs, found_mask)

    # Sweep M from small to large
    for M in range(30, 200):
        if time.time() > deadline:
            break

        if find_set_with_max(M):
            if best_solution is not None:
                return best_solution

    # Fallback: use a known construction or greedy
    if best_solution is not None:
        return best_solution

    # Greedy fallback
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in range(1, 2 * max(a) + d + 2):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    return a
