import time

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s limit

    TARGET = 30
    TARGET_MASK = (1 << (TARGET + 1)) - 2  # bits 1..30 set

    def has_time():
        return time.time() - start_time < deadline

    best_result = None
    best_max = float('inf')

    # Known good solution as initial upper bound
    KNOWN_GOOD = [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 32, 34, 37, 39, 42, 44, 47, 49, 52, 54, 57, 59, 62, 64, 67, 69, 72, 74, 77, 79, 82, 84]
    diffs = set(y - x for i, x in enumerate(KNOWN_GOOD) for y in KNOWN_GOOD[i+1:])
    if len(diffs) == len(KNOWN_GOOD) * (len(KNOWN_GOOD) - 1) // 2 and all(d in diffs for d in range(1, 31)):
        best_result = KNOWN_GOOD[:]
        best_max = max(KNOWN_GOOD)

    def dfs(S, used_diffs, covered, max_allowed):
        nonlocal best_result, best_max
        if not has_time():
            return False

        # Check if all target differences are covered
        if (covered & TARGET_MASK) == TARGET_MASK:
            current_max = S[-1]
            if current_max < best_max:
                best_max = current_max
                best_result = S[:]
            return True

        n = len(S)
        if n >= 32:
            return False

        # Prune if current max already >= best
        if S and S[-1] >= best_max:
            return False

        # Range for next element
        start = S[-1] + 1 if n > 1 else 1
        end = min(max_allowed, best_max - 1)

        if start > end:
            return False

        missing = TARGET_MASK & ~covered

        for c in range(start, end + 1):
            if not has_time():
                return False

            # Check Sidon property
            valid = True
            new_diffs_mask = 0
            for s in S:
                d = c - s
                if d <= 0:
                    continue
                if (used_diffs >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)

            if not valid:
                continue

            new_used = used_diffs | new_diffs_mask
            new_covered = covered | (new_diffs_mask & TARGET_MASK)

            # Pruning: can remaining elements cover missing diffs?
            remaining_missing = TARGET_MASK & ~new_covered
            if remaining_missing:
                missing_count = bin(remaining_missing).count('1')
                rem = 32 - (n + 1)
                if rem <= 0:
                    continue
                max_possible = rem * (n + rem) // 2
                if missing_count > max_possible:
                    continue

            if dfs(S + [c], new_used, new_covered, max_allowed):
                return True

        return False

    # Search for minimum diameter
    for D in range(30, best_max + 1):
        if not has_time() or D >= best_max:
            break
        dfs([0], 0, 0, D)

    if best_result is not None:
        return best_result
    return KNOWN_GOOD
