import time
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    # Greedy initialization
    best = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        for a in reversed(best):
            total += a
            tails.append(total)
        tails.append(x)
        if not sums.intersection(tails):
            best.append(x)
            sums.update(tails)

    def dfs(prefix, diffs, diff_set):
        nonlocal best
        if time.time() - start_time > TIME_LIMIT:
            return

        n = len(diffs)
        if n > len(best):
            best = list(diffs)

        if not diffs:
            min_diff = 1
            max_additional = MAX_VAL
        else:
            min_diff = diffs[-1] + 1
            max_additional = MAX_VAL - diffs[-1]

        if min_diff > MAX_VAL:
            return

        if n + max_additional <= len(best):
            return

        for d in range(min_diff, MAX_VAL + 1):
            if time.time() - start_time > TIME_LIMIT:
                return

            new_p = prefix[-1] + d

            valid = True
            new_diffs = []
            for j in range(len(prefix)):
                diff = new_p - prefix[j]
                if diff in diff_set:
                    valid = False
                    break
                new_diffs.append(diff)

            if not valid:
                continue

            prefix.append(new_p)
            diffs.append(d)
            diff_set.update(new_diffs)

            dfs(prefix, diffs, diff_set)

            prefix.pop()
            diffs.pop()
            diff_set.difference_update(new_diffs)

    dfs([0], [], set())
    return best
