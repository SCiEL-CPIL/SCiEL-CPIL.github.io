import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

    target_diffs = set(range(1, 31))

    # We'll sweep M from small to large, trying to find a Sidon set with max element M
    # that includes 0 and M, and realizes all differences 1..30.
    # For each M, we use backtracking to fill in the middle elements.

    best_solution = None

    def find_set_with_max(M, deadline):
        """Find a Sidon set with max element exactly M, including 0, realizing diffs 1..30."""
        # The set must include 0 and M.
        # We need to choose intermediate elements from 1..M-1.
        # All positive differences must be distinct.
        # All differences 1..30 must appear.

        # Start with {0, M}. The difference M is already used.
        # If M <= 30, then M is one of our target diffs.
        used_diffs = {M}
        found_diffs = {M} if M in target_diffs else set()
        current_set = [0, M]

        # We need to add elements from 1 to M-1.
        # Use backtracking: try adding elements in increasing order.
        # But we can be smart: we need to cover all missing diffs.

        def backtrack(current_set, used_diffs, found_diffs, next_start):
            nonlocal best_solution
            if time.time() > deadline:
                return False

            # Check if we have a complete solution
            if found_diffs.issuperset(target_diffs):
                # Valid solution found
                best_solution = list(current_set)
                return True

            if len(current_set) >= 32:
                return False

            # Determine which diffs are still missing
            missing = target_diffs - found_diffs
            if not missing:
                return True

            # Try adding next element x starting from next_start
            # We can bound x: x < M (since M is already in set)
            # Actually, we should try x in increasing order from next_start to M-1

            for x in range(next_start, M):
                if time.time() > deadline:
                    return False

                # Check if x can be added
                valid = True
                new_found = set(found_diffs)

                for y in current_set:
                    d = abs(x - y)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    if d in target_diffs:
                        new_found.add(d)

                if not valid:
                    continue

                # If x doesn't help with any missing diff, it might still be useful
                # for positioning, but let's prioritize x that adds new diffs.
                # Actually, for correctness we should try all valid x.
                # But for speed, we can skip x that don't add any new target diff
                # if we're already behind? No, keep it simple.

                new_used = set(used_diffs)
                for y in current_set:
                    d = abs(x - y)
                    if d > 0:
                        new_used.add(d)

                # Recurse
                if backtrack(current_set + [x], new_used, new_found, x + 1):
                    return True

            return False

        # Start backtracking from x=1
        return backtrack(current_set, used_diffs, found_diffs, 1)

    # Sweep M from small to large
    # Lower bound: we need at least 30 distinct positive diffs, so M must be >= 30
    # Actually, the minimum M is at least 30 since diff 30 must appear.
    # Let's start from M=30 and go up.

    for M in range(30, 1000000):
        if time.time() > deadline:
            break

        if find_set_with_max(M, deadline):
            if best_solution is not None:
                return best_solution

    # Fallback: use a known construction or greedy
    if best_solution is not None:
        return best_solution

    # Greedy fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
