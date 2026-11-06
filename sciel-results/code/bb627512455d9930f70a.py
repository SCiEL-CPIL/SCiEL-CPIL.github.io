"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 120.0

    # ---- helper functions ----
    def all_pos_diffs(s):
        s = sorted(s)
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]

    def is_sidon(s):
        s = sorted(s)
        seen = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def covers_1_to_30(s):
        s = sorted(s)
        diff_set = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diff_set.add(s[j] - s[i])
        for d in range(1, 31):
            if d not in diff_set:
                return False
        return True

    def valid_solution(s):
        return (0 in s and len(s) <= 32 and all(x >= 0 and x <= 1000000 for x in s)
                and len(set(s)) == len(s) and is_sidon(s) and covers_1_to_30(s))

    # ---- Phase 1: randomized constructive search with diameter bound ----
    best = None
    best_max = 1000000

    def try_random_construct(max_elem):
        """Try to build a Sidon set covering 1..30 with max element <= max_elem."""
        nonlocal best, best_max, deadline
        if time.time() > deadline:
            return None

        # Start with 0, then try to add elements randomly
        # We need at least enough elements to cover 30 differences
        # n elements give n*(n-1)/2 differences, so we need n*(n-1)/2 >= 30 -> n >= 9
        # But we also need the specific differences 1..30

        # Strategy: random restarts, build up by adding random elements in [1, max_elem]
        # and check if we can complete the coverage

        # Use a more structured approach: try to place elements to cover missing diffs
        missing = set(range(1, 31))
        a = [0]

        # Try several random strategies
        for attempt in range(200):
            if time.time() > deadline:
                return None
            if not missing:
                break

            a = [0]
            remaining = set(range(1, 31))

            # Greedy with randomness: for each missing diff, try to add an element
            for d in sorted(remaining, key=lambda x: random.random()):
                if d not in remaining:
                    continue
                if d > max_elem:
                    return None
                found = False
                # Try to add a single element x such that x - 0 = d or 0 - (x-d) ... 
                # Actually we need x - y = d for some y in a, or we add x such that x - a[i] = d for some i
                # Or we could add x such that a[i] - x = d, but x >= 0
                # Simplest: try x = d (so 0 and d give diff d), or x = a[i] + d, or x = a[i] - d
                candidates = set()
                for v in a:
                    candidates.add(v + d)
                    if v - d >= 0:
                        candidates.add(v - d)
                # Also try x = d directly
                candidates.add(d)
                # Filter: must be in [1, max_elem], not already in a
                candidates = {c for c in candidates if 1 <= c <= max_elem and c not in a}
                # Shuffle for randomness
                clist = list(candidates)
                random.shuffle(clist)
                for x in clist:
                    trial = sorted(a + [x])
                    if len(trial) > 32:
                        continue
                    if is_sidon(trial):
                        a = trial
                        # Update remaining
                        new_diffs = set(all_pos_diffs(a))
                        remaining -= new_diffs
                        found = True
                        break
                if not found:
                    # Try adding two elements to cover this diff
                    # x + d - x = d is trivial; we need x and y such that y - x = d
                    # Try x random, y = x + d
                    for _ in range(50):
                        x = random.randint(0, max_elem - d)
                        y = x + d
                        if x in a or y in a or x == y:
                            continue
                        trial = sorted(a + [x, y])
                        if len(trial) > 32:
                            continue
                        if is_sidon(trial):
                            a = trial
                            new_diffs = set(all_pos_diffs(a))
                            remaining -= new_diffs
                            found = True
                            break
                if not found:
                    a = None
                    break

            if a is not None and not remaining:
                if max(a) < best_max:
                    best_max = max(a)
                    best = a[:]
                    if time.time() > deadline:
                        return best
                return a

        return a

    # Try progressively smaller diameters
    # Theoretical lower bound: we need 30 distinct positive diffs from n elements
    # Max possible diffs from n elements is n*(n-1)/2, so n >= 9
    # With n elements in [0, M], the max diff is M, and we need diffs 1..30, so M >= 30
    # But Sidon constraint makes it much larger.
    # Known: for a Sidon set covering 1..30, the minimal diameter is around 50-70
    # Let's try from small to large

    for max_elem in [30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120, 150, 200, 300, 500, 1000, 5000, 10000]:
        if time.time() > deadline:
            break
        result = try_random_construct(max_elem)
        if result is not None and valid_solution(result):
            if max(result) < best_max:
                best_max = max(result)
                best = result[:]
            # If we found one, try to improve with local optimization
            if best is not None and time.time() < deadline:
                best = local_shrink(best, deadline)
                if best is not None and valid_solution(best) and max(best) < best_max:
                    best_max = max(best)
                    best = best[:]
        # If we found a solution, we can try tighter bounds
        if best is not None and max_elem <= 100:
            # Try even tighter
            tighter = max_elem - 5
            if tighter >= 30:
                result2 = try_random_construct(tighter)
                if result2 is not None and valid_solution(result2):
                    if max(result2) < best_max:
                        best_max = max(result2)
                        best = result2[:]
                        if time.time() < deadline:
                            best = local_shrink(best, deadline)
                            if best is not None and valid_solution(best) and max(best) < best_max:
                                best_max = max(best)
                                best = best[:]

    # ---- Phase 2: local optimization (shrink) ----
    def local_shrink(solution, dl):
        """Try to reduce the max element by removing/replacing elements."""
        nonlocal best, best_max
        if solution is None:
            return None
        sol = sorted(solution[:])
        improved = True
        while improved and time.time() < dl:
            improved = False
            # Try removing an element
            for i in range(len(sol)):
                if time.time() > dl:
                    break
                if i == 0 and sol[i] == 0:
                    continue  # must keep 0
                trial = [x for j, x in enumerate(sol) if j != i]
                if len(trial) < 9:
                    continue
                if is_sidon(trial) and covers_1_to_30(trial):
                    sol = trial
                    improved = True
                    break
            if not improved and time.time() < dl:
                # Try replacing the max element with a smaller one
                if len(sol) > 1:
                    mx = max(sol)
                    idx = sol.index(mx)
                    # Try replacing with smaller values
                    for new_val in range(mx - 1, 0, -1):
                        if time.time() > dl:
                            break
                        if new_val in sol:
                            continue
                        trial = sorted([x if j != idx else new_val for j, x in enumerate(sol)])
                        if len(set(trial)) != len(trial):
                            continue
                        if is_sidon(trial) and covers_1_to_30(trial):
                            sol = trial
                            improved = True
                            break
        return sol

    # ---- Final result ----
    if best is not None and valid_solution(best):
        return best

    # Fallback: greedy construction
    a = [0]
    for d in range(1, 31):
        cur_diffs = set(all_pos_diffs(a))
        if d in cur_diffs:
            continue
        found = False
        candidates = set()
        for v in a:
            candidates.add(v + d)
            if v - d >= 0:
                candidates.add(v - d)
        candidates = {c for c in candidates if c >= 0 and c not in a}
        for x in sorted(candidates):
            trial = sorted(a + [x])
            if len(trial) <= 32 and is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(trial) <= 32 and is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
