"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110  # safety margin before 120s kill

def construct():
    start = time.time()
    deadline = start + 110

    def all_diffs(s):
        """Return list of all positive differences from sorted list s."""
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]

    def is_sidon(s):
        """Check if sorted list s is a Sidon set (all positive differences distinct)."""
        d = all_diffs(s)
        return len(set(d)) == len(d)

    def has_diff(s, target):
        """Check if target appears as a positive difference in sorted list s."""
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[j] - s[i] == target:
                    return True
        return False

    def missing_diffs(s, max_d=30):
        """Return list of d in 1..max_d not appearing as a difference in s."""
        dset = set(all_diffs(s))
        return [d for d in range(1, max_d + 1) if d not in dset]

    def greedy_build(order):
        """
        Build a Sidon set by processing differences in the given order.
        Returns (set, max_element) or None if fails.
        """
        a = [0]
        for d in order:
            if has_diff(a, d):
                continue
            # Try to add a single element x such that x - y = d or y - x = d for some y in a
            candidates = set()
            for y in a:
                candidates.add(y + d)
                candidates.add(y - d)
            candidates = sorted([c for c in candidates if c >= 0 and c not in set(a)])

            best_trial = None
            best_max = None
            for x in candidates:
                trial = sorted(a + [x])
                if is_sidon(trial):
                    mx = max(trial)
                    if best_max is None or mx < best_max:
                        best_trial = trial
                        best_max = mx
            if best_trial is not None:
                a = best_trial
                continue

            # Try adding two elements y, y+d
            # Search for smallest y >= 0 such that {y, y+d} can be added to a
            y = max(a) + 1
            while y < 1000000:
                trial = sorted(a + [y, y + d])
                if len(trial) <= 32 and is_sidon(trial):
                    a = trial
                    break
                y += 1
            else:
                return None
        return a

    def try_construct(order):
        """Try to build a Sidon set with the given order, return None on failure."""
        if time.time() > deadline:
            return None
        return greedy_build(order)

    # Strategy 1: Try natural order 1..30
    result = try_construct(list(range(1, 31)))
    if result is not None:
        return result

    # Strategy 2: Try various orderings - start with large differences to anchor structure
    # Large diffs first: 30, 29, 28, ...
    result = try_construct(list(range(30, 0, -1)))
    if result is not None:
        return result

    # Strategy 3: Interleaved - odd then even, or prime-like ordering
    # Try: 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, then fill in
    primes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    rest = [d for d in range(1, 31) if d not in primes]
    order = primes + rest
    result = try_construct(order)
    if result is not None:
        return result

    # Strategy 4: Descending from 30 with some mixing
    # 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1
    evens_desc = list(range(30, 0, -2))
    odds_desc = list(range(29, 0, -2))
    order = evens_desc + odds_desc
    result = try_construct(order)
    if result is not None:
        return result

    # Strategy 5: Randomized restarts with different heuristics
    import random
    rng = random.Random(42)
    best_result = None
    best_max = float('inf')

    while time.time() < deadline:
        # Try random permutations with bias toward certain structures
        order = list(range(1, 31))
        rng.shuffle(order)
        result = try_construct(order)
        if result is not None:
            mx = max(result)
            if mx < best_max:
                best_max = mx
                best_result = result
                if best_max <= 150:  # good enough threshold
                    break

    if best_result is not None:
        return best_result

    # Fallback: original greedy
    a = [0]
    for d in range(1, 31):
        if has_diff(a, d):
            continue
        candidates = set()
        for y in a:
            candidates.add(y + d)
            candidates.add(y - d)
        candidates = sorted([c for c in candidates if c >= 0 and c not in set(a)])
        found = False
        for x in candidates:
            trial = sorted(a + [x])
            if is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while y < 1000000:
                trial = sorted(a + [y, y + d])
                if is_sidon(trial):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
