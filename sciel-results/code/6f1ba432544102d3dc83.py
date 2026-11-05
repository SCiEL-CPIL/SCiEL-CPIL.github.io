"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 85.0  # 5s safety margin before 90s kill

    # Precompute masks[i] = bitmask with bits 0..i-1 set (i bits)
    masks = [0]
    m = 0
    for i in range(1, 130):
        m |= (1 << (i - 1))
        masks.append(m)

    # Precompute the sum of the last k elements for a sequence, given a running total.
    # We'll compute prefix sums of the *reversed* sequence for O(1) suffix-sum.

    def get_valid_candidates(values, sums, last_val):
        """
        Fast candidate generation using bitmask collision check.
        values: current increasing sequence
        sums: set of all existing consecutive-block sums
        last_val: last value in values (0 if empty)
        """
        n = len(values)
        valid = []
        # Precompute suffix sums of values: suffix[i] = values[i] + ... + values[n-1]
        # suffix has n+1 entries, suffix[n] = 0, suffix[i] = values[i] + suffix[i+1]
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = values[i] + suffix[i + 1]

        for x in range(last_val + 1, 257):
            # New tails: x, x+suffix[0], x+suffix[1], ..., x+suffix[n-1]
            # tails[j] = x + suffix[j] for j in 0..n-1
            # Collision if any tail is in sums
            # Bitmask approach: we need to check if any (x + suffix[j]) is in sums.
            # sums is a set; for small n (<= ~128), we can use a bitmask over the sum range.
            # Sum range: max possible sum = 256 * 128 = 32768, which is too large for a bitmask.
            # Alternative: since n is small, just check each tail against the set.
            # But we can optimize: build a set of tails and check intersection.
            # For n up to ~128, building a set of n elements and checking intersection with sums
            # is O(n) per candidate, and there are at most 256 candidates, so O(256 * n) per step.
            # This is acceptable.

            # Quick check: if x itself is in sums, skip
            if x in sums:
                continue
            # Check all tails
            collision = False
            for j in range(n):
                if (x + suffix[j]) in sums:
                    collision = True
                    break
            if not collision:
                valid.append(x)

        return valid

    def add_sums(values, sums, x):
        """Add all new consecutive-block sums ending at x to sums."""
        total = x
        sums.add(total)
        for a in reversed(values):
            total += a
            sums.add(total)

    def greedy_increasing():
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            x = candidates[0]  # smallest
            values.append(x)
            add_sums(values[:-1], sums, x)
            last_val = x
        return values

    def greedy_decreasing():
        """Build decreasing then reverse."""
        values = []
        sums = set()
        next_val = 257
        while next_val > 1:
            candidates = []
            n = len(values)
            if n > 0:
                suffix = [0] * (n + 1)
                for i in range(n - 1, -1, -1):
                    suffix[i] = values[i] + suffix[i + 1]
            for x in range(next_val - 1, 0, -1):
                if x in sums:
                    continue
                collision = False
                for j in range(n):
                    if (x + suffix[j]) in sums:
                        collision = True
                        break
                if not collision:
                    candidates.append(x)
            if not candidates:
                break
            x = candidates[0]  # largest (since we iterated down)
            values.append(x)
            add_sums(values[:-1], sums, x)
            next_val = x
        return values[::-1]

    def randomized_trial(strategy, rand, values_init=None):
        if values_init is not None:
            values = list(values_init)
            sums = set()
            # Rebuild sums
            for i, v in enumerate(values):
                total = v
                sums.add(total)
                for a in reversed(values[:i]):
                    total += a
                    sums.add(total)
            last_val = values[-1] if values else 0
        else:
            values = []
            sums = set()
            last_val = 0

        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break

            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                k = min(10, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                k = min(10, len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'median':
                x = candidates[len(candidates) // 2]
            elif strategy == 'uniform':
                x = rand.choice(candidates)
            else:
                x = rand.choice(candidates)

            values.append(x)
            add_sums(values[:-1], sums, x)
            last_val = x

        return values

    def local_search(base_seq, rand, time_deadline):
        """
        Try to extend base_seq by modifying the last few elements.
        """
        best = list(base_seq)
        best_len = len(best)

        for _ in range(200):
            if time.time() > time_deadline:
                break
            # Pick a position to "restart" from
            restart_pos = max(0, len(best) - rand.randint(1, min(5, len(best))))
            prefix = best[:restart_pos]

            # Rebuild sums for prefix
            sums = set()
            for i, v in enumerate(prefix):
                total = v
                sums.add(total)
                for a in reversed(prefix[:i]):
                    total += a
                    sums.add(total)

            last_val = prefix[-1] if prefix else 0
            values = list(prefix)

            while last_val < 256:
                candidates = get_valid_candidates(values, sums, last_val)
                if not candidates:
                    break
                # Random choice
                x = rand.choice(candidates)
                values.append(x)
                add_sums(values[:-1], sums, x)
                last_val = x

            if len(values) > best_len:
                best = values
                best_len = len(values)

        return best

    def dfs_construct(target_len, time_deadline):
        """
        DFS to find a sequence of length >= target_len.
        """
        best_seq = []

        def dfs(values, sums, last_val, depth):
            nonlocal best_seq
            if time.time() > time_deadline:
                return
            if len(values) > len(best_seq):
                best_seq = values[:]
            if len(values) >= target_len:
                return

            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                return

            # Prune
            if len(values) + (256 - last_val) < len(best_seq):
                return

            for x in candidates:
                if len(values) + (256 - x) < len(best_seq):
                    break

                new_sums = set(sums)
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)

                dfs(values + [x], new_sums, x, depth + 1)

                if time.time() > time_deadline:
                    return

        dfs([], set(), 0, 0)
        return best_seq

    # 1. Baselines
    best_seq = greedy_increasing()
    best_len = len(best_seq)

    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)

    # 2. Randomized Search - many trials
    strategies = ['random', 'large', 'small', 'median', 'uniform']
    trial_count = 0
    while time.time() - start_time < time_limit - 10.0:
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count * 7919 + 13)
        seq = randomized_trial(strategy, rand)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        trial_count += 1

    # 3. Local search from best
    if time.time() - start_time < time_limit - 5.0:
        rand = random.Random(12345)
        improved = local_search(best_seq, rand, time.time() + (time_limit - (time.time() - start_time)) - 5.0)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)

    # 4. DFS if time permits
    if time.time() - start_time < time_limit - 2.0:
        dfs_deadline = time.time() + (time_limit - (time.time() - start_time)) - 2.0
        dfs_seq = dfs_construct(best_len + 1, dfs_deadline)
        if len(dfs_seq) > best_len:
            best_seq = dfs_seq
            best_len = len(best_seq)

    return best_seq

# EVOLVE-BLOCK-END
