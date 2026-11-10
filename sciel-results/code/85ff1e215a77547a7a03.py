import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    deadline = time.time() + 115.0  # Safety margin before 120s

    def can_add_fast(chosen, counts, x, chosen_set):
        """Fast check if x can be added."""
        if x in chosen_set:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        return True

    def add_element(chosen, counts, chosen_set, x):
        """Add x to the set, updating counts."""
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        chosen_set.add(x)

    def remove_element(chosen, counts, chosen_set, x):
        """Remove x from the set, updating counts."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
        chosen_set.discard(x)

    def greedy_build(order='random', seed=None):
        """Build a set using greedy in a given order."""
        rng = random.Random(seed) if seed is not None else random
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)

        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)

        for x in candidates:
            if can_add_fast(chosen, counts, x, chosen_set):
                add_element(chosen, counts, chosen_set, x)

        return chosen, counts, chosen_set

    def fast_local_search(chosen, counts, chosen_set, time_limit=10.0):
        """
        Fast local search using pre-computed conflict analysis.
        Key insight: for each element x, precompute which sums would be
        violated if x were removed, then quickly test if a replacement y
        fits in the "freed" space.
        """
        start_time = time.time()
        improved = True
        max_rounds = 50

        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(chosen)

            # Phase 1: Try pure insertions
            # Find all y that can be added
            for y in range(1, N + 1):
                if y in chosen_set:
                    continue
                if counts[2 * y] + 1 > 2:
                    continue
                ok = True
                for x in chosen:
                    if counts[x + y] + 1 > 2:
                        ok = False
                        break
                if ok:
                    add_element(chosen, counts, chosen_set, y)
                    improved = True
                    break

            if improved or (time.time() - start_time) >= time_limit:
                continue

            # Phase 2: Try swaps - remove one, add one
            # For efficiency, precompute for each x which y's are "almost" addable
            # (i.e., would be addable if x weren't there)
            chosen_arr = np.array(chosen)

            for i in range(n):
                if (time.time() - start_time) >= time_limit:
                    break
                x = chosen[i]

                # Temporarily remove x
                for j in range(n):
                    if j != i:
                        counts[x + chosen[j]] -= 1
                counts[2 * x] -= 1
                chosen_set.discard(x)

                # Find a y that can now be added
                found = False
                # Try candidates in a smart order: those that were blocked by x
                # First try numbers near x, then random
                candidates_to_try = []
                # Numbers that conflict with x's sums
                for s in range(2 * x, min(2 * x + n * 5, max_sum + 1)):
                    for y in range(1, N + 1):
                        if y not in chosen_set and y != x:
                            if x + y == s or 2 * y == s:
                                candidates_to_try.append(y)
                                break

                # Add some random candidates
                for _ in range(20):
                    y = random.randint(1, N)
                    if y not in chosen_set:
                        candidates_to_try.append(y)

                # Deduplicate and try
                tried = set()
                for y in candidates_to_try:
                    if y in tried or y in chosen_set:
                        continue
                    tried.add(y)

                    if counts[2 * y] + 1 > 2:
                        continue
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        if counts[chosen[j] + y] + 1 > 2:
                            ok = False
                            break
                    if ok:
                        # Do the swap
                        for j in range(n):
                            if j != i:
                                counts[chosen[j] + y] += 1
                        counts[2 * y] += 1
                        chosen[i] = y
                        chosen_set.add(y)
                        improved = True
                        found = True
                        break

                if not found:
                    # Restore x
                    for j in range(n):
                        if j != i:
                            counts[x + chosen[j]] += 1
                    counts[2 * x] += 1
                    chosen_set.add(x)
                    chosen[i] = x

                if improved:
                    break

            # Phase 3: Try 2-for-1 swaps (remove 2, add 2) for more flexibility
            if not improved and (time.time() - start_time) < time_limit:
                for i in range(n):
                    if (time.time() - start_time) >= time_limit:
                        break
                    for j in range(i + 1, n):
                        if (time.time() - start_time) >= time_limit:
                            break
                        x1 = chosen[i]
                        x2 = chosen[j]

                        # Remove both
                        for k in range(n):
                            if k != i and k != j:
                                counts[x1 + chosen[k]] -= 1
                                counts[x2 + chosen[k]] -= 1
                        counts[2 * x1] -= 1
                        counts[2 * x2] -= 1
                        if i != j:
                            counts[x1 + x2] -= 1  # Wait, this was already counted
                        chosen_set.discard(x1)
                        chosen_set.discard(x2)

                        # Try to add two new elements
                        added = []
                        for y in range(1, N + 1):
                            if y in chosen_set or y in added:
                                continue
                            if counts[2 * y] + 1 > 2:
                                continue
                            ok = True
                            for k in range(n):
                                if k != i and k != j:
                                    if counts[chosen[k] + y] + 1 > 2:
                                        ok = False
                                        break
                            if ok:
                                # Check against already-added in this round
                                for ya in added:
                                    if counts[ya + y] + 1 > 2:
                                        ok = False
                                        break
                                if ok:
                                    added.append(y)
                                    if len(added) == 2:
                                        break

                        if len(added) == 2:
                            # Apply the swap
                            for y in added:
                                for k in range(n):
                                    if k != i and k != j:
                                        counts[chosen[k] + y] += 1
                                counts[2 * y] += 1
                            counts[added[0] + added[1]] += 1
                            chosen[i] = added[0]
                            chosen[j] = added[1]
                            chosen_set.add(added[0])
                            chosen_set.add(added[1])
                            improved = True
                            break
                        else:
                            # Restore
                            for k in range(n):
                                if k != i and k != j:
                                    counts[x1 + chosen[k]] += 1
                                    counts[x2 + chosen[k]] += 1
                            counts[2 * x1] += 1
                            counts[2 * x2] += 1
                            counts[x1 + x2] += 1
                            chosen_set.add(x1)
                            chosen_set.add(x2)
                            chosen[i] = x1
                            chosen[j] = x2

                        if improved:
                            break

        return chosen, counts, chosen_set

    def structured_build(seed_val=0):
        """Build using structured modular arithmetic patterns."""
        rng = random.Random(seed_val)
        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)

        # Start with a sparse base pattern
        # Use numbers that are congruent to 1 or 2 mod 5 (reduces collisions)
        base_candidates = [x for x in range(1, N + 1) if x % 5 in (1, 2)]
        rng.shuffle(base_candidates)

        for x in base_candidates:
            if can_add_fast(chosen, counts, x, chosen_set):
                add_element(chosen, counts, chosen_set, x)

        # Fill in remaining numbers in random order
        remaining = [x for x in range(1, N + 1) if x not in chosen_set]
        rng.shuffle(remaining)
        for x in remaining:
            if can_add_fast(chosen, counts, x, chosen_set):
                add_element(chosen, counts, chosen_set, x)

        return chosen, counts, chosen_set

    best_set = []
    best_len = 0

    # Initial builds with different strategies
    strategies = [
        ('ascending', None),
        ('descending', None),
        ('random', 42),
        ('random', 123),
        ('sparse', 7),
        ('random', 999),
    ]

    for order, seed in strategies:
        if time.time() > deadline - 20:
            break
        chosen, counts, chosen_set = greedy_build(order, seed)
        chosen, counts, chosen_set = fast_local_search(chosen, counts, chosen_set, time_limit=5.0)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]

    # Structured builds
    for s in range(5):
        if time.time() > deadline - 15:
            break
        chosen, counts, chosen_set = structured_build(s)
        chosen, counts, chosen_set = fast_local_search(chosen, counts, chosen_set, time_limit=8.0)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]

    # More random restarts with local search
    attempt = 0
    while time.time() < deadline - 10:
        order = random.choice(['random', 'random', 'sparse'])
        seed = attempt * 7919 + 12345
        chosen, counts, chosen_set = greedy_build(order, seed)

        time_left = deadline - time.time()
        ls_time = min(10.0, max(2.0, time_left - 5.0))
        chosen, counts, chosen_set = fast_local_search(chosen, counts, chosen_set, time_limit=ls_time)

        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]

        attempt += 1

    return best_set
