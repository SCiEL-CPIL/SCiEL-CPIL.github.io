import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill limit

    N = 1000

    def greedy_build(seed=42, noise_scale=0.0):
        """Fast greedy: pick element minimizing sum-count violations, with noise."""
        random.seed(seed)
        chosen = []
        counts = defaultdict(int)
        chosen_set = set()

        while True:
            if time.time() > deadline:
                break

            best_x = None
            best_score = float('inf')
            best_sums = None

            for x in range(1, N + 1):
                if x in chosen_set:
                    continue

                valid = True
                tight = 0
                sums_list = []

                # Check 2*x
                s2 = 2 * x
                c2 = counts.get(s2, 0)
                if c2 >= 2:
                    valid = False
                else:
                    if c2 == 1:
                        tight += 1
                    sums_list.append((s2, c2))

                if not valid:
                    continue

                # Check x + y for y in chosen
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                    sums_list.append((s, c))

                if not valid:
                    continue

                score = tight + random.random() * noise_scale
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_sums = sums_list

            if best_x is None:
                break

            chosen.append(best_x)
            chosen_set.add(best_x)
            for s, c in best_sums:
                counts[s] = c + 1

        return chosen, counts, chosen_set

    def can_add(x, chosen_set, counts):
        """Check if x can be added to the current set."""
        if x in chosen_set:
            return False
        s2 = 2 * x
        if counts.get(s2, 0) >= 2:
            return False
        for y in chosen_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def local_search(chosen, counts, chosen_set, max_iters=500):
        """Improve set by trying swaps: remove one element, add another if possible."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        # Rebuild counts from scratch to be safe
        counts = defaultdict(int)
        for i, x in enumerate(chosen):
            counts[2 * x] += 1
            for j in range(i + 1, len(chosen)):
                counts[x + chosen[j]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Try to add any missing element
            for x in range(1, N + 1):
                if can_add(x, chosen_set, counts):
                    chosen.append(x)
                    chosen_set.add(x)
                    counts[2 * x] += 1
                    for y in chosen:
                        if y != x:
                            counts[x + y] += 1
                    improved = True
                    break

            if improved:
                continue

            # No simple addition; try swaps
            # Try removing each element and see if we can add a different one
            for rem in chosen[:]:
                if time.time() > deadline:
                    break

                # Build temp counts without rem
                temp_counts = defaultdict(int)
                temp_set = chosen_set - {rem}
                temp_chosen = [x for x in chosen if x != rem]
                for i, x in enumerate(temp_chosen):
                    temp_counts[2 * x] += 1
                    for j in range(i + 1, len(temp_chosen)):
                        temp_counts[x + temp_chosen[j]] += 1

                # Try to find an element y not in temp_set that can be added
                found = False
                for y in range(1, N + 1):
                    if y in temp_set:
                        continue
                    if y == rem:
                        continue
                    # Check if y can be added to temp set
                    valid = True
                    if temp_counts.get(2 * y, 0) >= 2:
                        valid = False
                    else:
                        for z in temp_chosen:
                            if temp_counts.get(y + z, 0) >= 2:
                                valid = False
                                break

                    if valid:
                        # Perform the swap
                        # Update main counts: remove rem's contributions, add y's
                        # Remove rem
                        for z in chosen:
                            if z == rem:
                                counts[2 * rem] -= 1
                                if counts[2 * rem] == 0:
                                    del counts[2 * rem]
                            else:
                                s = rem + z
                                counts[s] -= 1
                                if counts[s] == 0:
                                    del counts[s]

                        # Remove rem from structures
                        chosen.remove(rem)
                        chosen_set.remove(rem)

                        # Add y
                        chosen.append(y)
                        chosen_set.add(y)
                        counts[2 * y] = counts.get(2 * y, 0) + 1
                        for z in chosen:
                            if z != y:
                                s = y + z
                                counts[s] = counts.get(s, 0) + 1

                        improved = True
                        found = True
                        break

                if found:
                    break

        return chosen, counts, chosen_set

    best_set = []
    best_size = 0

    # Phase 1: Deterministic greedy with various seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=0.0)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 2: Local search on best found
    if best_set and time.time() < deadline:
        # Rebuild counts for best_set
        counts = defaultdict(int)
        chosen_set = set(best_set)
        for i, x in enumerate(best_set):
            counts[2 * x] += 1
            for j in range(i + 1, len(best_set)):
                counts[x + best_set[j]] += 1
        best_set, counts, chosen_set = local_search(best_set, counts, chosen_set, max_iters=200)
        best_size = len(best_set)

    # Phase 3: Randomized greedy restarts with local search
    while time.time() < deadline:
        seed = random.randint(0, 100000)
        noise = random.choice([0.0, 0.1, 0.3, 0.5, 1.0])
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=noise)

        # Run local search if it looks promising
        if len(chosen) >= best_size - 3 and time.time() < deadline:
            # Rebuild counts
            counts = defaultdict(int)
            chosen_set = set(chosen)
            for i, x in enumerate(chosen):
                counts[2 * x] += 1
                for j in range(i + 1, len(chosen)):
                    counts[x + chosen[j]] += 1
            chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_iters=100)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    return best_set
