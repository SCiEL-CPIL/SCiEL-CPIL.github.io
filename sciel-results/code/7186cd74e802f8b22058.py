import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill limit

    N = 1000

    def verify_set(s):
        """Verify B2[2] property: every sum has at most 2 representations a+b with a<=b."""
        counts = defaultdict(int)
        n = len(s)
        for i in range(n):
            counts[2 * s[i]] += 1
            for j in range(i + 1, n):
                counts[s[i] + s[j]] += 1
        return max(counts.values()) <= 2

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
            for rem in chosen[:]:
                if time.time() > deadline:
                    break

                temp_counts = defaultdict(int)
                temp_set = chosen_set - {rem}
                temp_chosen = [x for x in chosen if x != rem]
                for i, x in enumerate(temp_chosen):
                    temp_counts[2 * x] += 1
                    for j in range(i + 1, len(temp_chosen)):
                        temp_counts[x + temp_chosen[j]] += 1

                found = False
                for y in range(1, N + 1):
                    if y in temp_set:
                        continue
                    if y == rem:
                        continue
                    valid = True
                    if temp_counts.get(2 * y, 0) >= 2:
                        valid = False
                    else:
                        for z in temp_chosen:
                            if temp_counts.get(y + z, 0) >= 2:
                                valid = False
                                break

                    if valid:
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

    def probabilistic_construct(seed, p):
        """
        Randomly select each integer 1..N with probability p, then greedily repair
        any sum that exceeds two representations by removing one contributor.
        """
        random.seed(seed)
        # Randomly select elements
        S = [x for x in range(1, N + 1) if random.random() < p]
        S_set = set(S)

        # Build sum counts
        counts = defaultdict(int)
        for i in range(len(S)):
            counts[2 * S[i]] += 1
            for j in range(i + 1, len(S)):
                counts[S[i] + S[j]] += 1

        # Greedily repair: find sums with >2 reps, remove a contributor
        max_iters = 10000
        for _ in range(max_iters):
            if time.time() > deadline:
                break

            # Find a violating sum
            violated_sum = None
            for s, c in counts.items():
                if c > 2:
                    violated_sum = s
                    break

            if violated_sum is None:
                break

            # Find all pairs (a,b) with a<=b in S such that a+b = violated_sum
            pairs = []
            for a in S:
                b = violated_sum - a
                if b >= a and b in S_set:
                    pairs.append((a, b))

            if not pairs:
                # Shouldn't happen, but skip
                break

            # Choose which element to remove: the one that appears in most pairs
            # or randomly among contributors
            contributor_counts = defaultdict(int)
            for a, b in pairs:
                contributor_counts[a] += 1
                if a != b:
                    contributor_counts[b] += 1

            # Remove the contributor with highest count (breaks most pairs)
            rem = max(contributor_counts.items(), key=lambda x: x[1])[0]

            # Remove rem from S
            S_set.remove(rem)
            S.remove(rem)

            # Update counts
            for x in S:
                s = rem + x
                if s in counts:
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
            if 2 * rem in counts:
                counts[2 * rem] -= 1
                if counts[2 * rem] == 0:
                    del counts[2 * rem]

        # Try to add back elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if x not in S_set and can_add(x, S_set, counts):
                    S.append(x)
                    S_set.add(x)
                    counts[2 * x] += 1
                    for y in S:
                        if y != x:
                            counts[x + y] += 1
                    changed = True
                    break

        return S, counts, S_set

    def kick_and_refill(chosen, counts, chosen_set, kick_size, seed):
        """Remove kick_size random elements, then greedily refill."""
        random.seed(seed)
        chosen = list(chosen)
        chosen_set = set(chosen)

        # Remove kick_size random elements
        to_remove = random.sample(chosen, min(kick_size, len(chosen)))
        for rem in to_remove:
            chosen_set.remove(rem)
            chosen.remove(rem)

        # Rebuild counts
        counts = defaultdict(int)
        for i, x in enumerate(chosen):
            counts[2 * x] += 1
            for j in range(i + 1, len(chosen)):
                counts[x + chosen[j]] += 1

        # Greedily refill
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

                score = tight + random.random() * 0.1
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

    best_set = []
    best_size = 0

    # Phase 1: Deterministic greedy with various seeds
    for seed in range(15):
        if time.time() > deadline:
            break
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=0.0)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)

    # Phase 2: Probabilistic constructions with repair
    for p in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for seed in range(5):
            if time.time() > deadline:
                break
            S, counts, S_set = probabilistic_construct(seed * 1000 + int(p * 100), p)
            if len(S) > best_size:
                best_size = len(S)
                best_set = list(S)

    # Phase 3: Local search on best found
    if best_set and time.time() < deadline:
        counts = defaultdict(int)
        chosen_set = set(best_set)
        for i, x in enumerate(best_set):
            counts[2 * x] += 1
            for j in range(i + 1, len(best_set)):
                counts[x + best_set[j]] += 1
        best_set, counts, chosen_set = local_search(best_set, counts, chosen_set, max_iters=300)
        best_size = len(best_set)

    # Phase 4: Kick-and-refill to escape local optima
    if best_set and time.time() < deadline:
        for kick_size in [2, 3, 5]:
            for seed in range(10):
                if time.time() > deadline:
                    break
                S, counts, S_set = kick_and_refill(best_set, counts, chosen_set, kick_size, seed * 777)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)

    # Phase 5: More randomized greedy restarts with local search
    while time.time() < deadline:
        seed = random.randint(0, 100000)
        noise = random.choice([0.0, 0.1, 0.3, 0.5])
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=noise)

        if len(chosen) >= best_size - 3 and time.time() < deadline:
            counts = defaultdict(int)
            chosen_set = set(chosen)
            for i, x in enumerate(chosen):
                counts[2 * x] += 1
                for j in range(i + 1, len(chosen)):
                    counts[x + chosen[j]] += 1
            chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_iters=80)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)

    # Final verification
    if best_set and not verify_set(best_set):
        # Fallback: rebuild from scratch if verification fails
        chosen, counts, chosen_set = greedy_build(seed=42, noise_scale=0.0)
        best_set = chosen
        best_size = len(chosen)

    return best_set
