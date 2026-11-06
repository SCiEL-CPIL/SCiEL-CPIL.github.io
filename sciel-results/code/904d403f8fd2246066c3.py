"""Improved program for B_3[1] set construction."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_set = []
    best_size = 0

    def check_and_build(order, time_limit):
        """Build a valid set by greedily adding elements in the given order."""
        selected = []
        triple_sums = set()

        for x in order:
            if time.time() > time_limit:
                break

            # Generate all new triple sums involving x
            new_sums = set()

            # x + x + x
            s = 3 * x
            if s in triple_sums:
                continue
            new_sums.add(s)

            # x + x + y for y in selected
            valid = True
            for y in selected:
                s = 2 * x + y
                if s in triple_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue

            # x + y + z for y <= z in selected
            n = len(selected)
            for i in range(n):
                yi = selected[i]
                for j in range(i, n):
                    zj = selected[j]
                    s = x + yi + zj
                    if s in triple_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue

            # All new sums are valid, add x
            triple_sums.update(new_sums)
            selected.append(x)

        return selected

    # Try different strategies

    # 1. Simple ascending greedy
    order = list(range(1, 4097))
    result = check_and_build(order, deadline)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)

    if time.time() > deadline - 5:
        return best_set

    # 2. Try descending order
    order = list(range(4096, 0, -1))
    result = check_and_build(order, deadline)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)

    if time.time() > deadline - 5:
        return best_set

    # 3. Randomized greedy with multiple restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 2))  # ~2s per restart

    for _ in range(num_restarts):
        if time.time() > deadline - 1:
            break

        # Random permutation
        order = list(range(1, 4097))
        rng.shuffle(order)
        result = check_and_build(order, deadline - 1)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)

    # 4. Try biased orderings - favor certain ranges
    if time.time() < deadline - 10:
        # Try powers of 2 first, then fill in
        powers_of_2 = [2**k for k in range(0, 13) if 2**k <= 4096]
        others = [x for x in range(1, 4097) if x not in powers_of_2]
        rng.shuffle(others)
        order = powers_of_2 + others
        result = check_and_build(order, deadline - 5)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)

    # 5. Try with larger numbers first (sparse sets tend to work better)
    if time.time() < deadline - 5:
        # Sample from upper range first
        upper = list(range(2048, 4097))
        lower = list(range(1, 2049))
        rng.shuffle(upper)
        rng.shuffle(lower)
        order = upper + lower
        result = check_and_build(order, deadline - 2)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)

    # 6. Local search: try to improve the best set found
    if time.time() < deadline - 2 and best_size > 0:
        # Try removing and replacing elements
        best_set_copy = best_set[:]
        removed = set(best_set_copy)

        # Build triple sums for current best set
        def build_triple_sums(S):
            sums = set()
            n = len(S)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        sums.add(S[i] + S[j] + S[k])
            return sums

        triple_sums = build_triple_sums(best_set_copy)

        # Try to add more elements
        for x in range(1, 4097):
            if time.time() > deadline - 1:
                break
            if x in removed:
                continue

            new_sums = set()
            valid = True

            s = 3 * x
            if s in triple_sums:
                continue
            new_sums.add(s)

            for y in best_set_copy:
                s = 2 * x + y
                if s in triple_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue

            n = len(best_set_copy)
            for i in range(n):
                yi = best_set_copy[i]
                for j in range(i, n):
                    zj = best_set_copy[j]
                    s = x + yi + zj
                    if s in triple_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue

            triple_sums.update(new_sums)
            best_set_copy.append(x)
            removed.add(x)

        if len(best_set_copy) > best_size:
            best_set = best_set_copy
            best_size = len(best_size)

    return best_set[:80]  # Ensure at most 80 elements
# EVOLVE-BLOCK-END
