import time
import random

def construct():
    deadline = time.time() + 52
    max_val = 256
    max_size = 60

    def get_bitsets(S):
        """Return list of bitsets for each cardinality level."""
        n = len(S)
        max_sum = n * 256
        mask = (1 << (max_sum + 1)) - 1
        sums = [0] * (n + 1)
        sums[0] = 1
        for x in S:
            for k in range(n, 0, -1):
                sums[k] |= (sums[k-1] << x) & mask
        return sums

    def is_valid(S):
        """Check if set S is valid (no cross-cardinality sum collisions)."""
        if not S:
            return True
        sums = get_bitsets(S)
        n = len(S)
        for k in range(n + 1):
            for j in range(k + 1, n + 1):
                if sums[k] & sums[j]:
                    return False
        return True

    def try_add_incremental(S, x, sums):
        """Try to add x to S. Returns (valid, new_S, new_sums) or (False, S, sums)."""
        if x in S:
            return False, S, sums
        n = len(S)
        new_n = n + 1
        max_sum = new_n * 256
        mask = (1 << (max_sum + 1)) - 1
        new_sums = sums[:]
        # Update in reverse order
        for k in range(n, 0, -1):
            new_sums[k] |= (sums[k-1] << x) & mask
        new_sums[new_n] = (sums[n-1] << x) & mask if n > 0 else (1 << x) & mask
        # Check collisions only for the new level and existing levels
        for k in range(new_n + 1):
            for j in range(k + 1, new_n + 1):
                if new_sums[k] & new_sums[j]:
                    return False, S, sums
        return True, S + [x], new_sums

    def greedy_build(candidates, deadline):
        """Greedily build a valid set from candidates."""
        result = []
        sums = [1]
        for x in candidates:
            if x < 1 or x > 256:
                continue
            if x in result:
                continue
            valid, new_result, new_sums = try_add_incremental(result, x, sums)
            if valid:
                result = new_result
                sums = new_sums
                if len(result) >= max_size:
                    break
            if time.time() > deadline:
                break
        return result, sums

    random.seed(42)
    best = []
    best_sums = [1]

    # Try multiple orderings
    orderings = [
        range(1, 257),
        range(256, 0, -1),
        sorted(range(1, 257), key=lambda x: x % 7),
        sorted(range(1, 257), key=lambda x: x % 13),
        sorted(range(1, 257), key=lambda x: (x % 5, x)),
        sorted(range(1, 257), key=lambda x: x * 3 % 257),
        sorted(range(1, 257), key=lambda x: x * 7 % 257),
        sorted(range(1, 257), key=lambda x: x * 11 % 257),
    ]

    for ordering in orderings:
        res, res_sums = greedy_build(ordering, deadline)
        if len(res) > len(best):
            best = res
            best_sums = res_sums
        if time.time() > deadline:
            break

    # Local search: try swaps to improve
    if len(best) < max_size and time.time() < deadline - 2:
        current_set = best[:]
        current_sums = best_sums[:]
        used = set(current_set)
        remaining = [x for x in range(1, 257) if x not in used]
        best_set = current_set[:]
        best_sums = current_sums[:]

        temperature = 50.0
        no_improve = 0

        while time.time() < deadline - 2:
            improved = False

            # Try adding a random element
            if remaining:
                for _ in range(min(10, len(remaining))):
                    idx = random.randint(0, len(remaining) - 1)
                    x = remaining[idx]
                    valid, new_set, new_sums = try_add_incremental(current_set, x, current_sums)
                    if valid:
                        current_set = new_set
                        current_sums = new_sums
                        used.add(x)
                        remaining.pop(idx)
                        if len(current_set) > len(best_set):
                            best_set = current_set[:]
                            best_sums = current_sums[:]
                            no_improve = 0
                        improved = True
                        break

            if improved:
                temperature = max(1.0, temperature * 0.98)
                continue

            # Try swaps
            if current_set and remaining:
                i = random.randint(0, len(current_set) - 1)
                old_val = current_set[i]
                j = random.randint(0, len(remaining) - 1)
                new_val = remaining[j]

                test_set = current_set[:]
                test_set[i] = new_val

                if len(set(test_set)) != len(test_set):
                    continue

                if is_valid(test_set):
                    # Accept with some probability
                    if random.random() < 0.5 or temperature > 10:
                        current_set = test_set
                        current_sums = get_bitsets(current_set)
                        used.remove(old_val)
                        used.add(new_val)
                        remaining[j] = old_val
                        no_improve += 1
                        improved = True

            if not improved:
                temperature = max(1.0, temperature * 0.95)
                no_improve += 1

            if no_improve > 100:
                # Restart from best
                current_set = best_set[:]
                current_sums = best_sums[:]
                used = set(current_set)
                remaining = [x for x in range(1, 257) if x not in used]
                no_improve = 0
                temperature = 50.0

    best = best_set
    best = [int(x) for x in best]
    return best
