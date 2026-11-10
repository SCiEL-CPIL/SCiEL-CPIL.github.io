import time
import random

def construct():
    deadline = time.time() + 85

    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses bitset DP."""
        n = len(others)
        if n < 2:
            return False

        max_sum = sum(others)
        max_s = min(max_sum, n * element)

        dp = [0] * (n + 1)
        dp[0] = 1  # bit 0 set: sum 0 with 0 elements

        for val in others:
            for j in range(n, 0, -1):
                if dp[j - 1]:
                    dp[j] |= (dp[j - 1] << val)
                    dp[j] &= (1 << (max_s + 1)) - 1

        for k in range(2, n + 1):
            target = k * element
            if target <= max_s and (dp[k] >> target) & 1:
                return True

        return False

    def can_add(element, current):
        """Check if element can be added to current set."""
        # Check 1: is element the mean of any subset of current?
        if is_mean_of_subset(element, current):
            return False

        # Check 2: for each a in current, does adding element create a violation?
        # a is mean of subset S of (current \ {a}) ∪ {element} with |S| >= 2
        # If element in S: S = T ∪ {element} where T ⊆ current \ {a}, |T| >= 1
        #   sum(T) + element = (|T|+1) * a
        #   sum(T) = (|T|+1)*a - element
        for a in current:
            others_without_a = [x for x in current if x != a]
            n = len(others_without_a)
            if n < 1:
                continue

            max_s = sum(others_without_a)
            dp = [0] * (n + 1)
            dp[0] = 1

            for val in others_without_a:
                for j in range(n, 0, -1):
                    if dp[j - 1]:
                        dp[j] |= (dp[j - 1] << val)
                        dp[j] &= (1 << (max_s + 1)) - 1

            for j in range(1, n + 1):
                target = (j + 1) * a - element
                if 0 <= target <= max_s and (dp[j] >> target) & 1:
                    return False

        return True

    def is_valid_set(s):
        """Verify the entire set is valid."""
        for i, elem in enumerate(s):
            others = s[:i] + s[i+1:]
            if is_mean_of_subset(elem, others):
                return False
        return True

    # Diverse orderings
    all_nums = list(range(1, 513))

    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    powers_of_2 = [2**i for i in range(1, 10) if 2**i <= 512]
    non_powers = [x for x in range(1, 513) if x not in set(powers_of_2)]

    # Alternating small/large
    alt = []
    for i in range(256):
        alt.append(i + 1)
        alt.append(512 - i)

    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        powers_of_2 + non_powers,      # powers of 2 first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        [x for x in range(1, 513) if x % 5 != 0] + [x for x in range(1, 513) if x % 5 == 0],
        primes + non_primes,           # primes first
        alt,                           # alternating
        list(range(512, 256, -1)) + list(range(256, 0, -1)),
    ]

    rng = random.Random(42)
    for _ in range(8):
        order = list(range(1, 513))
        rng.shuffle(order)
        orderings.append(order)

    best_set = []

    for ordering in orderings:
        if time.time() > deadline - 10:
            break

        selected = []
        for x in ordering:
            if time.time() > deadline - 10:
                break
            if len(selected) >= 48:
                break
            if can_add(x, selected):
                selected.append(x)

        if len(selected) > len(best_set):
            best_set = selected[:]

        if len(best_set) >= 48:
            break

    # Local search: try to swap elements to allow adding more
    if best_set and len(best_set) < 48 and time.time() < deadline - 15:
        improved = best_set[:]
        remaining = [x for x in range(1, 513) if x not in improved]

        # Try adding elements
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if len(improved) >= 48:
                break
            if can_add(x, improved):
                improved.append(x)

        # Try swaps: remove one element, try to add two new ones
        if len(improved) < 48 and time.time() < deadline - 15:
            for idx in range(len(improved) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                if len(improved) >= 48:
                    break
                removed = improved.pop(idx)
                added_any = False
                for x in remaining:
                    if time.time() > deadline - 10:
                        break
                    if len(improved) >= 48:
                        break
                    if x == removed:
                        continue
                    if can_add(x, improved):
                        improved.append(x)
                        added_any = True
                        remaining.remove(x)
                        break
                if not added_any:
                    improved.insert(idx, removed)
                else:
                    remaining.append(removed)

        if len(improved) > len(best_set):
            best_set = improved

    # Final attempt to fill remaining slots
    if best_set and len(best_set) < 48 and time.time() < deadline - 5:
        final = best_set[:]
        remaining = [x for x in range(1, 513) if x not in final]
        for x in remaining:
            if time.time() > deadline - 3:
                break
            if len(final) >= 48:
                break
            if can_add(x, final):
                final.append(x)
        if len(final) > len(best_set):
            best_set = final

    return best_set
