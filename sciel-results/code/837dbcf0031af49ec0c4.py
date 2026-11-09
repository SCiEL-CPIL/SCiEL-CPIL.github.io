"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1  # bits 0..200

    def reachable_bitset(elements):
        """Compute bitset of reachable subset sums up to TARGET."""
        r = 1  # bit 0 set (sum 0)
        for x in elements:
            r |= (r << x)
            r &= MASK
            if (r >> TARGET) & 1:
                return r
        return r

    def has_sum_200(elements):
        """Check if any subset of elements sums to exactly 200 using bitset."""
        r = 1
        for x in elements:
            r |= (r << x)
            r &= MASK
            if (r >> TARGET) & 1:
                return True
        return False

    def try_add(elements, x):
        """Try adding x to elements; return True if no subset sums to 200."""
        r = 1
        for v in elements:
            r |= (r << v)
            r &= MASK
        r |= (r << x)
        r &= MASK
        return not ((r >> TARGET) & 1)

    # Known-good base: [66, 100], size 35
    base = list(range(66, 101))
    best = base[:]
    best_size = len(best)

    # Multiple greedy passes with different orderings
    rng = random.Random(12345)
    
    # Greedy descending
    current = base[:]
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        if try_add(current, x):
            current.append(x)
    if len(current) > best_size:
        best = current
        best_size = len(best)

    # Greedy ascending
    current = base[:]
    for x in range(1, 66):
        if time.time() > deadline:
            break
        if try_add(current, x):
            current.append(x)
    if len(current) > best_size:
        best = current
        best_size = len(best)

    # Many random-order greedy passes
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 66))
        rng.shuffle(order)
        current = base[:]
        for x in order:
            if time.time() > deadline - 5:
                break
            if try_add(current, x):
                current.append(x)
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Local search: try to improve by removing one and adding one or two
    best_set = set(best)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        best_set = set(best)
        remaining_pool = [x for x in range(1, 66) if x not in best_set]
        
        # Try single swaps: remove one from best, add one from pool
        for remove in list(best_set):
            if time.time() > deadline - 15:
                break
            remaining = [x for x in best if x != remove]
            for add in remaining_pool:
                if time.time() > deadline - 15:
                    break
                test = remaining + [add]
                if not has_sum_200(test):
                    # Same size, but might enable further improvements
                    # Only accept if it leads to improvement later; for now skip same-size swaps
                    pass
        
        # Try remove one, add two (net +1)
        for remove in list(best_set):
            if time.time() > deadline - 15:
                break
            remaining = [x for x in best if x != remove]
            pool = [x for x in range(1, 66) if x not in set(remaining)]
            for i in range(len(pool)):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, len(pool)):
                    test = remaining + [pool[i], pool[j]]
                    if not has_sum_200(test):
                        best = test
                        best_set = set(best)
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Try to add elements one by one in random order to best
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        current = best[:]
        current_set = set(current)
        pool = [x for x in range(1, 66) if x not in current_set]
        rng.shuffle(pool)
        for x in pool:
            if time.time() > deadline - 10:
                break
            if try_add(current, x):
                current.append(x)
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Aggressive local search with bitset: try all single-element removals and re-additions
    if time.time() < deadline - 10:
        current = best[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for remove in list(current_set):
                if time.time() > deadline - 10:
                    break
                remaining = [x for x in current if x != remove]
                pool = [x for x in range(1, 66) if x not in set(remaining)]
                # Try adding each pool element
                for add in pool:
                    if time.time() > deadline - 10:
                        break
                    test = remaining + [add]
                    if not has_sum_200(test):
                        # Check if this leads to a larger set
                        # Try to fill further
                        test_set = set(test)
                        further_pool = [x for x in range(1, 66) if x not in test_set]
                        rng.shuffle(further_pool)
                        for x in further_pool:
                            if time.time() > deadline - 10:
                                break
                            if try_add(test, x):
                                test.append(x)
                        if len(test) > len(current):
                            current = test
                            current_set = set(current)
                            best = current[:]
                            best_size = len(best)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final: try to maximize by greedily adding in multiple random orders
    for trial in range(1000):
        if time.time() > deadline - 5:
            break
        current = best[:]
        current_set = set(current)
        pool = [x for x in range(1, 66) if x not in current_set]
        rng.shuffle(pool)
        for x in pool:
            if time.time() > deadline - 5:
                break
            if try_add(current, x):
                current.append(x)
        if len(current) > best_size:
            best = current
            best_size = len(best)

    # Final verification
    if has_sum_200(best):
        best = list(range(66, 101))

    return best
# EVOLVE-BLOCK-END
