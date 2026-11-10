import time
import random

def construct():
    deadline = time.time() + 80
    n = 256

    # Precompute conflict bitmasks: conflicts[i] is a bitmask of j where (i*j)%(i+j)==0
    conflicts = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)

    def check_conflict(x, mask):
        """Check if x conflicts with any element in mask (bitmask)."""
        return (conflicts[x] & mask) != 0

    odds = list(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))

    best_set = set(odds)
    best_size = len(best_set)
    best_mask = 0
    for x in odds:
        best_mask |= (1 << x)

    def greedy_add(start_set, order, time_limit=None):
        """Greedily add elements from order to start_set."""
        s = set(start_set)
        mask = 0
        for x in s:
            mask |= (1 << x)
        for e in order:
            if e not in s and not check_conflict(e, mask):
                s.add(e)
                mask |= (1 << e)
            if time_limit and time.time() > time_limit:
                break
        return s, mask, len(s)

    # Phase 1: Randomized greedy add evens to odds
    iter_count = 0
    while time.time() < deadline - 20:
        order = evens[:]
        random.shuffle(order)
        s, mask, size = greedy_add(odds, order)
        if size > best_size:
            best_size = size
            best_set = s
            best_mask = mask
        iter_count += 1

        # Periodically do local search on best
        if iter_count % 200 == 0 and time.time() < deadline - 15:
            # Local search: try adding any element
            improved = True
            while improved:
                improved = False
                for x in range(1, n + 1):
                    if not (best_mask & (1 << x)):
                        if not check_conflict(x, best_mask):
                            best_mask |= (1 << x)
                            best_set.add(x)
                            best_size += 1
                            improved = True
                            break
            # Try 1-for-1 swaps
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                for x in range(1, n + 1):
                    if best_mask & (1 << x):
                        continue
                    # Try removing one element y to add x
                    for y in best_set:
                        new_mask = best_mask & ~(1 << y)
                        if not check_conflict(x, new_mask):
                            best_mask = new_mask | (1 << x)
                            best_set.discard(y)
                            best_set.add(x)
                            improved = True
                            break
                    if improved:
                        break
            # Try 1-for-2 swaps (remove two, add one)
            if time.time() < deadline - 15:
                improved = True
                while improved and time.time() < deadline - 15:
                    improved = False
                    for x in range(1, n + 1):
                        if best_mask & (1 << x):
                            continue
                        # Try removing two elements
                        elems = list(best_set)
                        for i in range(len(elems)):
                            for j in range(i + 1, len(elems)):
                                y, z = elems[i], elems[j]
                                new_mask = best_mask & ~(1 << y) & ~(1 << z)
                                if not check_conflict(x, new_mask):
                                    best_mask = new_mask | (1 << x)
                                    best_set.discard(y)
                                    best_set.discard(z)
                                    best_set.add(x)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

    # Phase 2: Try different cores
    while time.time() < deadline - 5:
        # Try starting from odds only but with different random subsets removed
        if random.random() < 0.3:
            # Remove a random subset of odds, then greedily add
            core = odds[:]
            k = random.randint(0, 10)
            for _ in range(k):
                idx = random.randint(0, len(core) - 1)
                core.pop(idx)
        else:
            core = odds[:]
        order = evens[:]
        random.shuffle(order)
        s, mask, size = greedy_add(core, order)
        if size > best_size:
            best_size = size
            best_set = s
            best_mask = mask

        # Also try: start from a random subset of all numbers
        if random.random() < 0.2:
            all_nums = list(range(1, n + 1))
            random.shuffle(all_nums)
            core = all_nums[:random.randint(50, 100)]
            # Filter to only non-conflicting pairs (greedy build)
            filtered = []
            fm = 0
            for x in core:
                if not check_conflict(x, fm):
                    filtered.append(x)
                    fm |= (1 << x)
            order = [x for x in all_nums if x not in set(filtered)]
            random.shuffle(order)
            s, mask, size = greedy_add(filtered, order)
            if size > best_size:
                best_size = size
                best_set = s
                best_mask = mask

    # Final local search
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in range(1, n + 1):
            if not (best_mask & (1 << x)):
                if not check_conflict(x, best_mask):
                    best_mask |= (1 << x)
                    best_set.add(x)
                    best_size += 1
                    improved = True
                    break

    return list(best_set)
