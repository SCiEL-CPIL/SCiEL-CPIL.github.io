"""Rewritten program: maximize cardinality of a subset of [1,100] with no subset summing to 200."""

def construct():
    target = 200
    # Start with the greedy "large-first" seed: take elements from 100 down to 1,
    # including x only if no subset of already-chosen elements sums to target - x.
    # This yields {66, 67, ..., 100} (size 35) and is a strong baseline.
    seed = set()
    reachable = {0}
    for x in range(100, 0, -1):
        if (target - x) not in reachable:
            seed.add(x)
            # update reachable sums (capped at target)
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums

    best_set = frozenset(seed)
    best_score = len(best_set)

    # Try to improve via local search: swap one excluded element for one included element,
    # or drop one included element and add two excluded elements (2-for-1 upgrade).
    # We validate each candidate with a full subset-sum DP (fast: O(n * target)).
    import time
    start = time.time()
    deadline = start + 50.0  # safety margin well under 60s

    all_elements = list(range(1, 101))
    excluded = [x for x in all_elements if x not in seed]

    def is_valid(s):
        # returns True if no subset of s sums to target
        # bitset DP
        bits = 1  # bit 0 set
        for x in s:
            bits |= (bits << x)
            if bits >> target & 1:
                return False
        return True

    # 1-for-1 swaps
    for ex in excluded:
        if time.time() > deadline:
            break
        for inc in list(seed):
            if inc == ex:
                continue
            candidate = (seed - {inc}) | {ex}
            if len(candidate) != best_score:
                continue
            if is_valid(candidate):
                # same size, but maybe enables further swaps later; keep best lex? 
                # Actually we want to maximize cardinality, so same-size swaps don't help cardinality.
                # Skip unless we're trying to find a different same-size set that allows future upgrades.
                pass

    # 2-for-1 upgrades: remove one included, add two excluded
    for ex1 in excluded:
        if time.time() > deadline:
            break
        for ex2 in excluded:
            if ex2 <= ex1:
                continue
            if time.time() > deadline:
                break
            for inc in list(seed):
                if inc == ex1 or inc == ex2:
                    continue
                candidate = (seed - {inc}) | {ex1, ex2}
                if len(candidate) <= best_score:
                    continue
                if is_valid(candidate):
                    best_set = frozenset(candidate)
                    best_score = len(best_set)
                    # update seed and excluded for further search
                    seed = set(best_set)
                    excluded = [x for x in all_elements if x not in seed]
                    break
            if time.time() > deadline:
                break

    # Also try 3-for-2: remove two included, add three excluded
    # Only if we still have time
    if time.time() < deadline - 10:
        for inc1 in list(seed):
            if time.time() > deadline - 5:
                break
            for inc2 in list(seed):
                if inc2 <= inc1:
                    continue
                if time.time() > deadline - 5:
                    break
                for ex1 in excluded:
                    if time.time() > deadline - 3:
                        break
                    for ex2 in excluded:
                        if ex2 <= ex1:
                            continue
                        if ex2 == inc1 or ex2 == inc2:
                            continue
                        if time.time() > deadline - 3:
                            break
                        for ex3 in excluded:
                            if ex3 <= ex2:
                                continue
                            if ex3 == inc1 or ex3 == inc2:
                                continue
                            if time.time() > deadline - 2:
                                break
                            candidate = (seed - {inc1, inc2}) | {ex1, ex2, ex3}
                            if len(candidate) <= best_score:
                                continue
                            if is_valid(candidate):
                                best_set = frozenset(candidate)
                                best_score = len(best_set)
                                seed = set(best_set)
                                excluded = [x for x in all_elements if x not in seed]

    return list(best_set)
