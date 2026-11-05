"""Small inclusion-maximal Sidon sets in [1,256] – randomized construction with local search."""
import random
import time

N = 256
DEADLINE = 58.0  # safety margin under 60s


def _sums_of(S):
    """Return set of all a+b for a<=b in S."""
    s = set()
    for i, a in enumerate(S):
        s.add(2 * a)
        for b in S[i + 1:]:
            s.add(a + b)
    return s


def _is_valid_sidon(S):
    """Check all unordered pair sums including diagonal are distinct."""
    sums = {}
    for i, a in enumerate(S):
        if 2 * a in sums:
            return False
        sums[2 * a] = (a, a)
        for b in S[i + 1:]:
            s = a + b
            if s in sums:
                return False
            sums[s] = (a, b)
    return True


def _is_maximal(S, N=N):
    """Check that adding any missing element from [1,N] destroys the Sidon property."""
    sums = _sums_of(S)
    Sset = set(S)
    for x in range(1, N + 1):
        if x in Sset:
            continue
        # Check if adding x would create a duplicate sum
        new_sums = [2 * x] + [x + a for a in S]
        for ns in new_sums:
            if ns in sums:
                return True  # adding x breaks it, so we're good
        # If none of the new sums conflict, x could be added → not maximal
        # But wait: we need to check if ALL new sums are already in sums
        # Actually, if ANY new sum is NOT in sums, x can be added
        # If ALL new sums ARE in sums, x cannot be added
        all_conflict = all(ns in sums for ns in new_sums)
        if not all_conflict:
            return False  # x can be added, so S is not maximal
    return True


def _greedy_random(order):
    """Greedy: add elements in given order if they don't create sum collisions."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected


def _local_search(S, N=N, time_limit=58.0):
    """Try to reduce cardinality by removing elements and re-greeding."""
    start = time.time()
    best = list(S)
    best_len = len(best)
    
    while time.time() < time_limit:
        improved = False
        # Try removing each element and see if the remaining set can be
        # re-saturated to a maximal Sidon set with fewer total elements
        for i in range(len(best)):
            if time.time() > time_limit:
                break
            candidate = best[:i] + best[i+1:]
            if not candidate:
                continue
            # Greedily add back missing elements to make it maximal
            sums = _sums_of(candidate)
            cset = set(candidate)
            # Try to extend greedily
            extended = list(candidate)
            esums = set(sums)
            for x in range(1, N + 1):
                if x in cset:
                    continue
                added = [x + a for a in extended] + [2 * x]
                if len(set(added)) == len(added) and not esums.intersection(added):
                    extended.append(x)
                    esums.update(added)
            # Check if extended is valid and maximal
            if len(extended) < len(best) and _is_valid_sidon(extended) and _is_maximal(extended, N):
                best = extended
                best_len = len(best)
                improved = True
                break
        if not improved:
            break
    return best


def construct():
    start = time.time()
    best = None
    best_len = float('inf')
    
    # Multiple randomized greedy restarts
    for trial in range(2000):
        if time.time() > DEADLINE:
            break
        # Various random orderings
        r = random.random()
        if r < 0.3:
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif r < 0.5:
            # Descending
            order = list(range(N, 0, -1))
            random.shuffle(order)
        elif r < 0.7:
            # Sorted by random weight
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: random.random())
        elif r < 0.85:
            # Start with a random subset, then fill
            order = list(range(1, N + 1))
            random.shuffle(order)
        else:
            # Prime-weighted
            order = list(range(1, N + 1))
            order.sort(key=lambda x: random.random() * x + random.random() * (N + 1 - x))
        
        S = _greedy_random(order)
        if not S:
            continue
        
        # Ensure maximality by greedily adding
        sums = _sums_of(S)
        Sset = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > DEADLINE:
                    break
                if x in Sset:
                    continue
                added = [x + a for a in S] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.append(x)
                    sums.update(added)
                    Sset.add(x)
                    changed = True
            if time.time() > DEADLINE:
                break
        
        if not _is_valid_sidon(S):
            continue
        if not _is_maximal(S, N):
            # Force maximality
            sums = _sums_of(S)
            Sset = set(S)
            for x in range(1, N + 1):
                if time.time() > DEADLINE:
                    break
                if x in Sset:
                    continue
                added = [x + a for a in S] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    S.append(x)
                    sums.update(added)
                    Sset.add(x)
        
        if _is_valid_sidon(S) and _is_maximal(S, N):
            if len(S) < best_len:
                best_len = len(S)
                best = list(S)
                # Try local search to improve
                if time.time() < DEADLINE - 5:
                    improved = _local_search(best, N, DEADLINE)
                    if _is_valid_sidon(improved) and _is_maximal(improved, N):
                        if len(improved) < best_len:
                            best_len = len(improved)
                            best = improved
    
    if best is None:
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best = selected
    
    return best
