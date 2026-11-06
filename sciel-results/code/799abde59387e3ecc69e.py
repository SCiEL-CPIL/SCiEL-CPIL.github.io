import time
import random

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s limit

    base = (1, 2, 5, 11)
    base_set = set(base)

    # Precompute all pairwise sums for the base
    def compute_sums(lst):
        sums = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                sums.add(a + lst[j])
        return sums

    def try_add(lst, sums, x):
        """Try to add x to lst. Return (new_lst, new_sums) or None."""
        n = len(lst)
        # Check x + y for all y in lst, and 2*x
        for y in lst:
            s = x + y
            if s in sums:
                return None
        s2 = 2 * x
        if s2 in sums:
            return None
        new_sums = sums | {x + y for y in lst} | {s2}
        return lst + [x], new_sums

    def greedy_expand(lst, sums, candidates, order):
        """Greedy expansion: try candidates in given order, add if valid."""
        for x in order:
            if x in base_set or x in lst:
                continue
            result = try_add(lst, sums, x)
            if result is not None:
                lst, sums = result
        return lst, sums

    def kick(lst, sums, kick_size, candidates):
        """Remove kick_size random elements (not base), return new lst/sums."""
        removable = [x for x in lst if x not in base_set]
        if len(removable) < kick_size:
            return lst, sums
        to_remove = set(random.sample(removable, kick_size))
        new_lst = [x for x in lst if x not in to_remove]
        new_sums = compute_sums(new_lst)
        return new_lst, new_sums

    def full_search(lst, sums, candidates, time_budget):
        """Try to expand lst greedily, with kicks if stuck."""
        lst = list(lst)
        sums = set(sums)
        best = (len(lst), lst[:])
        stuck_count = 0

        while time.time() < time_budget:
            # Try greedy expansion
            before = len(lst)
            lst, sums = greedy_expand(lst, sums, candidates, list(range(12, 285)))
            if len(lst) > best[0]:
                best = (len(lst), lst[:])
            if len(lst) == before:
                # Stuck - try a kick
                kick_size = random.choice([1, 2, 3])
                lst, sums = kick(lst, sums, kick_size, candidates)
                if len(lst) < 4:
                    # Too aggressive, reset to base
                    lst = list(base)
                    sums = compute_sums(lst)
                    break
            else:
                stuck_count = 0
            stuck_count += 1
            if stuck_count > 50:
                break
        return best

    # Initial greedy from base
    best_size = 0
    best_sol = list(base)

    # Multiple randomized restarts
    random.seed(42)
    while time.time() < deadline:
        lst = list(base)
        sums = compute_sums(lst)

        # Randomize the order of candidates
        candidates = list(range(12, 285))
        random.shuffle(candidates)

        # Greedy expand
        lst, sums = greedy_expand(lst, sums, candidates, candidates)
        if len(lst) > best_size:
            best_size = len(lst)
            best_sol = lst[:]

        # If not at 20, try kicks
        if len(lst) < 20 and time.time() < deadline - 5:
            # Try a few kicks
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                kick_size = random.choice([1, 2, 3])
                lst2, sums2 = kick(lst, sums, kick_size, candidates)
                if len(lst2) < 4:
                    continue
                # Re-expand
                lst2, sums2 = greedy_expand(lst2, sums2, candidates, candidates)
                if len(lst2) > best_size:
                    best_size = len(lst2)
                    best_sol = lst2[:]

        # Also try a more aggressive search with smaller kicks
        if len(lst) < 18 and time.time() < deadline - 10:
            for _ in range(50):
                if time.time() > deadline - 3:
                    break
                kick_size = random.choice([1, 2])
                lst2, sums2 = kick(lst, sums, kick_size, candidates)
                if len(lst2) < 4:
                    continue
                # Greedy with random order
                cand2 = list(range(12, 285))
                random.shuffle(cand2)
                lst2, sums2 = greedy_expand(lst2, sums2, cand2, cand2)
                if len(lst2) > best_size:
                    best_size = len(lst2)
                    best_sol = lst2[:]

    return best_sol
