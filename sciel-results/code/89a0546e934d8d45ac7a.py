import time
import random
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    base = [1, 2, 5, 11]
    base_set = set(base)
    best_size = len(base)
    best_solution = base[:]

    def compute_sums(lst):
        """Compute the set of all pairwise sums a+b where a<=b from lst."""
        s = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                s.add(a + lst[j])
        return s

    def can_add(lst, sums, x):
        """Check if x can be added to lst with given sums."""
        for y in lst:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_to(lst, sums, x):
        """Add x to lst and update sums in-place. Assumes can_add returned True."""
        lst.append(x)
        for y in lst[:-1]:
            sums.add(x + y)
        sums.add(2 * x)

    def greedy_expand(lst, candidates, deadline=None):
        """Greedily expand lst by adding candidates in order if valid."""
        cur = lst[:]
        sums = compute_sums(cur)
        used = set(cur)
        for x in candidates:
            if deadline is not None and time.time() > deadline:
                break
            if x in used:
                continue
            if can_add(cur, sums, x):
                add_to(cur, sums, x)
                used.add(x)
        cur.sort()
        return cur

    def kick(lst, k=3):
        """Remove k random non-base elements, return the reduced list."""
        non_base = [x for x in lst if x not in base_set]
        if len(non_base) < k:
            return lst[:]
        to_remove = set(random.sample(non_base, k))
        return [x for x in lst if x not in to_remove]

    def swap_kick(lst, k=1):
        """Remove k random non-base elements, return the reduced list."""
        non_base = [x for x in lst if x not in base_set]
        if len(non_base) < k:
            return lst[:]
        to_remove = set(random.sample(non_base, k))
        return [x for x in lst if x not in to_remove]

    def polish(lst, deadline=None):
        """Try to add any remaining candidate in ascending order."""
        cur = lst[:]
        sums = compute_sums(cur)
        used = set(cur)
        for x in range(1, 285):
            if deadline is not None and time.time() > deadline:
                break
            if x in used:
                continue
            if can_add(cur, sums, x):
                add_to(cur, sums, x)
                used.add(x)
        cur.sort()
        return cur

    # Phase 1: randomized greedy constructions from scratch
    all_cands = [x for x in range(1, 285) if x not in base_set]
    for trial in range(800):
        if time.time() > deadline - 15.0:
            break
        cands = all_cands[:]
        random.shuffle(cands)
        built = greedy_expand(base[:], cands, deadline=deadline - 15.0)
        # Polish
        built = polish(built, deadline=deadline - 15.0)
        if len(built) > best_size:
            best_size = len(built)
            best_solution = built[:]

    # Phase 2: local search with kick-and-regrow
    cur = best_solution[:]
    cur_size = len(cur)
    no_improve = 0
    while time.time() < deadline - 5.0:
        # Try different kick strategies
        r = random.random()
        if r < 0.4:
            kick_size = random.choice([2, 3, 4])
            reduced = kick(cur, kick_size)
        elif r < 0.7:
            kick_size = random.choice([1, 2])
            reduced = swap_kick(cur, kick_size)
        else:
            kick_size = random.choice([3, 5, 7])
            reduced = kick(cur, kick_size)

        cands = [x for x in range(1, 285) if x not in set(reduced)]
        random.shuffle(cands)
        grown = greedy_expand(reduced, cands, deadline=deadline - 5.0)
        # Polish
        grown = polish(grown, deadline=deadline - 5.0)

        if len(grown) > cur_size:
            cur = grown
            cur_size = len(grown)
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_solution = cur[:]
        else:
            no_improve += 1
            if no_improve > 60:
                # Fresh random start
                cands = all_cands[:]
                random.shuffle(cands)
                cur = greedy_expand(base[:], cands, deadline=deadline - 5.0)
                cur = polish(cur, deadline=deadline - 5.0)
                cur_size = len(cur)
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
                no_improve = 0

    # Final polish on best solution
    best_solution = polish(best_solution, deadline=deadline)

    return best_solution


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
