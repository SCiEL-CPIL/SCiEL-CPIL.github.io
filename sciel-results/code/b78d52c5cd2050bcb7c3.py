import time
import random
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    base = [1, 2, 5, 11]
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

    def try_add(lst, sums, x):
        """Check if x can be added to lst with given sums. Return (ok, new_sums) if ok."""
        # Check all sums x+y for y in lst
        for y in lst:
            if x + y in sums:
                return False, None
        # Check 2*x
        if 2 * x in sums:
            return False, None
        # Build new sums
        new_sums = sums | set()
        for y in lst:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return True, new_sums

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
            ok, new_sums = try_add(cur, sums, x)
            if ok:
                cur.append(x)
                sums = new_sums
                used.add(x)
        cur.sort()
        return cur

    def kick(lst, k=3):
        """Remove k random non-base elements, return the reduced list."""
        non_base = [x for x in lst if x not in (1, 2, 5, 11)]
        if len(non_base) < k:
            return lst[:]
        to_remove = set(random.sample(non_base, k))
        return [x for x in lst if x not in to_remove]

    # Phase 1: randomized greedy constructions from scratch
    base_set = set(base)
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
        cands = [x for x in range(1, 285) if x not in base_set]
        random.shuffle(cands)
        built = greedy_expand(base[:], cands, deadline=deadline - 10.0)
        if len(built) > best_size:
            best_size = len(built)
            best_solution = built[:]

    # Phase 2: local search with kick-and-regrow
    cur = best_solution[:]
    cur_size = len(cur)
    no_improve = 0
    while time.time() < deadline - 5.0:
        # Try different kick sizes
        kick_size = random.choice([2, 3, 4])
        reduced = kick(cur, kick_size)
        cands = [x for x in range(1, 285) if x not in set(reduced)]
        random.shuffle(cands)
        grown = greedy_expand(reduced, cands, deadline=deadline - 5.0)
        if len(grown) > cur_size:
            cur = grown
            cur_size = len(grown)
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_solution = cur[:]
        else:
            no_improve += 1
            if no_improve > 80:
                # Fresh random start
                cands = [x for x in range(1, 285) if x not in base_set]
                random.shuffle(cands)
                cur = greedy_expand(base[:], cands, deadline=deadline - 5.0)
                cur_size = len(cur)
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
                no_improve = 0

    return best_solution


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
    v, s = valid_and_sums(res)
    print("valid:", v)
