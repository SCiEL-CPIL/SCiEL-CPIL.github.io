import time
import random
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    base = [1, 2, 5, 11]
    best_size = len(base)
    best_solution = base[:]

    def valid_and_sums(lst):
        n = len(lst)
        s = set()
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                b = lst[j]
                v = a + b
                if v in s:
                    return False, None
                s.add(v)
        return True, s

    def greedy_extend(lst, order, max_iter=None):
        # extend lst by adding elements in `order` (a permutation of candidates)
        # whenever they keep the set valid
        cur = lst[:]
        sums = set()
        n = len(cur)
        for i in range(n):
            a = cur[i]
            for j in range(i, n):
                sums.add(a + cur[j])
        used = set(cur)
        it = 0
        for x in order:
            if max_iter is not None and it >= max_iter:
                break
            it += 1
            if x in used:
                continue
            # Check if x can be added: for all y in cur, x+y not in sums, and 2*x not in sums
            new_ok = True
            for y in cur:
                if x + y in sums:
                    new_ok = False
                    break
            if new_ok and 2 * x in sums:
                new_ok = False
            if new_ok:
                for y in cur:
                    sums.add(x + y)
                sums.add(2 * x)
                cur.append(x)
                used.add(x)
        cur.sort()
        return cur

    def kick_and_grow(lst, kick_size=None):
        # remove a few random non-base elements, then greedily re-extend
        n = len(lst)
        non_base = [x for x in lst if x not in (1, 2, 5, 11)]
        if not non_base:
            return lst
        if kick_size is None:
            kick_size = min(3, len(non_base))
        else:
            kick_size = min(kick_size, len(non_base))
        to_remove = random.sample(non_base, kick_size)
        removed = set(to_remove)
        kept = [x for x in lst if x not in removed]
        # candidates: everything not kept
        cands = [x for x in range(1, 285) if x not in set(kept)]
        random.shuffle(cands)
        grown = greedy_extend(kept, cands)
        return grown

    def simulated_annealing_local_search(lst, iterations=5000):
        # Local search with simulated annealing to escape local optima
        cur = lst[:]
        cur_size = len(cur)
        temperature = 10.0
        cooling = 0.995
        best_local = cur[:]
        best_local_size = cur_size

        for _ in range(iterations):
            if time.time() > deadline:
                break
            if not non_base_elements(cur):
                break
            # Random kick: remove 1-3 elements and re-extend
            kick_sz = random.choice([1, 1, 2, 2, 3])
            candidate = kick_and_grow(cur, kick_size=kick_sz)
            cand_size = len(candidate)

            if cand_size >= cur_size:
                # Accept if same or better
                cur = candidate
                cur_size = cand_size
                if cur_size > best_local_size:
                    best_local_size = cur_size
                    best_local = cur[:]
            else:
                # Accept worse with probability based on temperature
                delta = cur_size - cand_size
                prob = temperature / (delta + 1)
                if random.random() < prob:
                    cur = candidate
                    cur_size = cand_size

            temperature *= cooling
            if temperature < 0.01:
                break

        return best_local

    def non_base_elements(lst):
        return [x for x in lst if x not in (1, 2, 5, 11)]

    # Phase 1: randomized greedy constructions from scratch
    base_set = set(base)
    for trial in range(600):
        if time.time() > deadline:
            break
        cands = [x for x in range(1, 285) if x not in base_set]
        random.shuffle(cands)
        built = greedy_extend(base[:], cands)
        if len(built) > best_size:
            best_size = len(built)
            best_solution = built[:]

    # Phase 2: simulated annealing local search
    cur = best_solution[:]
    cur_size = len(cur)
    no_improve = 0

    while time.time() < deadline - 5.0:
        # Try simulated annealing from current best
        improved = simulated_annealing_local_search(cur, iterations=3000)
        if len(improved) > cur_size:
            cur = improved
            cur_size = len(cur)
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_solution = cur[:]
        else:
            no_improve += 1
            if no_improve > 50:
                # Occasionally try a fresh random start
                cands = [x for x in range(1, 285) if x not in base_set]
                random.shuffle(cands)
                cur = greedy_extend(base[:], cands)
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
