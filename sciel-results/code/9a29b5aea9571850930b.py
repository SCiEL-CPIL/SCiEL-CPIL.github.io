import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    base = [1, 2, 5, 11]
    base_set = frozenset(base)
    best_size = len(base)
    best_solution = base[:]

    def compute_sums(lst):
        s = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                s.add(a + lst[j])
        return s

    def greedy_extend(lst, order):
        cur = lst[:]
        sums = compute_sums(cur)
        used = set(cur)
        for x in order:
            if x in used:
                continue
            ok = True
            for y in cur:
                if x + y in sums:
                    ok = False
                    break
            if ok and 2 * x in sums:
                ok = False
            if ok:
                for y in cur:
                    sums.add(x + y)
                sums.add(2 * x)
                cur.append(x)
                used.add(x)
        cur.sort()
        return cur

    def kick_and_grow(lst, kick_size=None):
        non_base = [x for x in lst if x not in base_set]
        if not non_base:
            return lst
        if kick_size is None:
            kick_size = min(3, len(non_base))
        else:
            kick_size = min(kick_size, len(non_base))
        to_remove = random.sample(non_base, kick_size)
        removed = set(to_remove)
        kept = [x for x in lst if x not in removed]
        kept_set = set(kept)
        cands = [x for x in range(1, 285) if x not in kept_set]
        random.shuffle(cands)
        return greedy_extend(kept, cands)

    def get_forbidden(trial, trial_sums):
        """Precompute the set of candidates that cannot be added to trial."""
        forbidden = set(trial)
        # If z is added, z+y must not be in trial_sums for any y in trial
        # So z is forbidden if z+y in trial_sums for any y in trial
        # i.e., z in { s - y | s in trial_sums, y in trial }
        for s in trial_sums:
            for y in trial:
                z = s - y
                if 1 <= z <= 284:
                    forbidden.add(z)
        # Also 2z must not be in trial_sums
        # So z is forbidden if 2z in trial_sums
        # i.e., z in { s/2 | s in trial_sums, s % 2 == 0 }
        for s in trial_sums:
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= 284:
                    forbidden.add(z)
        return forbidden

    def fast_swap_polish(lst, max_rounds=50):
        """Optimized swap local search using precomputed forbidden sets."""
        cur = lst[:]
        cur_set = set(cur)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 2.0:
            improved = False
            rounds += 1
            non_base = [x for x in cur if x not in base_set]
            # Shuffle non_base to randomize which element we try to remove first
            random.shuffle(non_base)
            for rem in non_base:
                if time.time() > deadline - 2.0:
                    break
                trial = [x for x in cur if x != rem]
                trial_set = set(trial)
                trial_sums = compute_sums(trial)
                forbidden = get_forbidden(trial, trial_sums)
                
                # Find the first valid candidate to add
                # We want to maximize size, so any valid addition is an improvement (size +1)
                # But we should prefer candidates that allow further growth? 
                # For now, just find any valid one.
                # To be deterministic-ish or random, we can shuffle candidates or just pick the smallest?
                # Picking the smallest might lead to local optima. 
                # Let's shuffle the valid candidates.
                valid_cands = [x for x in range(1, 285) if x not in forbidden]
                if not valid_cands:
                    continue
                
                random.shuffle(valid_cands)
                add = valid_cands[0]
                
                # Accept the swap
                cur = trial + [add]
                cur.sort()
                cur_set = set(cur)
                improved = True
                break
        return cur

    def simulated_annealing_local_search(lst, iterations=3000):
        cur = lst[:]
        cur_size = len(cur)
        temperature = 10.0
        cooling = 0.995
        best_local = cur[:]
        best_local_size = cur_size

        non_base = [x for x in cur if x not in base_set]
        if not non_base:
            return best_local

        for _ in range(iterations):
            if time.time() > deadline:
                break
            kick_sz = random.choice([1, 1, 2, 2, 3])
            candidate = kick_and_grow(cur, kick_size=kick_sz)
            cand_size = len(candidate)

            if cand_size >= cur_size:
                cur = candidate
                cur_size = cand_size
                if cur_size > best_local_size:
                    best_local_size = cur_size
                    best_local = cur[:]
            else:
                delta = cur_size - cand_size
                prob = temperature / (delta + 1)
                if random.random() < prob:
                    cur = candidate
                    cur_size = cand_size

            temperature *= cooling
            if temperature < 0.01:
                break

        return best_local

    # Phase 1: randomized greedy constructions from scratch
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
            if no_improve > 30:
                # Occasionally try a fresh random start
                cands = [x for x in range(1, 285) if x not in base_set]
                random.shuffle(cands)
                cur = greedy_extend(base[:], cands)
                cur_size = len(cur)
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
                no_improve = 0

    # Phase 3: final polish pass with optimized swap-based local search
    if time.time() < deadline - 3.0:
        polished = fast_swap_polish(best_solution, max_rounds=100)
        if len(polished) > best_size:
            best_size = len(polished)
            best_solution = polished[:]

    # Additional polish rounds if time permits
    while time.time() < deadline - 1.0:
        prev = best_solution[:]
        polished = fast_swap_polish(prev, max_rounds=20)
        if len(polished) > len(prev):
            best_solution = polished
            best_size = len(polished)
        else:
            break

    return best_solution


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
    n = len(res)
    s = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            v = res[i] + res[j]
            if v in s:
                valid = False
                break
            s.add(v)
        if not valid:
            break
    print("valid:", valid)
