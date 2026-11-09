import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    def compute_reachable(S):
        """Compute all subset sums <= 200 for set S. Returns None if 200 is reachable."""
        reachable = {0}
        for x in S:
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 in new_sums:
                return None
            reachable |= new_sums
        return reachable

    def is_valid(S):
        return compute_reachable(S) is not None

    # Phase 1: Greedy initialization (high to low, then low to high, take best)
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            new = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new.add(ns)
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best = []
    for order in [list(range(100, 0, -1)), list(range(1, 101))]:
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    best_set = set(best)
    best_size = len(best)

    # Phase 2: Local search - try to add excluded elements
    def local_search_add(S):
        """Try to add any excluded element that keeps validity."""
        changed = True
        while changed:
            changed = False
            for x in range(1, 101):
                if x in S:
                    continue
                trial = S | {x}
                if is_valid(trial):
                    S = trial
                    changed = True
        return S

    # Phase 3: Try 1-for-2 swaps (remove one, add two)
    def try_swaps(S, time_limit):
        """Try removing one element and adding two excluded ones."""
        excluded = [x for x in range(1, 101) if x not in S]
        if not excluded:
            return S

        # Precompute reachable for S
        reach_S = compute_reachable(S)
        if reach_S is None:
            return S

        for rem in list(S):
            if time.time() > time_limit:
                break
            # Compute reachable without rem
            reach_no_rem = compute_reachable(S - {rem})
            if reach_no_rem is None:
                continue

            # Try to add as many excluded elements as possible
            trial = S - {rem}
            trial_reach = reach_no_rem
            added = 0
            for add in excluded:
                if time.time() > time_limit:
                    break
                if add in trial:
                    continue
                new_sums = set()
                for s in trial_reach:
                    ns = s + add
                    if ns <= 200:
                        new_sums.add(ns)
                if 200 not in new_sums:
                    trial = trial | {add}
                    trial_reach |= new_sums
                    added += 1

            if added >= 2:
                return trial

        return S

    # Phase 4: Randomized local search with swaps
    def randomized_search(S, iterations, time_limit):
        rng = random.Random(42)
        current = set(S)
        current_size = len(current)

        for i in range(iterations):
            if time.time() > time_limit:
                break

            # Randomly pick a removal candidate
            rem_candidates = list(current)
            if not rem_candidates:
                break
            rem = rng.choice(rem_candidates)

            # Try to add 2-3 random excluded elements
            excluded = [x for x in range(1, 101) if x not in current]
            if not excluded:
                break

            for num_add in [2, 3]:
                if len(excluded) < num_add:
                    continue
                adds = rng.sample(excluded, num_add)
                trial = current - {rem} | set(adds)
                if len(trial) > current_size:
                    if is_valid(trial):
                        current = trial
                        current_size = len(current)
                        break

        return current

    # Apply local search
    best_set = local_search_add(best_set)
    best_size = len(best_set)

    # Try swaps if time allows
    if time.time() < deadline - 30.0:
        swapped = try_swaps(best_set, deadline - 15.0)
        if len(swapped) > best_size:
            best_set = swapped
            best_size = len(best_set)
            best_set = local_search_add(best_set)
            best_size = len(best_set)

    # Randomized search if time allows
    if time.time() < deadline - 20.0:
        rand_result = randomized_search(best_set, 5000, deadline - 10.0)
        if len(rand_result) > best_size:
            best_set = rand_result
            best_size = len(best_set)
            best_set = local_search_add(best_set)
            best_size = len(best_set)

    # Final verification
    if not best_set:
        best_set = set(range(100, 50, -1))
        best_size = len(best_set)

    if not is_valid(best_set):
        # Fallback: guaranteed valid set
        best_set = set(range(67, 101))
        best_size = len(best_set)

    return list(best_set)
