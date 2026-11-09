import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    TARGET = 200

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

    # Phase 1: Greedy initialization with multiple orderings
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

    # Try multiple orderings: high-to-low, low-to-high, and density-minimizing heuristics
    best = []
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        # Sort by "impact" - numbers that create fewer new sums first
        sorted(range(1, 101), key=lambda x: x % 10),
        # Alternate large/small
        [x for x in range(100, 0, -2)] + [x for x in range(99, 0, -2)],
    ]

    for order in orders:
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    best_set = set(best)
    best_size = len(best)

    # Phase 2: Local search - try to add excluded elements
    def local_search_add(S, time_limit=None):
        """Try to add any excluded element that keeps validity."""
        if time_limit is None:
            time_limit = deadline - 5.0
        changed = True
        while changed:
            changed = False
            if time.time() > time_limit:
                break
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

        reach_S = compute_reachable(S)
        if reach_S is None:
            return S

        best_trial = S
        best_added = 0

        for rem in list(S):
            if time.time() > time_limit:
                break
            reach_no_rem = compute_reachable(S - {rem})
            if reach_no_rem is None:
                continue

            trial = S - {rem}
            trial_reach = set(reach_no_rem)
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

            if added > best_added:
                best_added = added
                best_trial = trial

        if best_added >= 2:
            return best_trial
        return S

    # Phase 4: Randomized local search with swaps
    def randomized_search(S, iterations, time_limit):
        rng = random.Random(42)
        current = set(S)
        current_size = len(current)

        for i in range(iterations):
            if time.time() > time_limit:
                break

            rem_candidates = list(current)
            if not rem_candidates:
                break
            rem = rng.choice(rem_candidates)

            excluded = [x for x in range(1, 101) if x not in current]
            if not excluded:
                break

            improved = False
            for num_add in [2, 3]:
                if len(excluded) < num_add:
                    continue
                adds = rng.sample(excluded, num_add)
                trial = current - {rem} | set(adds)
                if len(trial) > current_size:
                    if is_valid(trial):
                        current = trial
                        current_size = len(current)
                        improved = True
                        break

            # Also try single additions
            if not improved:
                for x in rng.sample(excluded, min(5, len(excluded))):
                    trial = current | {x}
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
        best_set = set(range(67, 101))
        best_size = len(best_set)

    return list(best_set)
