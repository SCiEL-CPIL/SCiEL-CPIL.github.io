"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    def elapsed():
        return time.time() - start_time

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to sum-free set S (given as a set)."""
        for z in S_set:
            if x + z in S_set:
                return False
            if x - z in S_set:
                return False
        return True

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset. Returns list of chosen elements."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def greedy_then_addback(arr, key, time_budget):
        """Greedy in given order, then iteratively try to add back excluded elements."""
        chosen = []
        chosen_set = set()
        excluded = []
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
            else:
                excluded.append(x)
        improved = True
        while improved and time_budget() > 0.3:
            improved = False
            new_excluded = []
            for x in excluded:
                if can_add(x, chosen_set):
                    chosen.append(x)
                    chosen_set.add(x)
                    improved = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        return chosen

    def local_search(arr, initial, time_budget):
        """Hill-climbing with add, swap, and swap-two (remove one, add two) moves."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = arr_set - current_set
        best = list(current)
        best_len = len(current)

        rng = random.Random(999)
        iterations = 0
        while time_budget() > 0.5 and iterations < 50000:
            iterations += 1
            if not not_in:
                break
            x = rng.choice(tuple(not_in))
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.discard(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Find conflicting elements
                conflicts = []
                for y in current:
                    if x + y in current_set or x - y in current_set:
                        conflicts.append(y)
                if conflicts:
                    # Move 1: remove one conflicting, add x (swap)
                    y = rng.choice(conflicts)
                    current.remove(y)
                    current_set.discard(y)
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        current.append(y)
                        current_set.add(y)
                    # Move 2: remove one conflicting, add x AND another (swap-two, net +1)
                    if not_in and len(conflicts) >= 1:
                        y2 = rng.choice(conflicts)
                        others = list(not_in)
                        if len(others) >= 2:
                            # pick two candidates from not_in
                            w, v = rng.sample(others, 2)
                            # try removing y2, adding x and w
                            current.remove(y2)
                            current_set.discard(y2)
                            ok1 = can_add(x, current_set)
                            if ok1:
                                current.append(x)
                                current_set.add(x)
                                ok2 = can_add(w, current_set)
                                if ok2:
                                    current.append(w)
                                    current_set.add(w)
                                    not_in.discard(x)
                                    not_in.discard(w)
                                    if len(current) > best_len:
                                        best = list(current)
                                        best_len = len(current)
                                else:
                                    current.remove(x)
                                    current_set.discard(x)
                            # try removing y2, adding x and v
                            if can_add(x, current_set):
                                current.append(x)
                                current_set.add(x)
                                if can_add(v, current_set):
                                    current.append(v)
                                    current_set.add(v)
                                    not_in.discard(x)
                                    not_in.discard(v)
                                    if len(current) > best_len:
                                        best = list(current)
                                        best_len = len(current)
                                else:
                                    current.remove(x)
                                    current_set.discard(x)
                            current.append(y2)
                            current_set.add(y2)
                elif current:
                    # No direct conflicts but x can't be added? Shouldn't happen, but fallback
                    y = current[rng.randrange(len(current))]
                    current.remove(y)
                    current_set.discard(y)
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        current.append(y)
                        current_set.add(y)
        return best

    def solve_instance(arr, time_budget):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        # Order 1: by absolute value ascending (small magnitudes first)
        cand = greedy_then_addback(arr, lambda x: (abs(x), x), time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 2: by value ascending
        cand = greedy_then_addback(arr, lambda x: x, time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 3: by value descending
        cand = greedy_then_addback(arr, lambda x: -x, time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 4: by absolute value descending (large magnitudes first)
        cand = greedy_then_addback(arr, lambda x: (-abs(x), x), time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 5: by absolute value descending, tie-break by value
        cand = greedy_then_addback(arr, lambda x: (-abs(x), -x), time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 6: by value modulo 3 (group by residue)
        cand = greedy_then_addback(arr, lambda x: (x % 3, x), time_budget)
        if len(cand) > len(best):
            best = cand
        # Order 7: by absolute value modulo 3
        cand = greedy_then_addback(arr, lambda x: (abs(x) % 3, x), time_budget)
        if len(cand) > len(best):
            best = cand
        # Random restarts while time remains
        rng = random.Random(12345)
        guard = 0
        while time_budget() > 3.0 and guard < 500:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = greedy_then_addback(order, lambda x: x, time_budget)
            if len(cand) > len(best):
                best = cand
        # Local search: try to improve by adding/swapping elements
        if time_budget() > 1.0:
            best = local_search(arr, best, time_budget)
        # Final safety: ensure validity
        if not is_sum_free(best):
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    # Dedicated time budgets per instance: 45s, 45s, 40s = 130s total
    budgets = [45.0, 45.0, 40.0]
    for i, (size, seed) in enumerate(((60, 801), (120, 802), (240, 803))):
        arr = instance(size, seed)
        # Use min of allocated budget and remaining time
        remaining = DEADLINE - elapsed()
        budget = min(budgets[i], remaining)
        # Create a time_budget function that checks against the instance deadline
        instance_start = elapsed()
        def time_budget():
            return budget - (elapsed() - instance_start)
        results.append(solve_instance(arr, time_budget))
    return results
# EVOLVE-BLOCK-END
