"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to sum-free set S (given as a set)."""
        for z in S_set:
            if x + z in S_set:
                return False
            if x - z in S_set:
                return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset with add-back refinement. Returns list of chosen elements."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        # Add-back pass: try adding excluded elements in reverse order
        all_set = set(arr)
        excluded = [x for x in sorted(arr, key=key, reverse=True) if x not in chosen_set]
        for x in excluded:
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def local_search_add(arr, initial, time_budget):
        """Hill-climbing: try adding elements one by one with conflict-aware priority."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = arr_set - current_set
        best = current
        best_len = len(current)
        
        rng = random.Random(999)
        iterations = 0
        # Pre-compute conflict counts for not_in elements
        conflict_count = {}
        for x in not_in:
            cnt = 0
            for z in current_set:
                if (x + z) in current_set or (x - z) in current_set:
                    cnt += 1
            conflict_count[x] = cnt
        
        while time_budget() > 0.5 and iterations < 15000:
            iterations += 1
            if not not_in:
                break
            # Try to find a zero-conflict element first
            found = False
            for x in list(not_in):
                if conflict_count.get(x, 0) == 0:
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                        # Update conflict counts for remaining not_in
                        for y in not_in:
                            if y != x:
                                if (x + y) in current_set or (x - y) in current_set or (y + x) in current_set or (y - x) in current_set:
                                    conflict_count[y] = conflict_count.get(y, 0) + 1
                        found = True
                        break
            if not found:
                # Pick random element from not_in
                x = rng.choice(tuple(not_in))
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    not_in.discard(x)
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                    # Update conflict counts
                    for y in not_in:
                        if y != x:
                            if (x + y) in current_set or (x - y) in current_set or (y + x) in current_set or (y - x) in current_set:
                                conflict_count[y] = conflict_count.get(y, 0) + 1
        return best

    def local_search_swap(arr, initial, time_budget):
        """Hill-climbing: try removing one element and adding one or two others."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = arr_set - current_set
        best = current
        best_len = len(current)
        
        rng = random.Random(777)
        iterations = 0
        while time_budget() > 0.5 and iterations < 8000:
            iterations += 1
            if not not_in or not current:
                break
            x = rng.choice(tuple(not_in))
            # Try adding x directly
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.discard(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                continue
            # Try removing one element and adding x
            idx = rng.randrange(len(current))
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)
            if can_add(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.discard(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Try removing two elements and adding x
                if current:
                    idx2 = rng.randrange(len(current))
                    removed2 = current[idx2]
                    current.pop(idx2)
                    current_set.discard(removed2)
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        # Revert both
                        current.append(removed2)
                        current_set.add(removed2)
                # Revert first
                current.append(removed)
                current_set.add(removed)
        return best

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        
        # Multiple deterministic orderings
        orderings = [
            lambda x: (abs(x), x),
            lambda x: x,
            lambda x: -x,
            lambda x: (-abs(x), x),
            lambda x: (-abs(x), -x),
            lambda x: (x % 3, abs(x)),
            lambda x: (-x % 3, -abs(x)),
            lambda x: (abs(x) % 3, abs(x)),
            lambda x: (abs(x) % 5, abs(x)),
        ]
        for key in orderings:
            cand = make_greedy(arr, key)
            if len(cand) > len(best):
                best = cand
        
        # Random restarts while time remains
        rng = random.Random(12345)
        guard = 0
        while remaining() > 10.0 and guard < 500:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)
            if len(cand) > len(best):
                best = cand
        
        # Local search: add elements
        if remaining() > 5.0:
            best = local_search_add(arr, best, remaining)
        
        # Local search: swap elements
        if remaining() > 2.0:
            best = local_search_swap(arr, best, remaining)
        
        # Final add-back pass on the best result
        if remaining() > 0.5:
            all_set = set(arr)
            not_in_best = all_set - set(best)
            for x in sorted(not_in_best, key=abs, reverse=True):
                if can_add(x, set(best)):
                    best.append(x)
        
        # Final safety: ensure validity
        if not is_sum_free(best):
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
