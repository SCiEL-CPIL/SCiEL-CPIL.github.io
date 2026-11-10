import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin vs 90s kill

    n = 12
    max_val_limit = 1048576

    def _valid(arr):
        sums = {0}
        for x in arr:
            sums |= {s + x for s in sums}
        return len(sums) == (1 << n)

    # ---- 1. Greedy "complete sequence" construction (smallest feasible value) ----
    def _greedy_construct(max_val_limit):
        arr = []
        sums = {0}
        for _ in range(n):
            chosen = None
            for val in range(1, max_val_limit + 1):
                if val in sums:
                    continue
                new_sums = {s + val for s in sums}
                if new_sums & sums:
                    continue  # collision -> not a valid extension
                arr.append(val)
                sums |= new_sums
                chosen = val
                break
            if chosen is None:
                return None
        return arr

    # ---- 2. Fast local search refinement ----
    def _refine(arr, budget):
        arr = arr[:]
        if not _valid(arr):
            return None
        cur_max = max(arr)
        best = arr[:]
        best_max = cur_max
        T = 30.0
        T_min = 0.5
        alpha = 0.995

        while time.time() < budget and T > T_min:
            neighbor = arr[:]
            r = random.random()
            if r < 0.5:
                idx = random.randrange(n)
                old = neighbor[idx]
                new = max(1, old - random.randint(1, max(1, old // 5)))
                if new != old:
                    neighbor[idx] = new
            elif r < 0.8:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                idx = random.randrange(n)
                old = neighbor[idx]
                new = old + random.randint(1, max(1, old // 5))
                if new <= max_val_limit:
                    neighbor[idx] = new

            sums = {0}
            for x in neighbor:
                sums |= {s + x for s in sums}
            is_valid = (len(sums) == (1 << n))
            new_max = max(neighbor)
            if is_valid:
                cost = new_max
            else:
                cost = new_max + ((1 << n) - len(sums)) * 1000

            delta = cost - cur_max
            if delta < 0:
                arr = neighbor
                cur_max = cost
                if is_valid and new_max < best_max:
                    best_max = new_max
                    best = neighbor[:]
            else:
                if T > T_min:
                    if random.random() < 2.71828 ** (-delta / T):
                        arr = neighbor
                        cur_max = cost
            T *= alpha
        return best

    # ---- Main flow ----
    # Establish a strong low-max starting point via greedy complete sequence.
    greedy = _greedy_construct(max_val_limit)
    if greedy is not None and _valid(greedy):
        best = greedy
        best_max = max(greedy)
    else:
        best = [1 << i for i in range(n)]
        best_max = best_max if 'best_max' in dir() else max(best)
        best_max = max(best)

    # Refine within the time budget.
    if time.time() < deadline:
        refined = _refine(best, deadline)
        if refined is not None and max(refined) < best_max:
            best = refined
            best_max = max(best)

    return best
