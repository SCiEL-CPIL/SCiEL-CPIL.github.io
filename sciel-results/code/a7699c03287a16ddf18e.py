import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    TARGET = 200
    N = 100
    UNIVERSE = list(range(1, N + 1))

    def has_subset_sum_200(selected):
        """Check if 'selected' contains a subset summing to exactly 200 via bitset DP."""
        bits = 1  # sum 0 achievable
        for x in selected:
            if x <= TARGET:
                bits |= (bits << x)
            if (bits >> TARGET) & 1:
                return True
        return False

    def greedy(order):
        selected = []
        for x in order:
            selected.append(x)
            if has_subset_sum_200(selected):
                selected.pop()
        return selected

    def add_local_search(selected, order, max_rounds=20):
        """Add missing elements if it keeps validity. Returns (list, improved_flag)."""
        selected_set = set(selected)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for x in order:
                if x in selected_set:
                    continue
                selected_set.add(x)
                if has_subset_sum_200(list(selected_set)):
                    selected_set.remove(x)
                else:
                    improved = True
            if not improved:
                break
        return list(selected_set), improved

    def swap_local_search(selected, order, max_rounds=10):
        """Try swapping an in-set element with a missing one to increase size or enable more adds.
        Actually swaps don't increase size directly; use remove+add. Instead do: remove one, then re-add greedily."""
        selected_set = set(selected)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for x in order:
                if x not in selected_set:
                    continue
                # try removing x then re-adding others
                selected_set.remove(x)
                new_set, _ = add_local_search(list(selected_set), order, max_rounds=3)
                if len(new_set) > len(selected_set) + 0:  # strictly better after re-add
                    # Actually we removed x, so compare len(new_set) to len(selected_set)+1 (original had x)
                    if len(new_set) > len(selected_set):
                        selected_set = set(new_set)
                        improved = True
                        break
                else:
                    selected_set.add(x)
            if not improved:
                break
        return list(selected_set)

    best = []
    best_len = 0

    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
        list(range(1, N // 2 + 1)) + list(range(N // 2 + 1, N + 1)),
        list(range(N, N // 2, -1)) + list(range(1, N // 2 + 1)),
        list(range(1, N // 2 + 1)) + list(range(N, N // 2, -1)),
    ]

    rng = random.Random(42)
    for seed in range(40):
        if time.time() > deadline - 10:
            break
        order = list(UNIVERSE)
        rng = random.Random(seed)
        rng.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline - 5:
            break
        cand = greedy(order)
        cand, _ = add_local_search(cand, order)
        cand = swap_local_search(cand, order)
        cand, _ = add_local_search(cand, order)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # Final polish: try to add any remaining element
    best_set = set(best)
    for x in UNIVERSE:
        if x not in best_set:
            best_set.add(x)
            if has_subset_sum_200(list(best_set)):
                best_set.remove(x)
    best = list(best_set)

    return best if best else [100]
