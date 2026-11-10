import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit

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

    def is_valid(selected):
        return not has_subset_sum_200(selected)

    # Structural insight: {66,...,100} is valid (size 35)
    # Any 2 elements sum to at most 199, any 3 elements sum to at least 201
    structural_seed = list(range(66, 101))

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

    def two_swap_local_search(selected, max_rounds=50):
        """Try removing two elements and adding two different ones to increase size."""
        selected_set = set(selected)
        missing = [x for x in UNIVERSE if x not in selected_set]
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(missing)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(missing)):
                    if time.time() > deadline - 3:
                        break
                    a, b = missing[i], missing[j]
                    # Try to find two elements to remove
                    for x in list(selected_set):
                        if time.time() > deadline - 2:
                            break
                        for y in list(selected_set):
                            if x == y:
                                continue
                            if time.time() > deadline - 1:
                                break
                            selected_set.remove(x)
                            selected_set.remove(y)
                            selected_set.add(a)
                            selected_set.add(b)
                            if is_valid(list(selected_set)):
                                # Size unchanged, but might enable further additions
                                # Try to add more
                                new_sel, imp = add_local_search(list(selected_set), UNIVERSE, max_rounds=5)
                                if len(new_sel) > len(selected_set):
                                    selected_set = set(new_sel)
                                    improved = True
                                    break
                                else:
                                    # Revert
                                    selected_set.remove(a)
                                    selected_set.remove(b)
                                    selected_set.add(x)
                                    selected_set.add(y)
                            else:
                                # Revert
                                selected_set.remove(a)
                                selected_set.remove(b)
                                selected_set.add(x)
                                selected_set.add(y)
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return list(selected_set)

    def remove_and_readd(selected, order, max_rounds=10):
        """Remove one element, then try to add multiple others."""
        selected_set = set(selected)
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for x in order:
                if x not in selected_set:
                    continue
                selected_set.remove(x)
                new_set, _ = add_local_search(list(selected_set), order, max_rounds=5)
                if len(new_set) > len(selected_set) + 1:
                    # We removed x (size decreased by 1), if new_set > original size, good
                    # Actually selected_set has x removed, so len(selected_set) = original - 1
                    # new_set should be > original size for improvement
                    if len(new_set) > len(selected_set) + 1:
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

    def update_best(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)

    # Start with structural seed
    update_best(structural_seed)

    # Try various orderings with greedy
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
        list(range(1, N // 2 + 1)) + list(range(N // 2 + 1, N + 1)),
        list(range(N, N // 2, -1)) + list(range(1, N // 2 + 1)),
        list(range(1, N // 2 + 1)) + list(range(N, N // 2, -1)),
        list(range(66, N + 1)) + list(range(1, 66)),
        list(range(1, 66)) + list(range(66, N + 1)),
    ]

    rng = random.Random(42)
    for seed in range(60):
        if time.time() > deadline - 15:
            break
        order = list(UNIVERSE)
        rng = random.Random(seed)
        rng.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline - 10:
            break
        cand = greedy(order)
        cand, _ = add_local_search(cand, order)
        cand = remove_and_readd(cand, order)
        cand, _ = add_local_search(cand, order)
        update_best(cand)

    # Try starting from structural seed and doing local search
    if time.time() < deadline - 10:
        cand = list(structural_seed)
        cand, _ = add_local_search(cand, UNIVERSE)
        cand = remove_and_readd(cand, UNIVERSE)
        cand, _ = add_local_search(cand, UNIVERSE)
        update_best(cand)

    # Simulated annealing-style local search on best
    if time.time() < deadline - 5:
        best_set = set(best)
        missing = [x for x in UNIVERSE if x not in best_set]
        temp = 10.0
        for iteration in range(500):
            if time.time() > deadline - 3:
                break
            if not missing:
                break
            # Randomly pick an element to try adding
            idx = random.Random(iteration).randint(0, len(missing) - 1)
            x = missing[idx]
            best_set.add(x)
            if is_valid(list(best_set)):
                best_set.add(x)
                missing.remove(x)
                if len(best_set) > best_len:
                    best = list(best_set)
                    best_len = len(best_set)
            else:
                best_set.remove(x)
                # Try removing a random element from best_set to make room
                if best_set:
                    remove_idx = random.Random(iteration + 1000).randint(0, len(best_set) - 1)
                    y = list(best_set)[remove_idx]
                    best_set.remove(y)
                    best_set.add(x)
                    if is_valid(list(best_set)):
                        if len(best_set) > best_len:
                            best = list(best_set)
                            best_len = len(best_set)
                        missing.remove(x)
                        missing.append(y)
                    else:
                        best_set.remove(x)
                        best_set.add(y)

    # Final polish: try to add any remaining element
    best_set = set(best)
    for x in UNIVERSE:
        if x not in best_set:
            best_set.add(x)
            if has_subset_sum_200(list(best_set)):
                best_set.remove(x)
    best = list(best_set)

    return best if best else [100]
