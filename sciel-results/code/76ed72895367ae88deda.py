import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    def is_valid(subset):
        # Bitset DP: bit s is set iff some subset sums to s
        bits = 1  # bit 0 set
        for x in subset:
            if x > 200:
                continue
            bits |= (bits << x)
            if bits & (1 << 200):
                return False
        return True

    def greedy_build(order):
        selected = []
        for x in order:
            if is_valid(selected + [x]):
                selected.append(x)
        return selected

    def local_search(initial_set):
        S = set(initial_set)
        U = set(range(1, 101)) - S

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: Try to add elements (sorted descending for efficiency)
            for x in sorted(U, reverse=True):
                if time.time() > deadline:
                    return S
                if is_valid(list(S) + [x]):
                    S.add(x)
                    U.remove(x)
                    improved = True

            # Phase 2: Try swaps if no additions were made
            if not improved:
                u_list = list(U)
                random.shuffle(u_list)
                for x in u_list:
                    if time.time() > deadline:
                        return S

                    s_list = list(S)
                    random.shuffle(s_list)
                    for y in s_list:
                        new_S = (S - {y}) | {x}
                        if is_valid(list(new_S)):
                            S = new_S
                            U = set(range(1, 101)) - S
                            improved = True
                            break
                    if improved:
                        break
        return S

    best_set = set()

    # Run multiple random restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break

        if i == 0:
            # First run: descending (largest first — safest)
            order = list(range(100, 0, -1))
        elif i == 1:
            # Second run: ascending
            order = list(range(1, 101))
        elif i == 2:
            # Third run: even numbers descending, then odd descending
            evens_desc = list(range(100, 0, -2))
            odds_desc = list(range(99, 0, -2))
            order = evens_desc + odds_desc
        else:
            # Randomized with bias toward larger numbers
            r = random.random()
            if r < 0.3:
                order = list(range(100, 0, -1))
                # Perturb with random swaps
                for _ in range(random.randint(5, 20)):
                    if len(order) > 1:
                        idx = random.randint(0, len(order) - 1)
                        jdx = random.randint(0, len(order) - 1)
                        order[idx], order[jdx] = order[jdx], order[idx]
            elif r < 0.6:
                # Weighted: larger numbers more likely to come first
                order = list(range(1, 101))
                random.shuffle(order)
                order.sort(key=lambda x: -x + random.randint(-30, 30))
            else:
                order = list(range(1, 101))
                random.shuffle(order)

        current_set = greedy_build(order)
        current_set = local_search(current_set)

        if len(current_set) > len(best_set):
            best_set = current_set

    return list(best_set)
