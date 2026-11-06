import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_n = 512
    max_size = 48

    def is_valid(S):
        """
        Check if S is a valid set.
        No element y in S is the mean of any subset T of S\{y} with |T| >= 2.
        Condition: sum(T) != |T| * y for all T subset S\{y}, |T| >= 2.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return True

        for i, y in enumerate(S_list):
            elements = [x for j, x in enumerate(S_list) if j != i]
            m = len(elements)

            dp = [0] * (m + 1)
            dp[0] = 1
            current_max_k = 0

            for x in elements:
                for k in range(current_max_k, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
                current_max_k += 1

            for k in range(2, m + 1):
                target_sum = k * y
                if (dp[k] >> target_sum) & 1:
                    return False

        return True

    def greedy_build(order):
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1

        for x in order:
            if time.time() > deadline - 5.0:
                break
            if len(S) >= max_size:
                break

            safe = True
            for y in S:
                for k in range(1, len(S) + 1):
                    target_sum = y * (k + 1) - x
                    if target_sum < 0:
                        continue
                    if (dp[k] >> target_sum) & 1:
                        safe = False
                        break
                if not safe:
                    break

            if safe:
                S.append(x)
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)

        return S

    # 1. Generate initial candidates
    candidates = []

    # Descending
    candidates.append(greedy_build(range(max_n, 0, -1)))

    # Ascending
    candidates.append(greedy_build(range(1, max_n + 1)))

    # Random orders
    random.seed(42)
    for _ in range(10):
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        candidates.append(greedy_build(order))

    # Find the best initial set
    best_set = max(candidates, key=len)

    # 2. Local Search / Improvement
    current_set = list(best_set)
    current_size = len(current_set)

    all_numbers = set(range(1, max_n + 1))

    while time.time() < deadline - 10.0:
        if current_size >= max_size:
            break

        added = False
        remaining = list(all_numbers - set(current_set))
        random.shuffle(remaining)

        for x in remaining:
            new_set = current_set + [x]
            if is_valid(new_set):
                current_set = new_set
                current_size += 1
                added = True
                break

        if added:
            continue

        swapped = False
        for _ in range(50):
            if not current_set:
                break
            y = random.choice(current_set)
            x = random.choice(remaining)

            new_set = [v for v in current_set if v != y]
            new_set.append(x)

            if is_valid(new_set):
                current_set = new_set
                swapped = True
                break

        if not swapped and not added:
            break

    return current_set
