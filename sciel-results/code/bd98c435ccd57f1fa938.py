import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of 2..48
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Weights in the scaled integer space
    weights = [lcm_val // d for d in denominators]

    # Sort indices by weight descending
    indices = list(range(n))
    sorted_indices = sorted(indices, key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in sorted_indices]

    # Suffix sums for upper bound pruning
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_weights[i]

    # Greedy initialization
    best_sum = 0
    best_selected = [False] * n

    current_sum = 0
    greedy_sel = [False] * n
    for i in range(n):
        if current_sum + sorted_weights[i] < lcm_val:
            current_sum += sorted_weights[i]
            greedy_sel[i] = True

    if current_sum > best_sum:
        best_sum = current_sum
        best_selected = greedy_sel[:]

    sys.setrecursionlimit(10000)

    def dfs(idx, current_sum, selected):
        nonlocal best_sum, best_selected

        if time.time() > deadline:
            return

        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = selected[:]
            return

        # Pruning: If even taking all remaining items doesn't beat best_sum
        if current_sum + suffix[idx] <= best_sum:
            return

        # Option 1: Include item idx
        if current_sum + sorted_weights[idx] < lcm_val:
            selected[idx] = True
            dfs(idx + 1, current_sum + sorted_weights[idx], selected)
            selected[idx] = False

        # Option 2: Exclude item idx
        dfs(idx + 1, current_sum, selected)

    selected = [False] * n
    dfs(0, 0, selected)

    # Map back to original indices
    result_indices = []
    for i in range(n):
        if best_selected[i]:
            result_indices.append(sorted_indices[i])

    return result_indices
