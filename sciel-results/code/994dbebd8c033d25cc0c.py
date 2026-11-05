"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]

    # Suffix sums for pruning
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]

    # Greedy + local search for a strong initial solution
    def get_initial_solution():
        selected = []
        total = Fraction(0)
        for i in range(n):
            if total + values[i] < 1:
                selected.append(i)
                total += values[i]

        # Local search: 1-for-1 swaps
        for _ in range(20):
            improved = False
            sel_set = set(selected)
            for add_i in range(n):
                if add_i in sel_set:
                    continue
                for j, rem_i in enumerate(selected):
                    new_total = total - values[rem_i] + values[add_i]
                    if new_total < 1 and new_total > total:
                        selected[j] = add_i
                        total = new_total
                        sel_set.discard(rem_i)
                        sel_set.add(add_i)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        # 2-for-1 swaps
        for _ in range(5):
            improved = False
            sel_set = set(selected)
            for rem_i in selected:
                for add_i in range(n):
                    if add_i in sel_set or add_i == rem_i:
                        continue
                    for add_j in range(n):
                        if add_j in sel_set or add_j == rem_i or add_j == add_i:
                            continue
                        new_total = total - values[rem_i] + values[add_i] + values[add_j]
                        if new_total < 1 and new_total > total:
                            selected.remove(rem_i)
                            selected.append(add_i)
                            selected.append(add_j)
                            total = new_total
                            sel_set.discard(rem_i)
                            sel_set.add(add_i)
                            sel_set.add(add_j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return selected, total

    best_indices, best_sum = get_initial_solution()

    # Branch and bound
    import sys
    sys.setrecursionlimit(100000)

    def bnb(i, current_sum, selected):
        nonlocal best_sum, best_indices
        if time.time() - start_time > DEADLINE:
            return

        # Prune: even taking all remaining can't beat best
        if current_sum + suffix[i] <= best_sum:
            return

        if i == n:
            if current_sum > best_sum and current_sum < 1:
                best_sum = current_sum
                best_indices = list(selected)
            return

        # Try taking item i first
        new_sum = current_sum + values[i]
        if new_sum < 1:
            selected.append(i)
            bnb(i + 1, new_sum, selected)
            selected.pop()

        # Try skipping item i
        bnb(i + 1, current_sum, selected)

    bnb(0, Fraction(0), [])

    if not best_indices:
        best_indices = [0]

    return best_indices
# EVOLVE-BLOCK-END
