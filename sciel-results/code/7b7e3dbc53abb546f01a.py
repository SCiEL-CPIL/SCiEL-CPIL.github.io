import time
from fractions import Fraction
import random
import heapq
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Use LCM of all denominators for integer arithmetic
    # LCM of 2..48
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator: 1/d = (lcm/d) / lcm
    nums = [lcm_val // d for d in denominators]

    # Target: sum of selected nums < lcm_val
    # We want to maximize sum while keeping it < lcm_val

    def greedy_init():
        """Greedy: add largest unit fractions that keep sum < 1."""
        selected = []
        total = 0
        # Sort by value descending (smallest denominator first)
        for i in range(n):
            if total + nums[i] < lcm_val:
                selected.append(i)
                total += nums[i]
        return selected, total

    def local_search(selected, total):
        """Improve via 1-add, 1-swap, 1-for-2-swap, and 2-for-2-swap moves."""
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 0.3:
            improved = False

            # 1-add: try adding an unselected element
            for j in range(n):
                if j not in selected_set:
                    if total + nums[j] < lcm_val:
                        selected.append(j)
                        selected_set.add(j)
                        total += nums[j]
                        improved = True
                        break

            if improved:
                continue

            # 1-swap: remove i, add j where j > i in value
            best_swap = None
            best_swap_gain = 0
            for i in selected:
                for j in range(n):
                    if j not in selected_set:
                        gain = nums[j] - nums[i]
                        if gain > 0 and total - nums[i] + nums[j] < lcm_val:
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap = (i, j)
            if best_swap:
                i, j = best_swap
                selected.remove(i)
                selected_set.remove(i)
                selected.append(j)
                selected_set.add(j)
                total += nums[j] - nums[i]
                improved = True

            if improved:
                continue

            # 1-for-2-swap: remove i, add j and k
            best_swap2 = None
            best_swap2_gain = 0
            for i in selected:
                base = total - nums[i]
                for j in range(n):
                    if j not in selected_set:
                        base2 = base + nums[j]
                        if base2 >= lcm_val:
                            continue
                        for k in range(n):
                            if k not in selected_set and k != j:
                                gain = nums[j] + nums[k] - nums[i]
                                if gain > 0 and base2 + nums[k] < lcm_val:
                                    if gain > best_swap2_gain:
                                        best_swap2_gain = gain
                                        best_swap2 = (i, j, k)
            if best_swap2:
                i, j, k = best_swap2
                selected.remove(i)
                selected_set.remove(i)
                selected.append(j)
                selected_set.add(j)
                selected.append(k)
                selected_set.add(k)
                total += nums[j] + nums[k] - nums[i]
                improved = True

            if improved:
                continue

            # 2-for-2-swap: remove i and m, add j and k
            best_swap22 = None
            best_swap22_gain = 0
            sel_list = list(selected)
            for ii in range(len(sel_list)):
                i = sel_list[ii]
                for mm in range(ii + 1, len(sel_list)):
                    m = sel_list[mm]
                    base = total - nums[i] - nums[m]
                    for j in range(n):
                        if j not in selected_set:
                            base2 = base + nums[j]
                            if base2 >= lcm_val:
                                continue
                            for k in range(n):
                                if k not in selected_set and k != j:
                                    gain = nums[j] + nums[k] - nums[i] - nums[m]
                                    if gain > 0 and base2 + nums[k] < lcm_val:
                                        if gain > best_swap22_gain:
                                            best_swap22_gain = gain
                                            best_swap22 = (i, m, j, k)
            if best_swap22:
                i, m, j, k = best_swap22
                selected.remove(i)
                selected_set.remove(i)
                selected.remove(m)
                selected_set.remove(m)
                selected.append(j)
                selected_set.add(j)
                selected.append(k)
                selected_set.add(k)
                total += nums[j] + nums[k] - nums[i] - nums[m]
                improved = True

        return selected, total

    # Best result tracking
    best_indices = None
    best_sum = 0

    # Multiple random restarts with different orderings
    random.seed(42)

    # First try greedy
    sel, tot = greedy_init()
    sel, tot = local_search(sel, tot)
    best_indices = sel
    best_sum = tot

    # Random restarts with shuffled order
    while time.time() < deadline - 0.5:
        # Create a random permutation of indices
        perm = list(range(n))
        random.shuffle(perm)

        # Greedy with random order
        selected = []
        total = 0
        for i in perm:
            if total + nums[i] < lcm_val:
                selected.append(i)
                total += nums[i]

        # Local search
        selected, total = local_search(selected, total)

        if total > best_sum:
            best_indices = selected
            best_sum = total

    return best_indices
