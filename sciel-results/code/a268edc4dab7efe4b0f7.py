import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin from 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)
    unit_fracs = [Fraction(1, d) for d in denominators]

    # Greedy initialization
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + unit_fracs[i] < 1:
            selected.append(i)
            total += unit_fracs[i]

    best_indices = selected[:]
    best_sum = total

    # Branch and Bound with suffix-sum pruning
    # Scale by LCM of all denominators to work with integers
    from math import gcd
    from functools import reduce

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, denominators)

    # Convert to integer weights: w[i] = L // denominators[i]
    # Target: sum of selected weights < L, maximize sum
    # Equivalently: find subset with max sum < L
    weights = [L // d for d in denominators]
    target = L  # we want sum < target

    # Suffix sums for pruning: suffix[i] = sum of weights[i:]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]

    best_int_sum = total * L  # current best as integer (total < 1, so this is < L)
    best_int_sum = int(best_int_sum)  # exact since total is rational with denominator dividing L

    # B&B search
    def branch_and_bound(pos, current_sum, selected_mask):
        nonlocal best_int_sum
        if time.time() > deadline:
            return

        # Pruning: even if we take all remaining, can't beat best
        if current_sum + suffix[pos] <= best_int_sum:
            return

        # Try including pos
        if current_sum + weights[pos] < target:
            new_sum = current_sum + weights[pos]
            if new_sum > best_int_sum:
                best_int_sum = new_sum
            # Recurse
            if pos + 1 < n:
                branch_and_bound(pos + 1, new_sum, selected_mask | (1 << pos))

        # Try excluding pos
        if pos + 1 < n:
            branch_and_bound(pos + 1, current_sum, selected_mask)

    # Run B&B
    branch_and_bound(0, 0, 0)

    # Reconstruct best solution from best_int_sum
    # Find which subset sums to best_int_sum
    # Since B&B found the max sum < L, we need to recover the indices
    # Re-run B&B but track the best mask
    best_mask = [0]
    best_int_sum2 = best_int_sum

    def branch_and_bound2(pos, current_sum, mask):
        nonlocal best_int_sum2
        if time.time() > deadline:
            return
        if current_sum + suffix[pos] <= best_int_sum2:
            return
        if current_sum + weights[pos] < target:
            new_sum = current_sum + weights[pos]
            if new_sum > best_int_sum2:
                best_int_sum2 = new_sum
                best_mask[0] = mask | (1 << pos)
            if pos + 1 < n:
                branch_and_bound2(pos + 1, new_sum, mask | (1 << pos))
        if pos + 1 < n:
            branch_and_bound2(pos + 1, current_sum, mask)

    branch_and_bound2(0, 0, 0)

    # Extract indices from best_mask
    bnb_indices = [i for i in range(n) if best_mask[0] & (1 << i)]
    if bnb_indices:
        bnb_sum = sum(unit_fracs[i] for i in bnb_indices)
        if bnb_sum < 1:
            best_indices = bnb_indices
            best_sum = bnb_sum

    # Local search refinement: 1-swap and 2-swap moves
    selected_set = set(best_indices)
    unselected = [i for i in range(n) if i not in selected_set]

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # 1-swap: remove one from selected, add one from unselected
        for i_out in list(best_indices):
            for i_in in unselected:
                new_total = best_sum - unit_fracs[i_out] + unit_fracs[i_in]
                if new_total < 1 and new_total > best_sum:
                    best_sum = new_total
                    best_indices = [i for i in best_indices if i != i_out] + [i_in]
                    best_indices.sort()
                    selected_set.remove(i_out)
                    selected_set.add(i_in)
                    unselected = [i for i in range(n) if i not in selected_set]
                    improved = True
                    break
            if improved:
                break

        if improved or time.time() >= deadline:
            continue

        # 2-swap: remove two from selected, add two from unselected
        if time.time() < deadline:
            sel_list = list(best_indices)
            unselected = [i for i in range(n) if i not in selected_set]

            for idx1 in range(len(sel_list)):
                if time.time() >= deadline:
                    break
                for idx2 in range(idx1 + 1, len(sel_list)):
                    if time.time() >= deadline:
                        break
                    i_out1 = sel_list[idx1]
                    i_out2 = sel_list[idx2]
                    base_total = best_sum - unit_fracs[i_out1] - unit_fracs[i_out2]

                    for i_in1 in unselected:
                        if time.time() >= deadline:
                            break
                        for i_in2 in unselected:
                            if i_in2 <= i_in1:
                                continue
                            new_total = base_total + unit_fracs[i_in1] + unit_fracs[i_in2]
                            if new_total < 1 and new_total > best_sum:
                                best_sum = new_total
                                best_indices = [i for i in best_indices if i != i_out1 and i != i_out2]
                                best_indices.append(i_in1)
                                best_indices.append(i_in2)
                                best_indices.sort()
                                selected_set.discard(i_out1)
                                selected_set.discard(i_out2)
                                selected_set.add(i_in1)
                                selected_set.add(i_in2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Simulated annealing escape: random perturbations to escape local optima
    if time.time() < deadline:
        import random
        random.seed(42)
        current_indices = best_indices[:]
        current_sum = best_sum
        current_set = set(current_indices)
        current_unselected = [i for i in range(n) if i not in current_set]

        for _ in range(200):
            if time.time() >= deadline:
                break

            # Random perturbation: swap 1-3 random pairs
            num_swaps = random.randint(1, 3)
            trial_indices = current_indices[:]
            trial_set = set(trial_indices)
            trial_unselected = current_unselected[:]
            trial_sum = current_sum

            valid_perturbation = True
            for _ in range(num_swaps):
                if not trial_indices or not trial_unselected:
                    valid_perturbation = False
                    break
                i_out = random.choice(trial_indices)
                i_in = random.choice(trial_unselected)
                new_trial_sum = trial_sum - unit_fracs[i_out] + unit_fracs[i_in]
                if new_trial_sum >= 1:
                    valid_perturbation = False
                    break
                trial_sum = new_trial_sum
                trial_indices = [i for i in trial_indices if i != i_out] + [i_in]
                trial_set.remove(i_out)
                trial_set.add(i_in)
                trial_unselected = [i for i in range(n) if i not in trial_set]

            if not valid_perturbation:
                continue

            # Accept if better, or with probability exp(-delta/T) for worse
            if trial_sum > best_sum:
                best_sum = trial_sum
                best_indices = trial_indices
                current_indices = trial_indices
                current_sum = trial_sum
                current_set = trial_set
                current_unselected = trial_unselected
            else:
                # Accept worse with some probability (annealing)
                deficit_worse = (1 - trial_sum) - (1 - best_sum)
                if deficit_worse > 0:
                    T = max(0.001, 0.1 * (1 - best_sum))
                    if random.random() < 0.3:  # simple acceptance for slightly worse
                        current_indices = trial_indices
                        current_sum = trial_sum
                        current_set = trial_set
                        current_unselected = trial_unselected

    return best_indices
