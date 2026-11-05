import time
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78.0  # safety margin before 90s kill

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255 nodes, index 0..254

    # Precompute conflict bitmasks
    # conflict_mask[i] = bitmask of all j where lcm(nums[i], nums[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        a = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Weights
    weights = [1.0 / nums[i] for i in range(n)]

    # Check deadline
    if time.time() > DEADLINE:
        return [nums[i] for i in range(n) if nums[i] > 128]

    # Phase 1: Greedy initial solution
    # Process in order of decreasing weight (smallest numbers first)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    selected = 0  # bitmask of selected nodes
    for i in order:
        if (conflict_mask[i] & selected) == 0:
            selected |= (1 << i)

    best_selected = selected
    best_weight = sum(weights[i] for i in range(n) if (selected >> i) & 1)

    # Phase 2: Local search with add/swap moves
    def get_selected_set(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        selected = best_selected

        # Try adding each non-selected node
        for i in range(n):
            if not ((selected >> i) & 1):
                if (conflict_mask[i] & selected) == 0:
                    new_mask = selected | (1 << i)
                    new_weight = best_weight + weights[i]
                    if new_weight > best_weight:
                        best_selected = new_mask
                        best_weight = new_weight
                        selected = new_mask
                        improved = True

        # Try swap: remove one selected, add one non-selected
        sel_list = get_selected_set(selected)
        for i in sel_list:
            if time.time() > DEADLINE - 5:
                break
            # Remove i
            reduced = selected & ~(1 << i)
            # Try adding each non-selected j
            for j in range(n):
                if ((selected >> j) & 1) or j == i:
                    continue
                if (conflict_mask[j] & reduced) == 0:
                    new_weight = best_weight - weights[i] + weights[j]
                    if new_weight > best_weight:
                        best_selected = reduced | (1 << j)
                        best_weight = new_weight
                        selected = best_selected
                        sel_list = get_selected_set(selected)
                        improved = True
                        break  # restart inner loop with updated selection

        # Try remove one, add two
        if time.time() < DEADLINE - 10:
            sel_list = get_selected_set(selected)
            for i in sel_list:
                if time.time() > DEADLINE - 10:
                    break
                reduced = selected & ~(1 << i)
                # Find candidates that can be added to reduced
                candidates = []
                for j in range(n):
                    if ((selected >> j) & 1) or j == i:
                        continue
                    if (conflict_mask[j] & reduced) == 0:
                        candidates.append(j)
                # Try pairs from candidates
                for a_idx in range(len(candidates)):
                    j = candidates[a_idx]
                    for b_idx in range(a_idx + 1, len(candidates)):
                        k = candidates[b_idx]
                        if (conflict_mask[j] & (1 << k)) or (conflict_mask[k] & (1 << j)):
                            continue
                        new_weight = best_weight - weights[i] + weights[j] + weights[k]
                        if new_weight > best_weight:
                            best_selected = reduced | (1 << j) | (1 << k)
                            best_weight = new_weight
                            selected = best_selected
                            sel_list = get_selected_set(selected)
                            improved = True
                            break
                    if improved:
                        break

    # Final local search polish
    if time.time() < DEADLINE - 1:
        improved = True
        while improved and time.time() < DEADLINE - 1:
            improved = False
            selected = best_selected
            # Add moves
            for i in range(n):
                if not ((selected >> i) & 1):
                    if (conflict_mask[i] & selected) == 0:
                        new_weight = best_weight + weights[i]
                        if new_weight > best_weight:
                            best_selected = selected | (1 << i)
                            best_weight = new_weight
                            selected = best_selected
                            improved = True
            # Swap moves
            sel_list = [i for i in range(n) if (best_selected >> i) & 1]
            for i in sel_list:
                if time.time() > DEADLINE - 1:
                    break
                reduced = best_selected & ~(1 << i)
                for j in range(n):
                    if ((best_selected >> j) & 1) or j == i:
                        continue
                    if (conflict_mask[j] & reduced) == 0:
                        new_weight = best_weight - weights[i] + weights[j]
                        if new_weight > best_weight:
                            best_selected = reduced | (1 << j)
                            best_weight = new_weight
                            selected = best_selected
                            sel_list = [i for i in range(n) if (best_selected >> i) & 1]
                            improved = True
                            break

    # Convert to result
    result = [nums[i] for i in range(n) if (best_selected >> i) & 1]
    
    # Validate result
    is_valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a = result[i]
            b = result[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to known valid set: all numbers in [129, 256]
        return list(range(129, 257))
        
    return result
