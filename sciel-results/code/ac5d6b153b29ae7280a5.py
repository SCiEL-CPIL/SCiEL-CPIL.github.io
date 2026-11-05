import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # Safety margin before 90s kill

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
            g = gcd(a, b)
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

        # Try swapping: remove one selected, add one non-selected
        selected_list = get_selected_set(selected)
        for i in selected_list:
            if time.time() > DEADLINE - 10:
                break
            for j in range(n):
                if ((selected >> j) & 1) or j == i:
                    continue
                # Check if j is compatible with selected minus i
                temp_selected = selected & ~(1 << i)
                if (conflict_mask[j] & temp_selected) == 0:
                    new_weight = best_weight - weights[i] + weights[j]
                    if new_weight > best_weight:
                        new_mask = temp_selected | (1 << j)
                        best_selected = new_mask
                        best_weight = new_weight
                        improved = True
                        break

    # Phase 3: Iterated Local Search with randomization
    if time.time() < DEADLINE - 10:
        rng = random.Random(42)
        for iteration in range(100):
            if time.time() > DEADLINE - 5:
                break
            
            # Create a perturbed version of the best solution
            current_selected = best_selected
            current_weight = best_weight
            
            # Randomly remove 1-3 nodes
            selected_list = get_selected_set(current_selected)
            if len(selected_list) < 3:
                continue
                
            removals = min(rng.randint(1, 3), len(selected_list))
            to_remove = rng.sample(selected_list, removals)
            
            for i in to_remove:
                current_selected &= ~(1 << i)
                current_weight -= weights[i]
            
            # Try to greedily add back nodes
            for i in order:
                if not ((current_selected >> i) & 1):
                    if (conflict_mask[i] & current_selected) == 0:
                        current_selected |= (1 << i)
                        current_weight += weights[i]
            
            # Local improvement on the perturbed solution
            improved_local = True
            while improved_local and time.time() < DEADLINE - 5:
                improved_local = False
                
                # Try adding
                for i in range(n):
                    if not ((current_selected >> i) & 1):
                        if (conflict_mask[i] & current_selected) == 0:
                            new_mask = current_selected | (1 << i)
                            new_weight = current_weight + weights[i]
                            if new_weight > current_weight:
                                current_selected = new_mask
                                current_weight = new_weight
                                improved_local = True
                
                # Try swapping
                selected_list = get_selected_set(current_selected)
                for i in selected_list:
                    if time.time() > DEADLINE - 5:
                        break
                    for j in range(n):
                        if ((current_selected >> j) & 1) or j == i:
                            continue
                        temp_selected = current_selected & ~(1 << i)
                        if (conflict_mask[j] & temp_selected) == 0:
                            new_weight = current_weight - weights[i] + weights[j]
                            if new_weight > current_weight:
                                current_selected = temp_selected | (1 << j)
                                current_weight = new_weight
                                improved_local = True
                                break
            
            # Accept if better
            if current_weight > best_weight:
                best_selected = current_selected
                best_weight = current_weight

    # Convert back to numbers
    result = [nums[i] for i in range(n) if (best_selected >> i) & 1]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.lcm(result[i], result[j]) <= 256:
                # Fallback to greedy if verification fails
                return [nums[i] for i in range(n) if nums[i] > 128]
    
    return result
