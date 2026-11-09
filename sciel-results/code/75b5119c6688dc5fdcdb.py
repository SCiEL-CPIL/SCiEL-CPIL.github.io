# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s kill
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    reciprocals = [1.0 / numbers[i] for i in range(n)]
    
    # Precompute conflict bitmasks
    conflict_mask = [0] * n
    for i in range(n):
        ai = numbers[i]
        for j in range(i + 1, n):
            aj = numbers[j]
            g = gcd(ai, aj)
            if ai * aj // g <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def can_add(idx, selected_mask):
        return (conflict_mask[idx] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(selected_mask):
        s = 0.0
        for i in range(n):
            if (selected_mask >> i) & 1:
                s += reciprocals[i]
        return s
    
    def greedy_construct(order):
        selected_mask = 0
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask |= (1 << idx)
        return selected_mask
    
    def local_search(selected_mask, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_sum = reciprocal_sum_from_mask(selected_mask)
            
            # Try adding any single element
            for idx in range(n):
                if not ((selected_mask >> idx) & 1):
                    if can_add(idx, selected_mask):
                        selected_mask |= (1 << idx)
                        current_sum += reciprocals[idx]
                        improved = True
                        break
            
            if time.time() >= time_limit:
                break
            
            # Try 1-for-1 swap: remove i (in set), add j (not in set)
            if not improved:
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if not ((selected_mask >> i) & 1):
                        continue
                    # i is in the set; try removing it
                    mask_without_i = selected_mask & ~(1 << i)
                    best_j = -1
                    best_gain = 0.0
                    for j in range(n):
                        if ((selected_mask >> j) & 1):
                            continue
                        # j is not in set; check if j can be added after removing i
                        if can_add(j, mask_without_i):
                            gain = reciprocals[j] - reciprocals[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                    if best_j >= 0 and best_gain > 1e-15:
                        selected_mask = mask_without_i | (1 << best_j)
                        current_sum += best_gain
                        improved = True
            
            # Try adding two elements simultaneously
            if not improved and time.time() < time_limit:
                best_pair = (-1, -1)
                best_gain = 0.0
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    if ((selected_mask >> i) & 1):
                        continue
                    if not can_add(i, selected_mask):
                        continue
                    for j in range(i + 1, n):
                        if ((selected_mask >> j) & 1):
                            continue
                        if not can_add(j, selected_mask | (1 << i)):
                            continue
                        gain = reciprocals[i] + reciprocals[j]
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (i, j)
                if best_pair != (-1, -1) and best_gain > 1e-15:
                    i, j = best_pair
                    selected_mask |= (1 << i) | (1 << j)
                    current_sum += best_gain
                    improved = True
        
        return selected_mask
    
    def verify(result):
        """Verify all pairs have lcm > 256."""
        for idx_a in range(len(result)):
            for idx_b in range(idx_a + 1, len(result)):
                a, b = result[idx_a], result[idx_b]
                if a * b // gcd(a, b) <= N:
                    return False
        return True
    
    best_mask = 0
    best_sum = 0.0
    
    # Deterministic first pass: sort by descending reciprocal (ascending number)
    order_sorted = list(range(n))
    mask = greedy_construct(order_sorted)
    mask = local_search(mask, deadline - 1.0)
    s = reciprocal_sum_from_mask(mask)
    if s > best_sum:
        best_sum = s
        best_mask = mask
    
    # Randomized restarts with diverse orderings
    trial = 0
    while time.time() < deadline - 1.0:
        trial += 1
        
        # Vary the ordering strategy
        strategy = trial % 3
        if strategy == 0:
            order = list(range(n))
            random.shuffle(order)
        elif strategy == 1:
            # Sort by reciprocal * (1 / (1 + conflict_count))
            order = sorted(range(n), key=lambda i: reciprocals[i] / (1.0 + bin(conflict_mask[i]).count('1')), reverse=True)
        else:
            # Random with bias toward low numbers
            order = list(range(n))
            random.shuffle(order)
            # Move some low-index elements forward
            for k in range(min(10, n)):
                pos = random.randint(0, n - 1)
                order.insert(random.randint(0, n - 1), order.pop(pos))
        
        mask = greedy_construct(order)
        mask = local_search(mask, deadline - 0.5)
        s = reciprocal_sum_from_mask(mask)
        if s > best_sum:
            best_sum = s
            best_mask = mask
        
        if trial >= 100:
            break
    
    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(numbers[i])
    
    # Verify and fix if needed
    if result and not verify(result):
        # Fall back to a safe single element
        result = [251]
    
    if not result:
        result = [251]
    
    return result
# EVOLVE-BLOCK-END
