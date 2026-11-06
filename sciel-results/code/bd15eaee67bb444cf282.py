"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import bisect
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48 for exact integer arithmetic
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
    
    # Integer weights: weight[i] = L // denominators[i], so sum of weights / L = reciprocal sum
    weights = [L // d for d in denominators]
    target = L  # we want sum of weights strictly below L, minimize L - sum

    # --- Phase 1: Branch and Bound with suffix sum pruning ---
    # Sort items by weight descending for better pruning
    indexed_weights = [(weights[i], i) for i in range(n)]
    indexed_weights.sort(key=lambda x: -x[0])
    sorted_weights = [w for w, _ in indexed_weights]
    sorted_indices = [i for _, i in indexed_weights]
    
    # Precompute suffix sums: suffix_sum[i] = sum of sorted_weights[i:]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]
    
    best_total = 0
    best_mask = 0  # bitmask over sorted positions
    
    def bnb_recursive(idx, current_sum, current_mask):
        nonlocal best_total, best_mask
        if time.time() > deadline - 5:
            return
        
        if idx == n:
            if current_sum > best_total and current_sum < target:
                best_total = current_sum
                best_mask = current_mask
            return
        
        # Prune: even if we take all remaining, can't beat best
        if current_sum + suffix_sum[idx] <= best_total:
            return
        
        # Prune: already over target
        if current_sum >= target:
            return
        
        # Option 1: Include item idx
        new_sum = current_sum + sorted_weights[idx]
        if new_sum < target:
            bnb_recursive(idx + 1, new_sum, current_mask | (1 << idx))
        
        # Option 2: Exclude item idx
        bnb_recursive(idx + 1, current_sum, current_mask)
    
    # Start B&B
    bnb_recursive(0, 0, 0)
    
    # Convert best_mask (over sorted positions) back to original indices
    bnb_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            bnb_indices.append(sorted_indices[i])
    
    # Validate and set as current best
    current_total = sum(weights[i] for i in bnb_indices)
    if current_total >= target or not bnb_indices:
        # Fallback to greedy
        greedy_selected = []
        greedy_total = 0
        for i in range(n):
            if greedy_total + weights[i] < target:
                greedy_selected.append(i)
                greedy_total += weights[i]
        bnb_indices = greedy_selected
        current_total = greedy_total
    
    best_indices = bnb_indices[:]
    best_deficit = target - current_total

    # --- Phase 2: Local Search Polish ---
    # Use the B&B result as starting point for local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = set(best_indices)
        current_total_val = sum(weights[i] for i in best_indices)
        current_deficit_val = target - current_total_val
        
        # Try adding any element
        for i in range(n):
            if i not in current_set:
                new_total = current_total_val + weights[i]
                if new_total < target:
                    new_deficit = target - new_total
                    if new_deficit < current_deficit_val:
                        best_indices.append(i)
                        current_set.add(i)
                        current_total_val = new_total
                        current_deficit_val = new_deficit
                        best_deficit = new_deficit
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try 1-swap
        for i_out in list(best_indices):
            total_without = current_total_val - weights[i_out]
            for i_in in range(n):
                if i_in in current_set:
                    continue
                new_total = total_without + weights[i_in]
                if new_total < target:
                    new_deficit = target - new_total
                    if new_deficit < current_deficit_val:
                        best_indices.remove(i_out)
                        best_indices.append(i_in)
                        current_set.remove(i_out)
                        current_set.add(i_in)
                        current_total_val = new_total
                        current_deficit_val = new_deficit
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved or time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
        
        # Try 2-swap (limited for time)
        idx_list = list(best_indices)
        if len(idx_list) >= 2:
            for i_out1, i_out2 in combinations(idx_list, 2):
                if time.time() > deadline - 5:
                    break
                total_without = current_total_val - weights[i_out1] - weights[i_out2]
                for i_in1 in range(n):
                    if i_in1 in current_set:
                        continue
                    for i_in2 in range(i_in1 + 1, n):
                        if i_in2 in current_set:
                            continue
                        new_total = total_without + weights[i_in1] + weights[i_in2]
                        if new_total < target:
                            new_deficit = target - new_total
                            if new_deficit < current_deficit_val:
                                best_indices.remove(i_out1)
                                best_indices.remove(i_out2)
                                best_indices.append(i_in1)
                                best_indices.append(i_in2)
                                current_set.discard(i_out1)
                                current_set.discard(i_out2)
                                current_set.add(i_in1)
                                current_set.add(i_in2)
                                current_total_val = new_total
                                current_deficit_val = new_deficit
                                best_deficit = new_deficit
                                improved = True
                                break
                    if improved or time.time() > deadline - 5:
                        break
                if improved or time.time() > deadline - 5:
                    break

    # --- Phase 3: Random Restarts with Local Search (if time remains) ---
    if time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(5000):
            if time.time() > deadline - 5:
                break
            subset_size = random.randint(5, 20)
            candidate = set(random.sample(range(n), subset_size))
            total = sum(weights[i] for i in candidate)
            if total >= target:
                continue
            deficit = target - total
            if deficit >= best_deficit:
                continue
            
            # Local improvement
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Add
                for i in range(n):
                    if i in candidate:
                        continue
                    new_total = total + weights[i]
                    if new_total < target:
                        new_deficit = target - new_total
                        if new_deficit < deficit:
                            candidate.add(i)
                            total = new_total
                            deficit = new_deficit
                            changed = True
                            break
                if changed:
                    continue
                # 1-swap
                for i_out in list(candidate):
                    total_without = total - weights[i_out]
                    for i_in in range(n):
                        if i_in in candidate:
                            continue
                        new_total = total_without + weights[i_in]
                        if new_total < target:
                            new_deficit = target - new_total
                            if new_deficit < deficit:
                                candidate.discard(i_out)
                                candidate.add(i_in)
                                total = new_total
                                deficit = new_deficit
                                changed = True
                                break
                    if changed:
                        break
            
            if deficit < best_deficit:
                best_indices = list(candidate)
                best_deficit = deficit

    # Final validation
    final_total = sum(weights[i] for i in best_indices)
    if final_total >= target or not best_indices:
        # Emergency fallback
        best_indices = [0]  # just 1/2
        final_total = weights[0]
    
    return best_indices
# EVOLVE-BLOCK-END
