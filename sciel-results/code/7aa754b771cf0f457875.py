"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin
    
    # Denominators are [2, 3, ..., 48]
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, 3, ..., 48)
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
    
    # Precompute L/d for each denominator
    L_div_d = [L // d for d in denominators]
    
    # We want to minimize deficit = L - sum(L/d_i) where sum < L
    # Equivalently, maximize sum(L/d_i) subject to sum < L
    
    # Use BFS/DFS with pruning to find best subset
    # State: (index, current_sum, selected_indices)
    # We'll use a greedy + local search approach for speed
    
    # First, try a greedy approach: sort by value descending, add if fits
    # But we need to be careful - greedy might not be optimal
    
    # Better: use dynamic programming or branch and bound
    # Given 47 items, we can't do 2^47, so we need smart search
    
    # Strategy: 
    # 1. Start with a good initial solution using greedy
    # 2. Use local search to improve
    
    # Greedy initial solution: process denominators in order, add if sum stays < L
    def greedy_solution():
        selected = []
        current_sum = 0
        for i in range(n):
            if current_sum + L_div_d[i] < L:
                selected.append(i)
                current_sum += L_div_d[i]
        return selected, current_sum
    
    selected, current_sum = greedy_solution()
    best_deficit = L - current_sum
    
    # Local search: try swapping or adding/removing elements
    def local_search(selected, current_sum, max_iterations=100000):
        nonlocal best_deficit
        selected_set = set(selected)
        deficit = L - current_sum
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding one element
            for i in range(n):
                if i not in selected_set:
                    new_sum = current_sum + L_div_d[i]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < deficit:
                            selected_set.add(i)
                            current_sum = new_sum
                            deficit = new_deficit
                            best_deficit = min(best_deficit, deficit)
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try removing one element and adding two others
            for i in list(selected_set):
                new_sum = current_sum - L_div_d[i]
                # Try to add two elements
                for j in range(n):
                    if j == i or j in selected_set:
                        continue
                    sum_with_j = new_sum + L_div_d[j]
                    if sum_with_j >= L:
                        continue
                    for k in range(j+1, n):
                        if k in selected_set or k == i:
                            continue
                        sum_with_jk = sum_with_j + L_div_d[k]
                        if sum_with_jk < L:
                            new_deficit = L - sum_with_jk
                            if new_deficit < deficit:
                                selected_set.remove(i)
                                selected_set.add(j)
                                selected_set.add(k)
                                current_sum = sum_with_jk
                                deficit = new_deficit
                                best_deficit = min(best_deficit, deficit)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return list(selected_set), current_sum
    
    selected, current_sum = local_search(selected, current_sum)
    
    # Additional optimization: try more sophisticated swaps
    # Convert to list for easier manipulation
    selected_list = list(selected)
    selected_set = set(selected_list)
    
    # Try 1-for-1 swaps
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in selected_list:
            for j in range(n):
                if j in selected_set:
                    continue
                # Swap i with j
                new_sum = current_sum - L_div_d[i] + L_div_d[j]
                if new_sum < L:
                    new_deficit = L - new_sum
                    old_deficit = L - current_sum
                    if new_deficit < old_deficit:
                        selected_set.remove(i)
                        selected_set.add(j)
                        selected_list.remove(i)
                        selected_list.append(j)
                        current_sum = new_sum
                        best_deficit = min(best_deficit, new_deficit)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Try 2-for-1 and 1-for-2 swaps
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        
        # 2-for-1: remove two, add one
        for i_idx in range(len(selected_list)):
            for j_idx in range(i_idx+1, len(selected_list)):
                i = selected_list[i_idx]
                j = selected_list[j_idx]
                base_sum = current_sum - L_div_d[i] - L_div_d[j]
                for k in range(n):
                    if k in selected_set:
                        continue
                    new_sum = base_sum + L_div_d[k]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        old_deficit = L - current_sum
                        if new_deficit < old_deficit:
                            selected_set.discard(i)
                            selected_set.discard(j)
                            selected_set.add(k)
                            selected_list = list(selected_set)
                            current_sum = new_sum
                            best_deficit = min(best_deficit, new_deficit)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
        
        # 1-for-2: remove one, add two
        for i in selected_list:
            base_sum = current_sum - L_div_d[i]
            for j in range(n):
                if j in selected_set:
                    continue
                sum_with_j = base_sum + L_div_d[j]
                if sum_with_j >= L:
                    continue
                for k in range(j+1, n):
                    if k in selected_set:
                        continue
                    new_sum = sum_with_j + L_div_d[k]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        old_deficit = L - current_sum
                        if new_deficit < old_deficit:
                            selected_set.discard(i)
                            selected_set.add(j)
                            selected_set.add(k)
                            selected_list = list(selected_set)
                            current_sum = new_sum
                            best_deficit = min(best_deficit, new_deficit)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    if time.time() > deadline:
        # Return current best
        pass
    
    # Ensure we have a valid solution (sum < 1)
    if not selected:
        # Fallback: just pick denominator 2
        return [0]
    
    # Verify the solution is valid
    final_sum = sum(L_div_d[i] for i in selected)
    if final_sum >= L:
        # Shouldn't happen, but fallback
        return [0]
    
    return sorted(selected)
# EVOLVE-BLOCK-END
