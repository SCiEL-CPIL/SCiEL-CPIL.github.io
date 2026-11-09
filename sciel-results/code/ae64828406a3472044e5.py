"""Optimized solver for closest strict underpacking using MILP + local search."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 items

    # Compute LCM of all denominators
    lcm_val = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    
    # Weights: LCM // denominator
    weights = [lcm_val // d for d in denominators]
    target = lcm_val  # sum of weights must be strictly less than this

    # Start with greedy solution
    selected = set()
    current_sum = 0
    for i in range(n):
        if current_sum + weights[i] < target:
            selected.add(i)
            current_sum += weights[i]
    
    best_sum = current_sum
    best_selected = set(selected)
    
    # Try to improve with local search using integer arithmetic
    # 1-add: try adding an unselected item
    # 1-swap: try swapping one selected for one unselected
    # 1-for-2-swap: try replacing one selected with two unselected
    
    def time_exceeded():
        return time.time() - start_time > DEADLINE

    # Local search loop
    improved = True
    while improved and not time_exceeded():
        improved = False
        
        # 1-add: try adding any unselected item
        for i in range(n):
            if i not in selected:
                if current_sum + weights[i] < target:
                    selected.add(i)
                    current_sum += weights[i]
                    improved = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_selected = set(selected)
                    break  # restart after one improvement
        
        if time_exceeded():
            break
            
        # 1-swap: try swapping one selected for one unselected (net gain)
        if not improved:
            for i in list(selected):
                for j in range(n):
                    if j not in selected:
                        new_sum = current_sum - weights[i] + weights[j]
                        if new_sum < target and new_sum > current_sum:
                            selected.remove(i)
                            selected.add(j)
                            current_sum = new_sum
                            improved = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_selected = set(selected)
                            break
                if improved or time_exceeded():
                    break
        
        if time_exceeded():
            break
            
        # 1-for-2-swap: replace one selected with two unselected
        if not improved:
            for i in list(selected):
                for j in range(n):
                    if j not in selected:
                        for k in range(n):
                            if k not in selected and k != j:
                                new_sum = current_sum - weights[i] + weights[j] + weights[k]
                                if new_sum < target and new_sum > current_sum:
                                    selected.remove(i)
                                    selected.add(j)
                                    selected.add(k)
                                    current_sum = new_sum
                                    improved = True
                                    if current_sum > best_sum:
                                        best_sum = current_sum
                                        best_selected = set(selected)
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved or time_exceeded():
                    break

    # Use remaining time for more aggressive local search or random restarts
    if not time_exceeded():
        # Try multiple random restarts with local search
        import random
        random.seed(42)
        
        for _ in range(20):
            if time_exceeded():
                break
                
            # Random initial solution
            selected = set()
            current_sum = 0
            # Start with a random subset
            for i in range(n):
                if random.random() < 0.5 and current_sum + weights[i] < target:
                    selected.add(i)
                    current_sum += weights[i]
            
            # Local search from this starting point
            local_improved = True
            while local_improved and not time_exceeded():
                local_improved = False
                
                # 1-add
                for i in range(n):
                    if i not in selected and current_sum + weights[i] < target:
                        selected.add(i)
                        current_sum += weights[i]
                        local_improved = True
                        break
                
                if time_exceeded():
                    break
                    
                # 1-swap
                if not local_improved:
                    for i in list(selected):
                        for j in range(n):
                            if j not in selected:
                                new_sum = current_sum - weights[i] + weights[j]
                                if new_sum < target and new_sum > current_sum:
                                    selected.remove(i)
                                    selected.add(j)
                                    current_sum = new_sum
                                    local_improved = True
                                    break
                        if local_improved or time_exceeded():
                            break
                
                if time_exceeded():
                    break
                    
                # 1-for-2-swap
                if not local_improved:
                    for i in list(selected):
                        for j in range(n):
                            if j not in selected:
                                for k in range(n):
                                    if k not in selected and k != j:
                                        new_sum = current_sum - weights[i] + weights[j] + weights[k]
                                        if new_sum < target and new_sum > current_sum:
                                            selected.remove(i)
                                            selected.add(j)
                                            selected.add(k)
                                            current_sum = new_sum
                                            local_improved = True
                                            break
                                    if local_improved:
                                        break
                                if local_improved:
                                    break
                            if local_improved:
                                break
                        if local_improved or time_exceeded():
                            break
                
                if time_exceeded():
                    break
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = set(selected)

    # Return the best solution found
    return sorted(best_selected)
# EVOLVE-BLOCK-END
