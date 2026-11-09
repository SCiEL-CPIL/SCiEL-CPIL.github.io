import time
from fractions import Fraction
import sys
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute: for each x in [1,256], the set of pairs (a,b) such that {a,b,x} forms a 3-AP
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        pairs = []
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                    pairs.append((a, b))
        conflict_pairs[x] = pairs
    
    # Precompute reciprocal values as Fractions
    recips = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    # Greedy construction: add elements in order of increasing x (i.e., decreasing 1/x)
    selected_mask = 0
    selected_list = []
    
    for x in range(1, N + 1):
        can_add = True
        for (a, b) in conflict_pairs[x]:
            if (selected_mask >> (a - 1)) & 1 and (selected_mask >> (b - 1)) & 1:
                can_add = False
                break
        if can_add:
            selected_mask |= (1 << (x - 1))
            selected_list.append(x)
    
    current_sum = Fraction(0, 1)
    for x in selected_list:
        current_sum += recips[x]
    
    best_mask = selected_mask
    best_sum = current_sum
    
    # Helper: check if adding x to mask is valid
    def is_valid_add(mask, x):
        for (a, b) in conflict_pairs[x]:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                return False
        return True
    
    # Helper: get list of elements in mask
    def mask_to_list(mask):
        result = []
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                result.append(x)
        return result
    
    # Local search: try single additions, then swaps, then 2-for-1 removals
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try all single additions
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((best_mask >> (x - 1)) & 1):
                if is_valid_add(best_mask, x):
                    new_mask = best_mask | (1 << (x - 1))
                    new_sum = best_sum + recips[x]
                    if new_sum > best_sum:
                        best_mask = new_mask
                        best_sum = new_sum
                        improved = True
        
        # Try swaps: remove x, add y
        if time.time() > deadline:
            break
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if (best_mask >> (x - 1)) & 1:
                trial_mask = best_mask & ~(1 << (x - 1))
                for y in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if y == x:
                        continue
                    if not ((trial_mask >> (y - 1)) & 1):
                        if is_valid_add(trial_mask, y):
                            new_sum = best_sum - recips[x] + recips[y]
                            if new_sum > best_sum:
                                best_mask = trial_mask | (1 << (y - 1))
                                best_sum = new_sum
                                improved = True
                                break  # restart scan after a swap
                if improved:
                    break
        
        # Try 2-for-1: remove x, y, add z where z > max(x,y) so 1/z < 1/x + 1/y might not help,
        # but we need to check if removing x and y frees up z
        if time.time() > deadline:
            break
        current_list = mask_to_list(best_mask)
        for i in range(len(current_list)):
            if time.time() > deadline:
                break
            x = current_list[i]
            for j in range(i + 1, len(current_list)):
                if time.time() > deadline:
                    break
                y = current_list[j]
                trial_mask = best_mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                for z in range(1, N + 1):
                    if time.time() > deadline:
                        break
                    if z == x or z == y:
                        continue
                    if not ((trial_mask >> (z - 1)) & 1):
                        if is_valid_add(trial_mask, z):
                            new_sum = best_sum - recips[x] - recips[y] + recips[z]
                            if new_sum > best_sum:
                                best_mask = trial_mask | (1 << (z - 1))
                                best_sum = new_sum
                                improved = True
                                break  # restart after improvement
                if improved:
                    break
            if improved:
                break
    
    # Extract final list
    final_list = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            final_list.append(x)
    
    return final_list
