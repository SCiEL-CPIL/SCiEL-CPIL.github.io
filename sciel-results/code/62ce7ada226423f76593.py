import time
import sys
from math import gcd

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Vertices are 1..511
    vertices = list(range(1, N))
    weights = [0.0] * N
    for a in vertices:
        weights[a] = 1.0 / (N - a)
    
    # Sort vertices by weight descending (largest a first)
    sorted_vertices = sorted(vertices, key=lambda x: -weights[x])
    
    # Precompute coprimality matrix or just use gcd on the fly?
    # N=512 is small. gcd is fast.
    # But for local search, we need to check coprimality with the whole set often.
    # Let's maintain a set of selected numbers.
    
    # 1. Greedy Construction
    selected = []
    selected_set = set()
    
    for a in sorted_vertices:
        # Check if a is coprime to all in selected
        is_coprime = True
        for b in selected:
            if gcd(a, b) > 1:
                is_coprime = False
                break
        if is_coprime:
            selected.append(a)
            selected_set.add(a)
            
    current_sum = sum(weights[a] for a in selected)
    best_sum = current_sum
    best_selected = list(selected)
    
    # 2. Local Search (Add and Swap)
    # To speed up, we can precompute for each number its "conflicts" with others?
    # Or just check on the fly.
    # Since N is 512, checking against ~50 selected numbers is ~25k gcd ops per candidate.
    # 512 candidates * 25k ops = 12.8M ops. This is fast in Python if optimized?
    # Actually, 12.8M gcd calls might be slow.
    # Let's optimize: 
    # If we add a number, we remove its conflicts from the "available" pool.
    
    available = [a for a in vertices if a not in selected_set]
    
    # Helper to check if a is compatible with a set S
    def is_compatible(a, S):
        for b in S:
            if gcd(a, b) > 1:
                return False
        return True
        
    # Helper to get weight of set
    def calc_weight(S):
        return sum(weights[a] for a in S)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        # Sort available by weight descending to find best add quickly
        # But we just need ANY improvement, or the BEST?
        # Let's find the best possible add.
        best_add = -1
        best_add_weight = 0.0
        
        for a in available:
            if is_compatible(a, selected):
                if weights[a] > best_add_weight:
                    best_add_weight = weights[a]
                    best_add = a
                    
        if best_add != -1:
            selected.append(best_add)
            selected_set.add(best_add)
            available.remove(best_add)
            current_sum += weights[best_add]
            improved = True
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue # Restart loop to check more adds/swaps
        
        # Try to swap
        # For each a in selected, try to replace with b in available
        # Condition: b is compatible with selected \ {a}
        # Gain: weights[b] - weights[a]
        best_swap_a = -1
        best_swap_b = -1
        best_gain = 0.0
        
        # To speed up, we can precompute compatibility of available with selected?
        # Let's just iterate.
        for a in selected:
            # If a is the largest weight, swapping with smaller is bad.
            # But b must be coprime to others.
            # Quick check: if max(available) weight <= weights[a], skip? No, b might be better but smaller?
            # We want max weights[b] - weights[a].
            
            # Optimization: If the max weight in available is <= weights[a], 
            # we can't gain by swapping a with any b? 
            # No, we want weights[b] > weights[a].
            max_avail_w = max((weights[b] for b in available), default=0)
            if max_avail_w <= weights[a]:
                continue
                
            for b in available:
                if weights[b] <= weights[a]:
                    continue
                # Check if b is compatible with selected \ {a}
                is_cop = True
                for c in selected:
                    if c == a:
                        continue
                    if gcd(b, c) > 1:
                        is_cop = False
                        break
                if is_cop:
                    gain = weights[b] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap_a = a
                        best_swap_b = b
                        
        if best_swap_a != -1:
            # Perform swap
            selected.remove(best_swap_a)
            selected_set.remove(best_swap_a)
            available.append(best_swap_a)
            
            selected.append(best_swap_b)
            selected_set.add(best_swap_b)
            available.remove(best_swap_b)
            
            current_sum += best_gain
            improved = True
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue # Restart loop
        
        # No add, no swap. 
        # Maybe try a "remove" and then "add" different?
        # Or just stop if we are stuck.
        # To avoid getting stuck in local optima, we could do a random restart or perturbation.
        # But for now, let's try a simple perturbation if time remains.
        if time.time() < deadline - 5.0:
             # Perturb: Remove a random low-weight element from selected, add best available
             # This might escape local optima.
             if len(selected) > 1:
                 # Pick the smallest weight element in selected
                 min_idx = 0
                 min_w = weights[selected[0]]
                 for i in range(1, len(selected)):
                     if weights[selected[i]] < min_w:
                         min_w = weights[selected[i]]
                         min_idx = i
                 
                 a_to_remove = selected[min_idx]
                 
                 # Find best b in available compatible with selected \ {a_to_remove}
                 best_b = -1
                 best_w_b = 0.0
                 for b in available:
                     is_cop = True
                     for c in selected:
                         if c == a_to_remove:
                             continue
                         if gcd(b, c) > 1:
                             is_cop = False
                             break
                     if is_cop and weights[b] > best_w_b:
                         best_w_b = weights[b]
                         best_b = b
                         
                 if best_b != -1 and weights[best_b] > weights[a_to_remove]:
                     selected.remove(a_to_remove)
                     selected_set.remove(a_to_remove)
                     available.append(a_to_remove)
                     
                     selected.append(best_b)
                     selected_set.add(best_b)
                     available.remove(best_b)
                     
                     current_sum = current_sum - weights[a_to_remove] + weights[best_b]
                     improved = True
                     if current_sum > best_sum:
                         best_sum = current_sum
                         best_selected = list(selected)
                     continue
             
             # If perturbation didn't work or wasn't applied, we are likely at a local optimum.
             # Break to save time or try more aggressive restarts?
             # Given the strict time limit, breaking is safer.
             break
             
    # Final check: ensure we return the best found
    return best_selected
