"""Optimized program for 'No element dividing a sum of three others' task.
Strategy: Fast greedy with multiple randomized orderings, then local search with 1-out-2-in swaps.
Key optimization: Use residue counting for small moduli to speed up conflict checks.
"""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def can_add_optimized(selected, candidate):
        """Check if candidate can be added to selected (list of ints)."""
        n = len(selected)
        
        # Check 1: no triple from selected has sum divisible by candidate
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    if (a + b + c) % candidate == 0:
                        return False
        
        # Check 2: for each a in selected, no pair (b,c) from selected\{a} 
        #           has (b+c+candidate) % a == 0
        for i in range(n):
            a = selected[i]
            target = (-candidate) % a
            
            # Optimization: Use residue counting for smaller a to avoid O(N^2) pair checks
            if a < 150:
                res_count = {}
                for j in range(n):
                    if j == i: continue
                    r = selected[j] % a
                    res_count[r] = res_count.get(r, 0) + 1
                
                for j in range(n):
                    if j == i: continue
                    b = selected[j]
                    rb = b % a
                    needed = (target - rb) % a
                    if needed == rb:
                        if res_count[needed] >= 2:
                            return False
                    else:
                        if needed in res_count and res_count[needed] >= 1:
                            return False
            else:
                for j in range(n):
                    if j == i: continue
                    b = selected[j]
                    for k in range(j+1, n):
                        if k == i: continue
                        c = selected[k]
                        if (b + c + candidate) % a == 0:
                            return False
        
        return True

    # Fast greedy: try multiple orderings
    best = []
    best_len = 0
    
    # Ordering 1: descending
    order = list(range(192, 0, -1))
    selected = []
    for x in order:
        if time.time() > deadline - 10:
            break
        if len(selected) >= 48:
            break
        if can_add_optimized(selected, x):
            selected.append(x)
    if len(selected) > best_len:
        best = selected[:]
        best_len = len(selected)
    
    # Ordering 2: descending with small perturbation
    random.seed(42)
    order = list(range(192, 0, -1))
    random.shuffle(order)
    # But keep large numbers somewhat prioritized
    order.sort(key=lambda x: -x + random.uniform(-10, 10))
    selected = []
    for x in order:
        if time.time() > deadline - 10:
            break
        if len(selected) >= 48:
            break
        if can_add_optimized(selected, x):
            selected.append(x)
    if len(selected) > best_len:
        best = selected[:]
        best_len = len(selected)
    
    # More randomized orderings
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        random.seed(1000 + trial)
        order = list(range(192, 0, -1))
        random.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline - 20:
                break
            if len(selected) >= 48:
                break
            if can_add_optimized(selected, x):
                selected.append(x)
        if len(selected) > best_len:
            best = selected[:]
            best_len = len(selected)
        if best_len >= 48:
            break
    
    # Local search: 1-out-2-in swaps
    if best_len < 48 and time.time() < deadline - 5:
        current = best[:]
        current_len = best_len
        improved = True
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                without_a = [v for j, v in enumerate(current) if j != i]
                # Try to add two new elements
                for b in range(192, 0, -1):
                    if b in without_a:
                        continue
                    if not can_add_optimized(without_a, b):
                        continue
                    with_b = without_a + [b]
                    for c in range(192, 0, -1):
                        if c in with_b:
                            continue
                        if can_add_optimized(with_b, c):
                            new_set = with_b + [c]
                            if len(new_set) > current_len:
                                current = new_set
                                current_len = len(current)
                                best = current[:]
                                best_len = current_len
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if current_len >= 48:
                break
    
    # Final polish: try to add any remaining elements
    if best_len < 48 and time.time() < deadline - 2:
        current = best[:]
        current_len = best_len
        for x in range(192, 0, -1):
            if time.time() > deadline - 2:
                break
            if len(current) >= 48:
                break
            if x in current:
                continue
            if can_add_optimized(current, x):
                current.append(x)
                current_len += 1
        if current_len > best_len:
            best = current[:]
            best_len = current_len
    
    return best[:48] if len(best) >= 48 else best

# EVOLVE-BLOCK-END
