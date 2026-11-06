<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # 78s safety margin
    
    # Build conflict graph
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts
    conflict_mask = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = nums[i], nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict_mask[i].add(j)
                conflict_mask[j].add(i)
    
    # Weights: 1/nums[i]
    weights = [1.0 / nums[i] for i in range(n)]
    
    def is_valid(selected_set, v):
        """Check if v can be added to selected_set"""
        for u in selected_set:
            if v in conflict_mask[u]:
                return False
        return True
    
    def total_weight(s):
        return sum(weights[i] for i in s)
    
    def local_search(selected, current_val, time_limit):
        """Improve solution via single swaps and add-only"""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try add-only first (cheapest)
            for add_i in range(n):
                if add_i not in selected and is_valid(selected, add_i):
                    selected = selected | {add_i}
                    current_val += weights[add_i]
                    improved = True
                    break
            
            if time.time() > time_limit:
                break
            
            # Try single swap
            for remove_i in list(selected):
                remaining = selected - {remove_i}
                best_add = -1
                best_gain = 0
                for add_i in range(n):
                    if add_i in remaining:
                        continue
                    if not is_valid(remaining, add_i):
                        continue
                    gain = weights[add_i] - weights[remove_i]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add_i
                if best_add >= 0:
                    selected = remaining | {best_add}
                    current_val += best_gain
                    improved = True
                    break
        
        return selected, current_val
    
    # Multiple restarts with different orderings
    best_selected = set()
    best_val = 0.0
    
    # Restart 1: greedy by weight (ascending num)
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = set()
    for i in order:
        if is_valid(selected, i):
            selected.add(i)
    selected, val = local_search(selected, total_weight(selected), deadline)
    if val > best_val:
        best_val = val
        best_selected = set(selected)
    
    # More restarts with perturbed orderings
    rng = random.Random(42)
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        # Perturbed order: mostly by weight, but with some randomness
        order = list(range(n))
        # Sort by weight but add random jitter
        order.sort(key=lambda i: -weights[i] + rng.uniform(0, 0.01))
        
        selected = set()
        for i in order:
            if is_valid(selected, i):
                selected.add(i)
        
        selected, val = local_search(selected, total_weight(selected), deadline - 2)
        if val > best_val:
            best_val = val
            best_selected = set(selected)
        
        if time.time() > deadline - 10:
            break
    
    result = sorted([nums[i] for i in best_selected])
    return result if result else [2]

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I'm replacing the same block twice. Let me just do one clean replacement.

Let me reconsider the approach. The local search with single swaps is O(n^2) per pass, and we might need many passes. With n=255, that's ~65k operations per pass. Even 1000 passes would be 65M, which is fine.

The key question is: how many restarts can we do? Each restart does greedy (O(n^2) = 65k) + local search (variable). If local search takes ~1 second per restart, we can do ~70 restarts. That's good.

Let me also add a **2-opt style** move: remove two consecutive elements and try to replace them with two better ones. But this is O(n^4) which is too slow. Skip it.

Actually, let me add a smarter local search: instead of just trying the best single swap, try ALL single swaps in one pass (not breaking early), which gives a steeper improvement.

