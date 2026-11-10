"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 256
    # Numbers are 2..256, index 0 = number 2, index 1 = number 3, ..., index 254 = number 256
    nums = list(range(2, 257))
    n = len(nums)  # 255 numbers
    
    # Precompute incompatibility: inc[i][j] = True if lcm(nums[i], nums[j]) <= 256
    # We only need to check pairs
    inc = [[False] * n for _ in range(n)]
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            aj = nums[j]
            if lcm(ai, aj) <= 256:
                inc[i][j] = True
                inc[j][i] = True
    
    def is_compatible_with_all(idx, selected_set):
        """Check if nums[idx] is compatible with all in selected_set (set of indices)"""
        for j in selected_set:
            if inc[idx][j]:
                return False
        return True
    
    def reciprocal_sum(selected_indices):
        return sum(1.0 / nums[i] for i in selected_indices)
    
    # Greedy initialization: add numbers in order of smallest value (largest reciprocal)
    # Sort by value ascending
    order = sorted(range(n), key=lambda i: nums[i])
    
    selected = set()
    for i in order:
        if is_compatible_with_all(i, selected):
            selected.add(i)
    
    best_set = set(selected)
    best_score = reciprocal_sum(selected)
    
    # Local search: try swapping each selected element with each unselected element
    # Also try adding unselected elements and removing selected elements
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Try all swaps: remove one selected, add one unselected
        sel_list = list(selected)
        unsel_list = [i for i in range(n) if i not in selected]
        
        for si in sel_list:
            if (time.time() - start_time) > deadline:
                break
            # Try removing si
            new_selected = selected - {si}
            for ui in unsel_list:
                if ui in new_selected:
                    continue
                # Check if ui is compatible with all in new_selected
                if is_compatible_with_all(ui, new_selected):
                    candidate = new_selected | {ui}
                    score = reciprocal_sum(candidate)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_set = set(candidate)
                        selected = set(candidate)
                        improved = True
                        unsel_list = [i for i in range(n) if i not in selected]
                        break  # restart inner loop after improvement
        
        # Also try just adding (if we can fit more)
        if (time.time() - start_time) > deadline:
            break
        unsel_list = [i for i in range(n) if i not in selected]
        for ui in unsel_list:
            if is_compatible_with_all(ui, selected):
                candidate = selected | {ui}
                score = reciprocal_sum(candidate)
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = set(candidate)
                    selected = set(candidate)
                    improved = True
    
    # Return the best set as the actual numbers
    return [nums[i] for i in sorted(best_set)]
# EVOLVE-BLOCK-END
