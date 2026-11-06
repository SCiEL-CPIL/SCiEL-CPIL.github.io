"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    # Build conflict graph: conflict[i][j] = True if lcm(i,j) <= 256
    # Numbers are 2..256, index 0..254
    N = 255  # numbers 2 to 256
    nums = list(range(2, 257))
    
    # Precompute conflicts
    conflict = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a, b = nums[i], nums[j]
            if lcm(a, b) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weight: 1/x for each number
    weights = [Fraction(1, nums[i]) for i in range(N)]
    
    # Split: small (2-16, indices 0-14) and large (17-256, indices 15-254)
    small_count = 15  # numbers 2 through 16
    large_start = small_count  # index 15 = number 17
    
    # For large numbers, precompute which small numbers they conflict with
    # and which large numbers they conflict with
    large_conflict_small = [0] * (N - large_start)  # bitmask of small conflicts
    large_conflict_large = [0] * (N - large_start)  # bitmask of large conflicts (only within large)
    
    for li in range(N - large_start):
        gi = li + large_start  # global index
        # Small conflicts
        mask = 0
        for si in range(small_count):
            if conflict[gi][si]:
                mask |= (1 << si)
        large_conflict_small[li] = mask
        # Large conflicts
        mask_l = 0
        for lj in range(N - large_start):
            if lj != li and conflict[gi][lj + large_start]:
                mask_l |= (1 << lj)
        large_conflict_large[li] = mask_l
    
    # Exact B&B for small numbers (15 numbers, 2^15 = 32768 states)
    # For each valid subset of small, compute best large selection greedily
    
    best_score = Fraction(0)
    best_small_mask = 0
    best_large_mask = 0
    
    def greedy_large(small_mask):
        """Given selected small numbers (as bitmask), greedily select large numbers."""
        # A large number li is compatible with current small selection if
        # (large_conflict_small[li] & small_mask) == 0
        # And among selected large numbers, no conflicts
        
        selected_large = set()
        # Sort large numbers by weight descending (1/x descending = x ascending)
        # For greedy: pick highest weight compatible first
        large_indices = list(range(N - large_start))
        large_indices.sort(key=lambda li: -weights[li + large_start])
        
        for li in large_indices:
            # Check small compatibility
            if large_conflict_small[li] & small_mask:
                continue
            # Check large compatibility with already selected
            compatible = True
            for sj in selected_large:
                if large_conflict_large[li] & (1 << sj):
                    compatible = False
                    break
            if compatible:
                selected_large.add(li)
        
        return selected_large
    
    # Enumerate all valid subsets of small numbers
    # Use backtracking for small numbers
    def enumerate_small(valid_mask, idx, current_mask, current_score):
        nonlocal best_score, best_small_mask, best_large_mask
        if time.time() > DEADLINE:
            return
        
        if idx == small_count:
            # Try greedy large
            large_sel = greedy_large(current_mask)
            total = current_score
            for li in large_sel:
                total += weights[li + large_start]
            if total > best_score:
                best_score = total
                best_small_mask = current_mask
                best_large_mask = 0
                for li in large_sel:
                    best_large_mask |= (1 << li)
            return
        
        # Option: don't include idx
        enumerate_small(valid_mask, idx + 1, current_mask, current_score)
        
        # Option: include idx if compatible with valid_mask
        # Check if idx conflicts with any in current_mask
        can_add = True
        for si in range(idx):
            if (current_mask & (1 << si)) and conflict[idx][si]:
                can_add = False
                break
        if can_add:
            enumerate_small(valid_mask, idx + 1, current_mask | (1 << idx), 
                          current_score + weights[idx])
    
    enumerate_small(0, 0, 0, Fraction(0))
    
    # Now try to improve: local search on the best solution found
    # Try swapping or adding/removing elements
    
    # Reconstruct current best
    current_small = set()
    for si in range(small_count):
        if best_small_mask & (1 << si):
            current_small.add(si)
    current_large = set()
    for li in range(N - large_start):
        if best_large_mask & (1 << li):
            current_large.add(li)
    
    # Local search: try to add any number not in set that's compatible
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try adding each non-selected number
        for i in range(N):
            if i in current_small or (i - large_start) in current_large if i >= large_start else False:
                continue
            # Check if i is already selected
            if i < large_start:
                if i in current_small:
                    continue
            else:
                if (i - large_start) in current_large:
                    continue
            
            # Check compatibility with all selected
            compatible = True
            for j in range(N):
                if j == i:
                    continue
                j_selected = (j < large_start and j in current_small) or \
                            (j >= large_start and (j - large_start) in current_large)
                if j_selected and conflict[i][j]:
                    compatible = False
                    break
            if compatible:
                # Add it
                if i < large_start:
                    current_small.add(i)
                else:
                    current_large.add(i - large_start)
                improved = True
    
    # Try removing pairs and adding better ones (2-opt style)
    if time.time() < DEADLINE - 5:
        # Try: for each pair in current set, try removing both and adding two new ones
        # This is expensive, so limit it
        all_selected = set()
        for si in current_small:
            all_selected.add(si)
        for li in current_large:
            all_selected.add(li + large_start)
        
        # Try single removals that allow additions
        for rem in list(all_selected):
            if time.time() > DEADLINE - 2:
                break
            # Try removing rem and see if we can add something better
            remaining = all_selected - {rem}
            # Find best compatible number to add
            best_add = None
            best_add_weight = Fraction(0)
            for cand in range(N):
                if cand in remaining:
                    continue
                compatible = True
                for j in remaining:
                    if conflict[cand][j]:
                        compatible = False
                        break
                if compatible and weights[cand] > best_add_weight:
                    best_add_weight = weights[cand]
                    best_add = cand
            
            if best_add is not None and best_add_weight > weights[rem]:
                # Make the swap
                if rem < large_start:
                    current_small.discard(rem)
                else:
                    current_large.discard(rem - large_start)
                if best_add < large_start:
                    current_small.add(best_add)
                else:
                    current_large.add(best_add - large_start)
    
    # Build final result
    result = []
    for si in current_small:
        result.append(nums[si])
    for li in current_large:
        result.append(nums[li + large_start])
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                # Fallback to greedy
                fallback = []
                for x in range(2, 257):
                    if all(lcm(x, y) > 256 for y in fallback):
                        fallback.append(x)
                return fallback
    
    return result

# EVOLVE-BLOCK-END
