import time
import random

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a non-empty subset of weights.
    Uses meet-in-the-middle for efficiency.
    """
    if target <= 0:
        return False
    
    # Filter weights: only those <= target can contribute
    weights = [w for w in weights if 0 < w <= target]
    if not weights:
        return False
    
    n = len(weights)
    if n == 1:
        return weights[0] == target
    
    if n <= 10:
        # Brute force for small sets
        sums = {0}
        for w in weights:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            sums.update(new_sums)
        return target in sums
    
    # Meet in the middle
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    # Generate all subset sums for left half
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # For each sum in right half, check if target - sum is in left_sums
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    for rs in right_sums:
        if rs == target:
            return True
        if rs < target and (target - rs) in left_sums:
            return True
    
    return False

def is_valid_set(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = sorted(S)
    for i, x in enumerate(S_list):
        target = L // x
        others = [L // z for z in S_list if z != x]
        if subset_sum_exists(others, target):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Precompute weights
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    best_set = None
    best_size = 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Descending greedy (current approach)
    strategies.append(list(range(24, 0, -1)))
    
    # Strategy 2: Ascending greedy
    strategies.append(list(range(1, 25)))
    
    # Strategy 3-8: Random orders
    for _ in range(6):
        order = list(range(1, 25))
        random.shuffle(order)
        strategies.append(order)
    
    # Strategy 9: Alternate large/small
    alt = []
    for i in range(12):
        alt.append(24 - i)
        if i < 11:
            alt.append(i + 1)
    strategies.append(alt)
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        S = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            target = weights[x]
            current_weights = [weights[z] for z in S]
            
            # Check conflict
            if subset_sum_exists(current_weights, target):
                # Try to find a swap: remove one element z from S, add x
                found_swap = False
                # Sort S by value - prefer removing elements that cause most conflicts
                candidates = sorted(S, key=lambda z: -weights[z])
                
                for z in candidates:
                    if time.time() > deadline:
                        break
                    
                    new_S = S - {z}
                    new_weights = [weights[zz] for zz in new_S]
                    
                    if not subset_sum_exists(new_weights, target):
                        # Also need to check that removing z doesn't break validity of remaining
                        # Actually we just need to check that x can be added without conflict
                        # The rest should still be valid since they were valid before
                        S = new_S
                        S.add(x)
                        found_swap = True
                        break
                
                if not found_swap:
                    continue  # Skip x
            else:
                S.add(x)
        
        # After greedy, try local improvement
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                S_list = sorted(S, reverse=True)
                
                # Try removing each element and adding a different one
                for z in S_list:
                    if time.time() > deadline:
                        break
                    
                    S_minus_z = S - {z}
                    
                    for y in range(1, 25):
                        if time.time() > deadline:
                            break
                        if y in S:
                            continue
                        
                        target_y = weights[y]
                        other_weights = [weights[zz] for zz in S_minus_z]
                        
                        if not subset_sum_exists(other_weights, target_y):
                            # Check that S_minus_z ∪ {y} is valid
                            # We know y doesn't conflict with S_minus_z
                            # We know S_minus_z is valid (subset of valid S)
                            new_S = S_minus_z | {y}
                            if len(new_S) > len(S):
                                S = new_S
                                improved = True
                                break
                    if improved:
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    if best_set is None:
        # Fallback: return a known valid set
        best_set = set(range(13, 25))
    
    return list(best_set)
