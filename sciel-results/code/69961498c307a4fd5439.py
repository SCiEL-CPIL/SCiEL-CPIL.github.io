import time
import random
from itertools import combinations

L = 5354228880  # lcm(1..24)

def subset_sums_bounded(weights, target):
    """Returns a set of all possible subset sums of weights that are <= target."""
    sums = {0}
    for w in weights:
        if w > target:
            continue
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        sums.update(new_sums)
    return sums

def is_valid_set(S, weights):
    """
    Check if S is a valid reciprocal-subsum-free set.
    For each x in S, target = weights[x] must NOT be in subset_sums of {weights[z] for z in S, z != x}.
    """
    S_list = list(S)
    for i, x in enumerate(S_list):
        target = weights[x]
        other_weights = [weights[z] for z in S_list if z != x]
        if not other_weights:
            continue
        sums = subset_sums_bounded(other_weights, target)
        if target in sums:
            return False
    return True

def greedy_construct(order, weights, deadline):
    """
    Greedy construction following a specific order with conflict resolution.
    """
    S = set()
    
    for x in order:
        if time.time() > deadline:
            break
            
        target = weights[x]
        current_weights = [weights[z] for z in S]
        
        if not current_weights:
            S.add(x)
            continue
        
        possible_sums = subset_sums_bounded(current_weights, target)
        
        if target in possible_sums:
            # Conflict! Try single swap
            found_swap = False
            candidates = sorted(S, key=lambda z: weights[z], reverse=True)
            
            for z in candidates:
                if time.time() > deadline:
                    break
                new_weights = [weights[zz] for zz in S if zz != z]
                new_sums = subset_sums_bounded(new_weights, target)
                
                if target not in new_sums:
                    S.remove(z)
                    S.add(x)
                    if is_valid_set(S, weights):
                        found_swap = True
                        break
                    else:
                        S.remove(x)
                        S.add(z)
            
            if not found_swap:
                # Try double swap
                candidates = sorted(S, key=lambda z: weights[z], reverse=True)
                n = len(candidates)
                found_double_swap = False
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n):
                        z1, z2 = candidates[i], candidates[j]
                        new_weights = [weights[zz] for zz in S if zz != z1 and zz != z2]
                        new_sums = subset_sums_bounded(new_weights, target)
                        
                        if target not in new_sums:
                            S.discard(z1)
                            S.discard(z2)
                            S.add(x)
                            if is_valid_set(S, weights):
                                found_double_swap = True
                                break
                            else:
                                S.remove(x)
                                S.add(z1)
                                S.add(z2)
                        if found_double_swap:
                            break
                    if found_double_swap:
                        break
        else:
            S.add(x)
    
    return S

def local_improve(S, weights, deadline):
    """
    Try to improve S by adding elements or swapping.
    """
    S = set(S)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try adding each non-selected element
        for y in range(24, 0, -1):
            if time.time() > deadline:
                break
            if y in S:
                continue
            
            target_y = weights[y]
            current_weights = [weights[z] for z in S]
            
            if not current_weights:
                S.add(y)
                improved = True
                continue
            
            sums = subset_sums_bounded(current_weights, target_y)
            if target_y in sums:
                # Conflict, try swapping out some x in S
                for x in sorted(S, key=lambda z: weights[z]):
                    if time.time() > deadline:
                        break
                    
                    new_S = set(S)
                    new_S.remove(x)
                    new_S.add(y)
                    
                    if is_valid_set(new_S, weights):
                        S = new_S
                        improved = True
                        break
            else:
                S.add(y)
                improved = True
                break
    
    return S

def systematic_search(weights, deadline, min_size=12):
    """
    Systematic backtracking search for large valid sets.
    Process elements in descending order. For each element, decide include/exclude.
    Use forward checking: when including x, verify no conflict with already-included larger elements.
    """
    best = set()
    elements = list(range(24, 0, -1))
    
    def backtrack(idx, current_set, current_weights):
        nonlocal best
        
        if time.time() > deadline:
            return
        
        # Update best if current is larger
        if len(current_set) > len(best):
            best = set(current_set)
        
        # If we've considered all elements, return
        if idx >= len(elements):
            return
        
        x = elements[idx]
        target = weights[x]
        
        # Option 1: Include x
        if current_weights:
            sums = subset_sums_bounded(current_weights, target)
            if target not in sums:
                # Check that adding x doesn't break validity of existing elements
                # This is guaranteed since x is smaller than all in current_set,
                # and 1/x > 1/z for z in current_set, so x's weight is larger
                # and can't help form a sum equal to some 1/z (which is smaller)
                # Actually we need to verify: could 1/z = sum including 1/x?
                # 1/z < 1/x for z > x, so 1/x alone > 1/z, impossible.
                # So including x is safe for existing elements.
                new_set = current_set | {x}
                new_weights = current_weights + [target]
                backtrack(idx + 1, new_set, new_weights)
        
        # Option 2: Exclude x
        backtrack(idx + 1, current_set, current_weights)
    
    backtrack(0, set(), [])
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Precompute reciprocal weights
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    best_S = set()
    
    # Strategy 1: Systematic backtracking search (most thorough)
    S_sys = systematic_search(weights, deadline)
    if len(S_sys) > len(best_S):
        best_S = set(S_sys)
    
    # Strategy 2: Greedy with multiple orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
    ]
    
    random.seed(42)
    for _ in range(10):
        order = list(range(1, 25))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        S = greedy_construct(order, weights, deadline)
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Local improvement on the best set found
    if time.time() < deadline:
        best_S = local_improve(best_S, weights, deadline)
    
    # Final validation
    if not is_valid_set(best_S, weights):
        # Fallback: return a known valid set
        best_S = set(range(13, 25))
    
    return list(best_S)
