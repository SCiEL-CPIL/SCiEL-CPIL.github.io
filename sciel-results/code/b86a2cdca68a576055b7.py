import time
import random
import sys

L = 5354228880  # lcm(1..24)

def subset_sums_bounded(weights, target):
    """
    Returns a set of all possible subset sums of weights that are <= target.
    Weights are positive integers.
    """
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
    for x in S:
        target = weights[x]
        other_weights = [weights[z] for z in S if z != x]
        sums = subset_sums_bounded(other_weights, target)
        if target in sums:
            return False
    return True

def find_conflict(S, weights):
    """
    Returns (x, Y) if 1/x == sum(1/y for y in Y) for Y subset of S\{x}, else None.
    """
    for x in S:
        target = weights[x]
        other_weights = {z: weights[z] for z in S if z != x}
        # We need to find if target is in subset sums of other_weights
        # And if so, identify which elements formed it?
        # For simplicity, just return x and the set of others involved if we track it.
        # But is_valid_set just checks existence.
        # Let's just return x and the list of others that can form it.
        
        # To find the specific subset, we can do a DFS or backtrack.
        # But for removal heuristic, we just need to know WHICH elements to remove.
        # If we just remove x, we break the conflict.
        # If we remove one from Y, we break it.
        
        # Let's just return x for now, and assume we might remove x or a random other.
        # Better: find the specific subset Y.
        
        # Modified subset_sums to track path?
        # Or just check existence.
        if target in subset_sums_bounded(list(other_weights.values()), target):
            # Conflict exists.
            # We need to identify Y to choose which element to remove.
            # Let's find one such Y.
            def find_subset(weights_dict, target):
                items = list(weights_dict.items())
                # Sort by weight desc?
                items.sort(key=lambda x: x[1], reverse=True)
                
                def backtrack(idx, current_sum, current_set):
                    if current_sum == target:
                        return current_set
                    if idx == len(items) or current_sum > target:
                        return None
                    
                    # Option 1: skip
                    res = backtrack(idx + 1, current_sum, current_set)
                    if res: return res
                    
                    # Option 2: take
                    w = items[idx][1]
                    if current_sum + w <= target:
                        res = backtrack(idx + 1, current_sum + w, current_set | {items[idx][0]})
                        if res: return res
                    return None

                Y = backtrack(0, 0, set())
                return x, Y
            return find_subset(other_weights, target)
    return None

def greedy_remove(S, weights, deadline):
    """
    Start with S, find conflicts, remove elements to resolve.
    """
    S = set(S)
    while time.time() < deadline:
        conflict = find_conflict(S, weights)
        if not conflict:
            return S
        
        x, Y = conflict
        # Candidates to remove: x or any y in Y
        candidates = [x] + list(Y)
        
        # Heuristic: Remove the element that is in the most conflicts?
        # Or just random?
        # Or remove the one with the largest weight (smallest index)?
        # Small indices (large weights) are more likely to be sums of others.
        # So removing x (if x is small) might be good.
        # But removing y (if y is large) might be safer?
        
        # Let's try removing the element with the largest weight (smallest index) from candidates.
        # This assumes small numbers are "more problematic".
        z_to_remove = min(candidates, key=lambda z: z)
        
        S.remove(z_to_remove)
        
    return S

def local_improve(S, weights, deadline):
    """
    Try to add back removed elements or swap.
    """
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any element not in S
        for y in range(1, 25):
            if y in S:
                continue
            
            new_S = S | {y}
            if is_valid_set(new_S, weights):
                S = new_S
                improved = True
                break
            else:
                # Try swapping y with some x in S
                for x in S:
                    swap_S = (S - {x}) | {y}
                    if is_valid_set(swap_S, weights):
                        S = swap_S
                        improved = True
                        break
                if improved:
                    break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    best_S = set()
    
    # Try multiple random starting points or removal orders
    random.seed(42)
    
    for i in range(50): # 50 restarts
        if time.time() > deadline:
            break
        
        S = set(range(1, 25))
        
        # Randomize removal order?
        # The greedy_remove picks min index.
        # Let's add some randomness to the choice of removal.
        
        # Modified greedy_remove to allow randomness
        S_local = set(S)
        while time.time() < deadline:
            conflict = find_conflict(S_local, weights)
            if not conflict:
                break
            
            x, Y = conflict
            candidates = [x] + list(Y)
            
            # Randomly pick one, or pick based on heuristic
            if random.random() < 0.5:
                z_to_remove = min(candidates, key=lambda z: z)
            else:
                z_to_remove = random.choice(candidates)
            
            S_local.remove(z_to_remove)
        
        if len(S_local) > len(best_S):
            best_S = set(S_local)
            
        # Local improvement
        if time.time() < deadline:
            S_improved = local_improve(S_local, weights, deadline)
            if len(S_improved) > len(best_S):
                best_S = set(S_improved)
                
    # Final validation
    if not is_valid_set(best_S, weights):
        # Fallback
        best_S = set(range(13, 25))
        
    return list(best_S)
