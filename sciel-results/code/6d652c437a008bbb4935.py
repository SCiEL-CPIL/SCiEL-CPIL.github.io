import time
import random
import numpy as np
from itertools import combinations

_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def _find_collision(indices):
    """
    Returns (is_dissociated, conflict_indices).
    conflict_indices is a set of indices involved in a collision, or None.
    """
    k = len(indices)
    if k == 0:
        return True, None
    
    # Generate sums in mask-index order
    sums = np.array([0], dtype=np.int64)
    for v in [_WEIGHTS[i-1] for i in indices]:
        sums = np.concatenate((sums, sums + v))
        
    # Sort and find duplicates
    order = np.argsort(sums)
    sorted_sums = sums[order]
    
    diffs = np.diff(sorted_sums)
    dup_indices = np.where(diffs == 0)[0]
    
    if len(dup_indices) == 0:
        return True, None
        
    # Take the first duplicate
    idx = dup_indices[0]
    m1 = order[idx]
    m2 = order[idx+1]
    
    # Symmetric difference of masks gives the conflicting elements
    diff_mask = m1 ^ m2
    
    n = len(indices)
    conflict_local_indices = []
    for i in range(n):
        if (diff_mask >> i) & 1:
            conflict_local_indices.append(indices[i])
            
    return False, set(conflict_local_indices)

def _get_all_subset_sums(indices):
    """Return all 2^k subset sums as a numpy array."""
    sums = np.array([0], dtype=np.int64)
    for v in [_WEIGHTS[i-1] for i in indices]:
        sums = np.concatenate((sums, sums + v))
    return sums

def _min_distance_to_sums(candidate_elem, existing_indices):
    """
    Compute the minimum distance from _WEIGHTS[candidate_elem-1] 
    to the convex hull of existing subset sums (approximated by min distance
    to existing sums). Returns a float score.
    """
    w = _WEIGHTS[candidate_elem - 1]
    sums = _get_all_subset_sums(existing_indices)
    
    # Distance from w to each existing sum
    distances = np.abs(sums - w)
    
    # Also consider distance from 0 to sums that include w (i.e., w + existing sums)
    # This is equivalent to checking if w collides with any subset sum
    # A collision occurs if w = s1 - s2 for two subset sums s1, s2
    # Which means w + s2 = s1
    # So we check if w + any existing sum equals any existing sum
    # i.e., if (w + sums) intersects with sums
    
    shifted = w + sums
    # Find minimum difference between shifted and sums
    # This is expensive, so use a simpler proxy: min |w - s| for s in sums
    # Plus check if w itself is in sums (collision with empty set)
    
    min_dist = np.min(distances)
    
    # Bonus: prefer elements whose weight is far from all existing sums
    # Use log-sum-exp style soft minimum for smoothness
    # But for speed, just use the raw minimum distance
    return float(min_dist)

def _greedy_geometric_construct(all_elements, deadline, seed=None):
    """
    Greedy construction: start empty, add elements one at a time
    choosing the one that maximizes minimum distance to existing subset sums.
    """
    if seed is not None:
        random.seed(seed)
    
    S = []
    remaining = list(all_elements)
    random.shuffle(remaining)
    
    while remaining and time.time() < deadline:
        best_elem = None
        best_score = -1
        
        # For each candidate, compute geometric score
        for elem in remaining:
            if not S:
                # First element: any is fine, prefer larger weights for separation
                score = float(_WEIGHTS[elem - 1])
            else:
                score = _min_distance_to_sums(elem, S)
            
            if score > best_score:
                best_score = score
                best_elem = elem
        
        if best_elem is None:
            break
            
        # Check if adding best_elem maintains dissociation
        test_S = S + [best_elem]
        is_diss, _ = _find_collision(test_S)
        
        if is_diss:
            S.append(best_elem)
            remaining.remove(best_elem)
        else:
            # Skip this element, try next best
            remaining.remove(best_elem)
    
    return S

def _try_add_back(S, removed_elements, deadline):
    """
    Try to add back removed elements one at a time if possible.
    Returns the improved set.
    """
    S = set(S)
    improved = False
    
    # Try adding back each removed element in random order
    candidates = list(removed_elements)
    random.shuffle(candidates)
    
    for elem in candidates:
        if time.time() > deadline:
            break
        if elem in S:
            continue
            
        test_S = S | {elem}
        is_diss, _ = _find_collision(list(test_S))
        if is_diss:
            S = test_S
            improved = True
            
    return S, improved

def _conflict_guided_removal(S, conflict, deadline):
    """
    Smart removal: analyze the conflict structure and remove the element
    that appears in the most potential future conflicts.
    """
    if not conflict:
        return None
    
    # Count how many times each element appears in the conflict
    # Also consider: removing an element that is "central" to the collision
    # Strategy: remove the element with the smallest weight among those
    # in the conflict, as it's most likely to cause further collisions
    
    # Alternative: remove the element that, when removed, leaves the
    # remaining set most "spread out"
    
    # For now, use a weighted score: prefer removing elements with
    # smaller weights (they cause more collisions due to finer granularity)
    scores = {}
    for elem in conflict:
        # Smaller weight -> more likely to cause collisions -> remove it
        scores[elem] = -float(_WEIGHTS[elem - 1])
    
    # Add some randomness
    for elem in conflict:
        scores[elem] += random.uniform(-0.1, 0.1) * max(_WEIGHTS)
    
    return max(conflict, key=lambda e: scores[e])

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    best = []
    best_len = 0
    
    # Strategy 1: Geometric greedy construction
    for trial in range(10):
        if time.time() > deadline:
            break
        S = _greedy_geometric_construct(all_elements, deadline, seed=trial * 7 + 1)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 2: Original removal-based approach with multiple strategies
    strategies = [0, 1, 2, 3, 4]
    
    for strat in strategies:
        for trial in range(10):  # Reduced trials to allow more time for other strategies
            if time.time() > deadline:
                break
                
            S = set(all_elements)
            removed = []
            
            while time.time() < deadline:
                is_diss, conflict = _find_collision(list(S))
                
                if is_diss:
                    # Try to add back removed elements
                    S, improved = _try_add_back(S, removed, deadline)
                    if improved:
                        # Re-check and continue trying to add
                        is_diss, conflict = _find_collision(list(S))
                        if not is_diss and conflict:
                            # Continue removal loop
                            pass
                        else:
                            if len(S) > best_len:
                                best = list(S)
                                best_len = len(S)
                            break
                    else:
                        if len(S) > best_len:
                            best = list(S)
                            best_len = len(S)
                        break # Found a valid set for this trial
                    
                if conflict is None:
                    break # Error?
                    
                # Choose element to remove
                if strat == 0:
                    # Min weight
                    x = min(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 1:
                    # Max weight
                    x = max(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 2:
                    # Random
                    x = random.choice(list(conflict))
                elif strat == 3:
                    # Max index
                    x = max(conflict)
                elif strat == 4:
                    # Min index
                    x = min(conflict)
                else:
                    # Conflict-guided removal
                    x = _conflict_guided_removal(S, conflict, deadline)
                    if x is None:
                        x = random.choice(list(conflict))
                    
                S.remove(x)
                removed.append(x)
                
            # If we exited due to deadline, check if current S is valid
            if time.time() >= deadline:
                if len(S) > 0:
                    is_diss, _ = _find_collision(list(S))
                    if is_diss:
                        # Try to add back
                        S, _ = _try_add_back(S, removed, deadline)
                        is_diss, _ = _find_collision(list(S))
                        if is_diss and len(S) > best_len:
                            best = list(S)
                            best_len = len(S)
    
    # Strategy 3: Local search / hill climbing on the best solution
    if best and time.time() < deadline:
        # Try swapping elements in/out of the best set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_set = set(best)
            for add_elem in all_elements:
                if add_elem in best_set:
                    continue
                for remove_elem in list(best_set):
                    test_set = (best_set - {remove_elem}) | {add_elem}
                    is_diss, _ = _find_collision(list(test_set))
                    if is_diss:
                        best = list(test_set)
                        best_set = test_set
                        improved = True
                        break
                if improved:
                    break
    
    # Final: try to add any remaining elements to best
    if best and time.time() < deadline:
        best_set = set(best)
        for elem in all_elements:
            if elem in best_set:
                continue
            test_set = best_set | {elem}
            is_diss, _ = _find_collision(list(test_set))
            if is_diss:
                best_set = test_set
                best = list(best_set)
    
    return best
