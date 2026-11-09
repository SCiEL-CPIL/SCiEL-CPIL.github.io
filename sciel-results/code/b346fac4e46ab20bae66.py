import random
import time
import sys
from itertools import combinations

L = 5354228880  # lcm(1..24)
L_weights = {x: L // x for x in range(1, 25)}

def subset_sum_with_solution(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns (True, solution_indices) if found, (False, None) otherwise.
    weights and indices must be the same length.
    """
    n = len(weights)
    if n == 0:
        return (target == 0, list(range(n)) if target == 0 else None)
    if target == 0:
        return (True, [])
    
    # Filter weights > target (keep track of original indices)
    filtered = [(w, i) for i, w in enumerate(weights) if w <= target]
    if not filtered:
        return (False, None)
        
    n_f = len(filtered)
    mid = n_f // 2
    left = filtered[:mid]
    right = filtered[mid:]
    
    # Build left sums with their index subsets
    left_sums = {0: []}
    for w, idx in left:
        new_entries = {}
        for s, idxs in left_sums.items():
            if s + w <= target:
                new_entries[s + w] = idxs + [idx]
        for s, idxs in new_entries.items():
            if s not in left_sums:
                left_sums[s] = idxs
            elif len(idxs) < len(left_sums[s]):
                left_sums[s] = idxs
        
    # Build right sums with their index subsets
    right_sums = {0: []}
    for w, idx in right:
        new_entries = {}
        for s, idxs in right_sums.items():
            if s + w <= target:
                new_entries[s + w] = idxs + [idx]
        for s, idxs in new_entries.items():
            if s not in right_sums:
                right_sums[s] = idxs
            elif len(idxs) < len(right_sums[s]):
                right_sums[s] = idxs
        
    # Check intersection
    for s, left_idxs in left_sums.items():
        rem = target - s
        if rem in right_sums:
            solution = left_idxs + right_sums[rem]
            return (True, solution)
    return (False, None)

def get_violation(S):
    """
    Returns (x, T) if x's reciprocal is sum of reciprocals of T, else None.
    S is a set of ints.
    """
    S_list = list(S)
    for x in S_list:
        target = L_weights[x]
        weights = []
        T = []
        for y in S_list:
            if y != x:
                w = L_weights[y]
                weights.append(w)
                T.append(y)
        
        found, solution = subset_sum_with_solution(weights, target)
        if found:
            T_subset = [T[i] for i in solution]
            return x, T_subset
                
    return None

def is_valid(S):
    return get_violation(S) is None

def count_violations(S):
    """Count how many elements in S are involved in violations."""
    S_list = list(S)
    violated_elements = set()
    for x in S_list:
        target = L_weights[x]
        weights = []
        for y in S_list:
            if y != x:
                weights.append(L_weights[y])
        found, _ = subset_sum_with_solution(weights, target)
        if found:
            violated_elements.add(x)
    return len(violated_elements)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    best_set = None
    best_size = 0
    
    all_elements = set(range(1, 25))
    
    # Try multiple random restarts with different strategies
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Different initial strategies
        if restart % 3 == 0:
            # Start with full set
            S = set(all_elements)
        elif restart % 3 == 1:
            # Start with random subset of size ~12
            S = set(random.sample(range(1, 25), 12))
        else:
            # Start with all odds or evens
            if restart % 2 == 0:
                S = set(range(1, 25, 2))
            else:
                S = set(range(2, 25, 2))
        
        # Repair phase with smarter candidate selection
        max_repair_steps = 50
        for step in range(max_repair_steps):
            if time.time() > deadline:
                break
            violation = get_violation(S)
            if violation is None:
                break
            
            x, T = violation
            candidates = {x} | set(T)
            
            # Score each candidate by how many violations it would resolve
            best_candidate = None
            best_score = float('inf')
            
            for c in candidates:
                S_test = S.copy()
                S_test.remove(c)
                if is_valid(S_test):
                    best_candidate = c
                    best_score = -1
                    break
                # Count violations as a proxy
                v_count = count_violations(S_test)
                if v_count < best_score:
                    best_score = v_count
                    best_candidate = c
            
            if best_candidate is None:
                best_candidate = random.choice(list(candidates))
            
            S.remove(best_candidate)
            
            if not S:
                break
        
        # Expand phase
        if S:
            missing = list(all_elements - S)
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    break
                S_test = S.copy()
                S_test.add(m)
                if is_valid(S_test):
                    S = S_test
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
                
        # If we found a good solution, do a local search to improve
        if best_size >= 15 and time.time() < deadline - 5:
            # Try swapping elements to increase size
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                current = list(best_set)
                for i in range(len(current)):
                    if time.time() > deadline - 1:
                        break
                    for m in missing:
                        if m in best_set:
                            continue
                        S_test = set(best_set)
                        S_test.remove(current[i])
                        S_test.add(m)
                        if is_valid(S_test) and len(S_test) > len(best_set):
                            best_set = S_test
                            best_size = len(best_set)
                            improved = True
                            break
                        elif is_valid(S_test) and len(S_test) == len(best_set):
                            # Try to keep going with this swap to see if we can add more
                            pass

    if best_set is None:
        return [1]
        
    return list(best_set)
