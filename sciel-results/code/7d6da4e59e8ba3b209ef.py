import random
import time
import sys

L = 5354228880  # lcm(1..24)
L_weights = [0] + [L // i for i in range(1, 25)]

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle with early termination.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
    if sum(weights) < target:
        return False
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        if target in left_sums:
            return True
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        left_sums.update(new_sums)
        
    right_sums = {0}
    for w in right:
        if target in right_sums:
            return True
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def is_valid(S):
    """
    Check if S is a valid set.
    S is a list of ints.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = L_weights[x]
        weights = [L_weights[y] for y in S_set if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def can_add(S, x):
    """
    Check if x can be added to valid set S.
    Returns True if S U {x} is valid.
    """
    if x in S:
        return False
    
    S_set = set(S)
    
    # 1. Check if x is covered by S
    target_x = L_weights[x]
    weights_for_x = [L_weights[y] for y in S_set]
    if subset_sum_exists(weights_for_x, target_x):
        return False
        
    # 2. Check if adding x covers any y in S
    # y is covered if L_y = L_x + sum(subset of S \ {x, y})
    # i.e., L_y - L_x is a subset sum of {L_z | z in S, z != x, z != y}
    for y in S:
        target_diff = L_weights[y] - L_weights[x]
        if target_diff < 0:
            continue
        if target_diff == 0:
            # L_y == L_x, implies x=y which is false, or subset is empty?
            # If L_y == L_x, then y=x, which is not possible as x not in S.
            # Wait, L_weights are distinct for distinct integers 1..24.
            # So target_diff == 0 implies L_y == L_x, impossible.
            continue
            
        weights_for_y = [L_weights[z] for z in S_set if z != y and z != x]
        if subset_sum_exists(weights_for_y, target_diff):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_set = None
    best_size = 0
    
    # Initial guess: Seed solution
    seed_set = list(range(13, 25))
    if is_valid(seed_set):
        best_set = seed_set
        best_size = len(seed_set)
    
    # Deterministic greedy construction
    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    # Try several deterministic orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search
    def local_search(S, max_iterations=100):
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each missing element
            missing = [i for i in range(1, 25) if i not in S_set]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    return S
                if can_add(S, m):
                    S.append(m)
                    S_set.add(m)
                    improved = True
            
            # Try 2-for-1 swap: remove one, add two
            if not improved and len(S) >= 1:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    y = S[i]
                    S_without_y = [x for x in S if x != y]
                    S_without_y_set = set(S_without_y)
                    candidates = [k for k in range(1, 25) if k not in S_without_y_set and k != y]
                    random.shuffle(candidates)
                    
                    # Try to find two candidates c1, c2 such that S_without_y + [c1, c2] is valid
                    # This is expensive. We can just try can_add twice.
                    # First add c1, then add c2.
                    for c1 in candidates:
                        if time.time() > deadline:
                            return S
                        if can_add(S_without_y, c1):
                            S_temp = S_without_y + [c1]
                            S_temp_set = S_without_y_set | {c1}
                            for c2 in candidates:
                                if c2 == c1:
                                    continue
                                if time.time() > deadline:
                                    return S
                                if can_add(S_temp, c2):
                                    S = S_temp + [c2]
                                    S_set = S_temp_set | {c2}
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
        return S

    # Apply local search to best set found so far
    if best_set is not None and time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Random restarts
    random.seed(42)
    attempts = 0
    while time.time() < deadline and best_size < 24:
        attempts += 1
        
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = greedy_construct(random_order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        
        if time.time() < deadline:
            improved_set = local_search(S, max_iterations=50)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
        
    return best_set
