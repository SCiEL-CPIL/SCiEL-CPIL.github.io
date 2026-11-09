import time
import random
from functools import lru_cache

L = 5354228880  # lcm(1..24)
L_weights = [0] + [L // i for i in range(1, 25)]

@lru_cache(maxsize=500000)
def subset_sum_cached(weight_tuple, target):
    """
    Check if target can be formed by a subset of weights (tuple for caching).
    Uses meet-in-the-middle with early termination.
    """
    if target == 0:
        return True
    if not weight_tuple:
        return False
    
    # Filter weights > target
    weights = [w for w in weight_tuple if w <= target]
    if not weights:
        return False
    if sum(weights) < target:
        return False
        
    n = len(weights)
    if n <= 1:
        return weights[0] == target if weights else False
        
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        left_sums.update(new_sums)
        if target in left_sums:
            return True
        
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
                if ns == target:
                    return True
        right_sums.update(new_sums)
        if target in right_sums:
            return True
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
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
    
    return subset_sum_cached(tuple(sorted(weights)), target)

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
            continue
            
        weights_for_y = [L_weights[z] for z in S_set if z != y and z != x]
        if subset_sum_exists(weights_for_y, target_diff):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
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
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13],
        # Try to specifically find {5..24}
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search with 1-for-1 swaps and additions
    def local_search(S, max_iterations=200):
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
            
            # Try 1-for-1 swap: remove y, add c (if c not in S)
            if not improved and len(S) >= 1:
                for y in S:
                    if time.time() > deadline:
                        return S
                    S_without_y = [x for x in S if x != y]
                    S_without_y_set = set(S_without_y)
                    candidates = [k for k in range(1, 25) if k not in S_without_y_set and k != y]
                    random.shuffle(candidates)
                    for c in candidates:
                        if time.time() > deadline:
                            return S
                        if can_add(S_without_y, c):
                            S = S_without_y + [c]
                            S_set = S_without_y_set | {c}
                            improved = True
                            break
                    if improved:
                        break
                        
            # Try 2-for-1 swap: remove one, add two (only if 1-for-1 didn't help)
            if not improved and len(S) >= 2 and time.time() < deadline:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    y = S[i]
                    S_without_y = [x for x in S if x != y]
                    S_without_y_set = set(S_without_y)
                    candidates = [k for k in range(1, 25) if k not in S_without_y_set and k != y]
                    random.shuffle(candidates)
                    # Limit candidates to first 10 for speed
                    candidates = candidates[:10]
                    
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
            improved_set = local_search(S, max_iterations=80)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
        
    return best_set
