import itertools
import math
import random
import time

def construct():
    N = 128
    deadline = time.time() + 50.0  # 50s safety margin
    
    # Precompute LCMs
    # lcm[i][j] for 1 <= i,j <= N
    # Using math.gcd for lcm: lcm(a,b) = a*b // gcd(a,b)
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)

    def is_bad_triple(a, b, c):
        return lcm[a][b] == lcm[a][c] == lcm[b][c]

    def check_conflict(selected, x):
        """
        Check if adding x to selected creates any bad triple.
        selected is a list of ints.
        """
        n = len(selected)
        # Check pairs in selected
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                if is_bad_triple(a, b, x):
                    return True
        return False

    def greedy_construct(order):
        selected = []
        for x in order:
            if not check_conflict(selected, x):
                selected.append(x)
        return selected

    # Initial Strategies
    best_set = []
    best_size = 0
    
    # 1. Standard Greedy (1 to 128)
    s1 = greedy_construct(range(1, N + 1))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
        
    # 2. Reverse Greedy (128 to 1)
    s2 = greedy_construct(range(N, 0, -1))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
        
    # 3. Randomized Greedy
    for _ in range(50):
        if time.time() > deadline:
            break
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        s3 = greedy_construct(random_order)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3

    # Local Search / Hill Climbing
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute list of all numbers for quick access
    all_nums = set(range(1, N + 1))
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any single element
        missing = list(all_nums - current_set)
        random.shuffle(missing)
        
        for v in missing:
            # Check if v can be added
            # We need to check against all pairs in current_set
            # Optimization: If current_set is large, this is O(K^2)
            if not check_conflict(list(current_set), v):
                current_set.add(v)
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
            
        # If no single addition works, try local moves to increase size
        # Strategy: Remove one element, see if we can add more than 1.
        
        # Randomize order of removal candidates
        removal_candidates = list(current_set)
        random.shuffle(removal_candidates)
        
        for u in removal_candidates:
            if time.time() > deadline:
                break
                
            temp_set = current_set - {u}
            
            # Try to greedily add missing elements to temp_set
            # We want to see if we can add > 1 elements
            added_count = 0
            new_elements_to_add = []
            
            # Use a copy of temp_set for checking conflicts during this phase
            # To speed up, we can just check against temp_set and the newly added ones
            working_set = list(temp_set)
            
            # Iterate through missing elements in random order
            missing_for_temp = list(all_nums - temp_set)
            random.shuffle(missing_for_temp)
            
            for v in missing_for_temp:
                if v == u:
                    continue
                if not check_conflict(working_set, v):
                    working_set.append(v)
                    added_count += 1
                    new_elements_to_add.append(v)
                    
            # If we can add more than 1 element, it's a net gain
            if added_count > 1:
                # Apply the move
                current_set = set(working_set)
                current_size = len(current_set)
                improved = True
                break
                
        if not improved:
            # Try 2-swap: Remove 2, add > 2
            # This is more expensive, so do it less frequently or only if stuck
            if random.random() < 0.1: # 10% chance to try 2-swap
                # Pick 2 random elements to remove
                if len(current_set) >= 2:
                    u1, u2 = random.sample(list(current_set), 2)
                    temp_set = current_set - {u1, u2}
                    working_set = list(temp_set)
                    added_count = 0
                    new_elements_to_add = []
                    
                    missing_for_temp = list(all_nums - temp_set)
                    random.shuffle(missing_for_temp)
                    
                    for v in missing_for_temp:
                        if v == u1 or v == u2:
                            continue
                        if not check_conflict(working_set, v):
                            working_set.append(v)
                            added_count += 1
                            new_elements_to_add.append(v)
                            
                    if added_count > 2:
                        current_set = set(working_set)
                        current_size = len(current_set)
                        improved = True

    return list(current_set)
