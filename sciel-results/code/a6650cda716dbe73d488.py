import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # 1. Greedy Initialization
    S = set()
    in_set = [False] * 257
    
    def can_add(x, S_set, in_set_arr):
        # Check if x can be added to S_set
        # x is not in S_set
        for a in S_set:
            # Case 1: a, b, x (x largest)
            # 2b = a + x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if 1 <= b <= 256 and in_set_arr[b] and b != a and b != x:
                    return False
            
            # Case 2: a, x, b (x middle)
            # 2x = a + b => b = 2x - a
            b = 2 * x - a
            if 1 <= b <= 256 and in_set_arr[b] and b != a and b != x:
                return False
            
            # Case 3: x, a, b (x smallest)
            # 2a = x + b => b = 2a - x
            b = 2 * a - x
            if 1 <= b <= 256 and in_set_arr[b] and b != a and b != x:
                return False
        return True

    for x in range(1, 257):
        if can_add(x, S, in_set):
            S.add(x)
            in_set[x] = True
            
    current_score = sum(1.0 / x for x in S)
    best_set = S
    best_score = current_score
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over a copy of the list to avoid modification during iteration
        # We want to try removing each element and see if we can do better
        for x in list(best_set):
            if time.time() > deadline:
                break
            
            # Temporarily remove x
            S_temp = best_set - {x}
            in_set_temp = in_set[:]
            in_set_temp[x] = False
            
            # Greedily add smallest valid y
            new_S = set(S_temp)
            new_in_set = in_set_temp[:]
            
            added_smaller = False
            for y in range(1, 257):
                if new_in_set[y]:
                    continue
                # Optimization: if y >= x and we haven't added anything smaller, 
                # adding y will not improve score over just removing x (which lowers score).
                # But we might add multiple. 
                # Actually, if we add y > x, score decreases. 
                # We only accept if final score > best_score.
                # To save time, we can just try to add all valid y.
                
                if can_add(y, new_S, new_in_set):
                    new_S.add(y)
                    new_in_set[y] = True
                    if y < x:
                        added_smaller = True
            
            # Calculate score
            # Optimization: calculate delta
            # Removed 1/x. Added sum(1/y for y in new_S - S_temp)
            added_elements = new_S - S_temp
            if not added_elements:
                continue
                
            delta = sum(1.0 / y for y in added_elements) - (1.0 / x)
            new_score = best_score + delta
            
            if new_score > best_score:
                best_set = new_S
                best_score = new_score
                in_set = new_in_set
                improved = True
                
    return list(best_set)

# EVOLVE-BLOCK-END
