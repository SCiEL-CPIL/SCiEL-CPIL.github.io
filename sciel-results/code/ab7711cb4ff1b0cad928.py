import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    def is_valid_set(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def check_add_x(S, x):
        """Check if adding x to S keeps it valid. x must not be in S."""
        S_list = sorted(S)
        
        # Case 1: x is 'a' (x < b <= c, x | (b+c))
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for k in range(i, len(S_list)):
                c = S_list[k]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (a < x <= c, a | (x+c))
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (a < b <= x, a | (b+x))
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True

    # 1. Find the best interval [L, 256]
    best_L = 171
    best_size = 0
    best_set = set()
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S = set(range(L, n + 1))
        
        # Quick check: for a=L, is there a multiple of L in [2(L+1), 2N]?
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            continue
        
        # If L check passes, do full check for promising L values
        if L >= 100:
            if is_valid_set(S):
                size = len(S)
                if size > best_size:
                    best_size = size
                    best_set = S
                    best_L = L

    # Current best is likely [171, 256] size 86.
    # Let's verify best_set is not empty.
    if not best_set:
        # Fallback to seed
        return list(range(192, 257))
    
    # 2. Greedy Expansion
    # Try to add elements from best_L - 1 down to 1
    current_set = set(best_set)
    
    candidates = list(range(best_L - 1, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        if check_add_x(current_set, x):
            current_set.add(x)

    best_found_set = set(current_set)
    best_found_size = len(current_set)
    
    # 3. Multi-start greedy with different orderings
    for restart in range(20):
        if time.time() > deadline:
            break
            
        cand = list(range(best_L - 1, 0, -1))
        random.shuffle(cand)
        
        test_set = set(best_set)
        
        for x in cand:
            if time.time() > deadline:
                break
            if check_add_x(test_set, x):
                test_set.add(x)
        
        if len(test_set) > best_found_size:
            best_found_size = len(test_set)
            best_found_set = set(test_set)
    
    # 4. Swap-based local search: try removing one element and adding another
    for iteration in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        S_list = sorted(best_found_set)
        
        for i, remove_x in enumerate(S_list):
            if time.time() > deadline:
                break
            
            test_set = set(best_found_set)
            test_set.discard(remove_x)
            
            for add_y in range(1, n + 1):
                if time.time() > deadline:
                    break
                if add_y in test_set:
                    continue
                if check_add_x(test_set, add_y):
                    new_set = set(test_set)
                    new_set.add(add_y)
                    if len(new_set) > best_found_size:
                        best_found_set = new_set
                        best_found_size = len(new_set)
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    # 5. Final polish: try adding any remaining element in multiple orders
    remaining = [x for x in range(1, n + 1) if x not in best_found_set]
    for order_seed in range(5):
        if time.time() > deadline:
            break
        
        random.seed(order_seed)
        test_set = set(best_found_set)
        remaining_shuffled = remaining[:]
        random.shuffle(remaining_shuffled)
        
        for x in remaining_shuffled:
            if time.time() > deadline:
                break
            if check_add_x(test_set, x):
                test_set.add(x)
        
        if len(test_set) > best_found_size:
            best_found_size = len(test_set)
            best_found_set = test_set
    
    return list(best_found_set)

# EVOLVE-BLOCK-END
