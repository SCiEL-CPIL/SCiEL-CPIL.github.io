import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0  # Safety margin before 90s kill
    n = 512
    max_size = 48
    # Max sum is 48 * 512 = 24576
    max_sum = max_size * n + 100  # 24676
    
    best_set = []
    best_size = 0
    
    # Pre-allocate DP arrays once to avoid repeated allocation
    # dp[k][s] = True if there exists a subset of size k with sum s
    dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
    
    def reset_dp():
        for k in range(max_size + 1):
            dp[k][:] = False
        dp[0][0] = True
    
    def is_valid_addition(x, S, current_max_k):
        # Check 1: x is not the mean of any subset of S (size >= 2)
        # mean(T) = x => sum(T) = k*x for |T| = k >= 2
        for k in range(2, current_max_k + 1):
            target = k * x
            if target > max_sum:
                break
            if dp[k][target]:
                return False
        
        # Check 2: No existing element y in S is the mean of any subset of S U {x}
        # where the subset includes x.
        # mean(T U {x}) = y => sum(T) + x = (k+1)*y => sum(T) = (k+1)*y - x
        # where T is a subset of S with |T| = k >= 1
        for y in S:
            for k in range(1, current_max_k + 1):
                target = (k + 1) * y - x
                if target < 0 or target > max_sum:
                    continue
                if dp[k][target]:
                    return False
        return True
    
    def update_dp(x, current_max_k):
        # Update DP in place, iterating k from high to low
        for k in range(current_max_k, -1, -1):
            if x <= max_sum:
                dp[k+1][x:] |= dp[k][:-x]
    
    def verify_set(S):
        """Full verification of a set. Returns True if valid."""
        if not S:
            return True
        if len(S) > max_size:
            return False
        if len(set(S)) != len(S):
            return False
        if any(x < 1 or x > n for x in S):
            return False
        
        # Build DP from scratch
        dp_check = [np.zeros(max_sum + 1, dtype=bool) for _ in range(len(S) + 1)]
        dp_check[0][0] = True
        S_sorted = sorted(S)
        
        for i, x in enumerate(S_sorted):
            for k in range(i, -1, -1):
                if x <= max_sum:
                    dp_check[k+1][x:] |= dp_check[k][:-x]
        
        # Check each element
        for idx, x in enumerate(S_sorted):
            # x must not be mean of any subset of OTHER elements (size >= 2)
            # Build DP without x
            dp_without = [np.zeros(max_sum + 1, dtype=bool) for _ in range(len(S) + 1)]
            dp_without[0][0] = True
            count = 0
            for j, y in enumerate(S_sorted):
                if j == idx:
                    continue
                for k in range(count, -1, -1):
                    if y <= max_sum:
                        dp_without[k+1][y:] |= dp_without[k][:-y]
                count += 1
            
            for k in range(2, count + 1):
                target = k * x
                if target > max_sum:
                    break
                if dp_without[k][target]:
                    return False
        return True
    
    # Strategy: Multiple restarts with greedy growth
    # Use different random orderings and occasionally bias towards certain ranges
    
    for restart in range(2000):
        if time.time() - start_time > max_time:
            break
        
        reset_dp()
        
        # Different strategies for candidate ordering
        strategy = restart % 5
        if strategy == 0:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        elif strategy == 1:
            # Bias towards larger numbers
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -x)
            # Partially shuffle
            for i in range(len(candidates) - 1, 0, -1):
                if random.random() < 0.3:
                    j = random.randint(0, i)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
        elif strategy == 2:
            # Bias towards smaller numbers
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            candidates.sort()
            for i in range(len(candidates)):
                if random.random() < 0.3:
                    j = random.randint(i, len(candidates) - 1)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
        elif strategy == 3:
            # Random with occasional large gaps
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        else:
            # Start from a known good region
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        
        S = []
        current_max_k = 0
        
        for x in candidates:
            if time.time() - start_time > max_time:
                break
            
            if is_valid_addition(x, S, current_max_k):
                update_dp(x, current_max_k)
                S.append(x)
                current_max_k += 1
                
                if current_max_k == max_size:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # Verify the best set to ensure correctness
            if best_size >= 40:
                if verify_set(best_set):
                    pass  # Valid
                else:
                    # Should not happen, but reset if invalid
                    best_size = 0
                    best_set = []
        
        # If we found a large set, try local improvement
        if best_size >= 40 and restart % 10 == 0:
            if time.time() - start_time > max_time:
                break
            
            # Try swapping elements to improve
            current_set = best_set[:]
            improved = True
            while improved and time.time() - start_time < max_time:
                improved = False
                current_set_copy = current_set[:]
                reset_dp()
                
                # Rebuild DP for current_set
                current_set_sorted = sorted(current_set_copy)
                for i, x in enumerate(current_set_sorted):
                    update_dp(x, i)
                
                # Try removing one element and adding another
                for idx in range(len(current_set_copy)):
                    if time.time() - start_time > max_time:
                        break
                    
                    x_remove = current_set_copy[idx]
                    
                    # Build DP without x_remove
                    dp_no_rm = [np.zeros(max_sum + 1, dtype=bool) for _ in range(len(current_set_copy) + 1)]
                    dp_no_rm[0][0] = True
                    cnt = 0
                    for j, y in enumerate(current_set_copy):
                        if j == idx:
                            continue
                        for k in range(cnt, -1, -1):
                            if y <= max_sum:
                                dp_no_rm[k+1][y:] |= dp_no_rm[k][:-y]
                        cnt += 1
                    
                    # Try adding a new element
                    for x_add in range(1, n + 1):
                        if x_add in current_set_copy:
                            continue
                        
                        # Check if x_add is valid with current_set without x_remove
                        valid = True
                        for k in range(2, cnt + 1):
                            target = k * x_add
                            if target > max_sum:
                                break
                            if dp_no_rm[k][target]:
                                valid = False
                                break
                        
                        if valid:
                            # Check no existing y is mean of subset including x_add
                            for y in current_set_copy:
                                if y == x_remove:
                                    continue
                                for k in range(1, cnt + 1):
                                    target = (k + 1) * y - x_add
                                    if target < 0 or target > max_sum:
                                        continue
                                    if dp_no_rm[k][target]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                        
                        if valid:
                            # Perform the swap
                            new_set = current_set_copy[:]
                            new_set.remove(x_remove)
                            new_set.append(x_add)
                            if len(new_set) == len(current_set_copy):
                                # Verify
                                if verify_set(new_set):
                                    current_set = new_set
                                    improved = True
                                    break
                    
                    if improved:
                        break
                
                if current_set and len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
    
    # Final verification of best set
    if best_set:
        if not verify_set(best_set):
            # Fallback: try a simple valid set
            # Powers of 2 are safe: no element is mean of others
            # 1, 2, 4, 8, 16, 32, 64, 128, 256 -> 9 elements
            # We can do better. Let's use a known construction.
            # Actually, let's just return an empty set or a small valid one
            best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            best_size = len(best_set)
    
    return best_set
