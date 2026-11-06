import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for safety
    
    # Precompute bit masks for 1..512 to speed up DP shifts
    # dp[k] |= dp[k-1] << x is equivalent to dp[k] |= dp[k-1] * (1 << x)
    # But bit shifting is faster. 
    # We need to mask to limit size. Max sum is 48 * 512 = 24576.
    # 1 << 24576 is a huge integer. 
    # Python integers are arbitrary precision, but operations get slower as they grow.
    # We can mask with (1 << 25000) - 1.
    
    MAX_SUM = 25000
    MASK = (1 << MAX_SUM) - 1
    
    def is_valid_set(S_list):
        n = len(S_list)
        if n == 0:
            return True
        
        # For each element a, check if it is the mean of any subset of others
        for i, a in enumerate(S_list):
            # T is S_list without a
            # We need to check if k*a is a subset sum of T of size k
            # for k >= 2
            
            # Optimization: If a is very small, k*a is small.
            # If a is very large, k*a is large.
            
            # DP for subset sums of T
            # dp[k] is bitset of sums using k elements from T
            # We iterate through elements of T
            # To avoid allocating a new list for each a, we can reuse, 
            # but the DP depends on T which changes for each a.
            # So we must recompute DP for each a.
            
            # However, we can optimize the inner loop.
            # T has n-1 elements.
            # dp[0] = 1
            # dp[1..n-1] = 0
            
            # We only care about sums up to (n-1)*a? No, up to MAX_SUM.
            # Actually, we only care if k*a is in dp[k].
            # k*a <= MAX_SUM.
            
            # Let's use a list of integers for dp.
            # Size n.
            dp = [0] * (n)
            dp[0] = 1
            
            # Iterate through T
            # We can skip elements that are too large? No, they contribute to sums.
            # But if x > MAX_SUM, it can't be part of a valid sum for k*a <= MAX_SUM?
            # Wait, if x > MAX_SUM, then any sum involving x is > MAX_SUM.
            # So we can ignore elements > MAX_SUM?
            # Max element is 512. MAX_SUM is 25000. So all elements are < MAX_SUM.
            
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                # Update dp in reverse
                # dp[k] |= (dp[k-1] << x)
                # We only need k up to n-1
                # And we only care about bits up to MAX_SUM
                for k in range(n - 1, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x) & MASK
                        # Early exit if we found a violation?
                        # We check after building full DP for this a.
                        # Or we can check on the fly?
                        # If (dp[k] & (1 << (k*a))) is set, return False.
                        # But dp[k] is being built.
                        # We can check after each element update?
                        # It might be faster to check after the full DP for a is built.
            
            # Check for violations
            for k in range(2, n):
                target = k * a
                if target > MAX_SUM:
                    break
                if dp[k] & (1 << target):
                    return False
        return True

    def try_add(S, x):
        if x in S:
            return False
        new_S = list(S) + [x]
        return is_valid_set(new_S)

    def local_search(S, deadline):
        S = set(S)
        S_list = list(S)
        all_vals = list(range(1, 513))
        
        # Sort all_vals randomly once per restart? Or shuffle each time?
        # Shuffling each time is better for diversity.
        
        stuck_count = 0
        while time.time() < deadline:
            added = False
            
            # Try to add a random element
            # We can prioritize elements that are "far" from existing means?
            # For now, random.
            candidates = [x for x in all_vals if x not in S]
            if not candidates:
                break
                
            random.shuffle(candidates)
            # Try a few candidates
            for x in candidates[:10]:
                if try_add(S_list, x):
                    S.add(x)
                    S_list.append(x)
                    added = True
                    stuck_count = 0
                    break
            
            if added:
                continue
            
            # If no addition, try swaps
            stuck_count += 1
            if stuck_count > 5:
                # Try a more aggressive swap strategy
                # Remove a random element, try to add a random one
                if S_list:
                    idx = random.randint(0, len(S_list) - 1)
                    remove_x = S_list[idx]
                    S.remove(remove_x)
                    S_list.pop(idx)
                    
                    # Now try to add something
                    candidates = [x for x in all_vals if x not in S]
                    random.shuffle(candidates)
                    swap_success = False
                    for x in candidates[:20]:
                        if try_add(S_list, x):
                            S.add(x)
                            S_list.append(x)
                            swap_success = True
                            break
                            
                    if not swap_success:
                        # Revert
                        S.add(remove_x)
                        S_list.append(remove_x)
                        # Maybe sort to keep it somewhat ordered?
                        # S_list.sort()
                        
            # Small chance to do a "kick" - remove 2, add 2?
            # Too complex for now.

        return S

    best_size = 0
    best_set = []
    
    # Multiple restarts
    while time.time() < deadline:
        # Start with a small random valid set
        S = set()
        S_list = []
        
        # Greedy initial construction
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(S_list, x):
                S.add(x)
                S_list.append(x)
                if len(S) == 48:
                    break
        
        # Local search to improve
        S = local_search(S, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            if best_size == 48:
                break

    return best_set
