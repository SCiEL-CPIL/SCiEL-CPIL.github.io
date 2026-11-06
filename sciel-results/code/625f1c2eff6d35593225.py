import time
import random

def _can_add_with_dp(dp, current_set, cand):
    n = len(current_set)
    if n == 0:
        return True
    
    for k in range(2, n + 1):
        target = k * cand
        if (dp[k] >> target) & 1:
            return False
            
    for a in current_set:
        for k in range(1, n + 1):
            target = a * k + a - cand
            if 0 <= target and (dp[k] >> target) & 1:
                return False
    return True

def _rebuild_dp(current_set):
    max_dp_size = 50
    dp = [0] * (max_dp_size + 1)
    dp[0] = 1
    for elem in current_set:
        k_prev = len(current_set) - 1
        for k in range(k_prev + 1, 0, -1):
            dp[k] |= (dp[k - 1] << elem)
    return dp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_len = 0
    
    max_dp_size = 50
    max_val = 512
    all_nums = list(range(1, max_val + 1))
    
    # Pre-compute conflict scores (number of 3-APs)
    # x is mean of a,b if a+b=2x.
    # Count pairs (a,b) with a<b, a!=x, b!=x, a,b in 1..512.
    conflicts = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        for a in range(1, max_val + 1):
            if a == x: continue
            b = 2 * x - a
            if b > x and b <= max_val and b != a:
                conflicts[x] += 1
            # Note: a < x < b or b < x < a.
            # If a < x, b > x.
            # If a > x, b < x.
            # We count each pair once.
            # If a < x, b = 2x - a > x.
            # If a > x, b = 2x - a < x.
            # So we can just iterate a from 1 to x-1.
            # b = 2x - a.
            # if b <= max_val: conflicts[x] += 1
    
    # Correct conflict calculation
    conflicts = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        for a in range(1, x):
            b = 2 * x - a
            if b <= max_val:
                conflicts[x] += 1
            else:
                break # b increases as a decreases? No.
                # a=1 -> b=2x-1.
                # a=x-1 -> b=x+1.
                # So b decreases as a increases.
                # If b > max_val for small a, it will be even larger?
                # No, 2x-1 is the largest b.
                # So if 2x-1 > max_val, then all b > max_val?
                # No, if a is close to x, b is close to x.
                # So we should iterate a from x-1 down to 1?
                # Or just check all. 512*512 is small.
                pass

    # Let's just do the O(N^2) pre-calc properly.
    conflicts = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        for a in range(1, max_val + 1):
            if a == x: continue
            b = 2 * x - a
            if b > a and b <= max_val and b != x:
                conflicts[x] += 1

    # Orders
    # 1. Low conflict first
    order_low_conflict = sorted(all_nums, key=lambda x: conflicts[x])
    # 2. High conflict first (maybe? to remove "bad" ones early? No, we want to KEEP them if possible? 
    #    Actually, high conflict numbers are hard to place. If we place them late, they might be blocked.
    #    If we place them early, they block others.
    #    Usually, "easy" constraints first is better. So low conflict first.)
    # 3. Edge first
    order_edge = []
    l, r = 1, max_val
    while l <= r:
        order_edge.append(l)
        if l != r:
            order_edge.append(r)
        l += 1
        r -= 1
        
    orders = [order_low_conflict, order_edge]
    
    # Add some random orders
    for _ in range(20):
        rnd = all_nums[:]
        random.shuffle(rnd)
        orders.append(rnd)

    for order in orders:
        if time.time() > deadline - 10.0:
            break
            
        current_set = []
        current_set_set = set()
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        # Greedy
        for cand in order:
            if time.time() > deadline - 2.0:
                break
            if _can_add_with_dp(dp, current_set, cand):
                current_set.append(cand)
                current_set_set.add(cand)
                k_prev = len(current_set) - 1
                for k in range(k_prev + 1, 0, -1):
                    dp[k] |= (dp[k - 1] << cand)
                    
        # Local Search
        # Try to improve
        rejected = [x for x in all_nums if x not in current_set_set]
        
        # Limit local search steps
        max_ls_steps = 50
        for _ in range(max_ls_steps):
            if time.time() > deadline - 2.0:
                break
            if not current_set or not rejected:
                break
                
            # Pick a random element to remove
            idx = random.randint(0, len(current_set) - 1)
            removed = current_set[idx]
            current_set.pop(idx)
            current_set_set.remove(removed)
            
            # Rebuild DP
            dp = _rebuild_dp(current_set)
            
            # Try to add elements from rejected
            # We want to add as many as possible
            added_any = False
            # Sort rejected by conflict? Or random?
            # Let's try adding random rejected elements
            # To avoid O(N^2) checks, we can just iterate through a shuffled copy of rejected
            # and add if possible.
            
            # But we need to update DP as we add.
            # And we need to update 'rejected' list?
            # If we add y, y is no longer rejected.
            
            # Let's try to add one element at a time.
            # Pick a random y from rejected.
            # If it fits, add it.
            # If we add it, we have "repaired" the hole.
            # Then we can try to add MORE.
            
            # Strategy: 
            # 1. Remove x.
            # 2. Try to add random y from rejected.
            # 3. If y fits, add it.
            # 4. Try to add random z from rejected (excluding y).
            # 5. If z fits, add it.
            # 6. If we added > 1 element, we improved size.
            # 7. If we added 1 element, size is same, but set changed.
            #    We can keep the new set if it allows further growth?
            #    Hard to know.
            #    Let's just keep the set if size >= previous size.
            
            # Actually, if we remove x and add y, size is same.
            # If we remove x and add y, z, size is +1.
            
            # Let's try to greedily fill the hole with as many rejected elements as possible.
            # But we must ensure we don't take too long.
            
            # Pick a random subset of rejected to try?
            # Or just try a few.
            
            success = False
            # Try to add one element
            if rejected:
                # Pick a few random candidates
                candidates_to_try = random.sample(rejected, min(5, len(rejected)))
                for y in candidates_to_try:
                    if _can_add_with_dp(dp, current_set, y):
                        # Add y
                        current_set.append(y)
                        current_set_set.add(y)
                        rejected.remove(y) # O(N)
                        k_prev = len(current_set) - 1
                        for k in range(k_prev + 1, 0, -1):
                            dp[k] |= (dp[k - 1] << y)
                        success = True
                        # Try to add more
                        if rejected:
                            candidates_to_try2 = random.sample(rejected, min(5, len(rejected)))
                            for z in candidates_to_try2:
                                if _can_add_with_dp(dp, current_set, z):
                                    current_set.append(z)
                                    current_set_set.add(z)
                                    rejected.remove(z)
                                    k_prev = len(current_set) - 1
                                    for k in range(k_prev + 1, 0, -1):
                                        dp[k] |= (dp[k - 1] << z)
                                    success = True
                                    break # One extra is enough for now?
                        break # Stop trying other y if we added one?
                        
            if len(current_set) > best_len:
                best_set = current_set
                best_len = len(best_set)
                if best_len >= 48:
                    return best_set

    return best_set
