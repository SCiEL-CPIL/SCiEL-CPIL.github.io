import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Helper to add two fractions represented as (num, den)
    def add_frac(f1, f2):
        n1, d1 = f1
        n2, d2 = f2
        num = n1 * d2 + n2 * d1
        den = d1 * d2
        g = math.gcd(num, den)
        if g > 1:
            num //= g
            den //= g
        return (num, den)
    
    # Helper to check if target (1, x) is a sum of a subset of others
    # others: list of integers
    def is_safe(x, others):
        if not others:
            return True
        
        target = (1, x)
        n = len(others)
        if n == 0:
            return True
        
        # Meet in the middle
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        # Generate all subset sums for left
        left_sums = {(0, 1)}
        for val in left:
            new_sums = set()
            for s in left_sums:
                ns = add_frac(s, (1, val))
                new_sums.add(ns)
            left_sums.update(new_sums)
            
        # Generate all subset sums for right and store in a set
        right_sums = set()
        current_sums = {(0, 1)}
        for val in right:
            new_sums = set()
            for s in current_sums:
                ns = add_frac(s, (1, val))
                new_sums.add(ns)
            current_sums.update(new_sums)
        right_sums.update(current_sums)
        
        # Check if any left_sum + right_sum == target
        # We need to check if (target - left_sum) is in right_sums
        # Note: target - left_sum might be negative or zero.
        # Since all sums are positive, if target - left_sum <= 0, it can't be in right_sums (unless 0, but target>0 so target-left_sum=0 implies left_sum=target, which means right_sum=0, i.e., empty right subset. This is a valid non-empty subset of the whole if left_subset is non-empty. If left_subset is empty, left_sum=0, target-right_sum=0 -> right_sum=target, valid non-empty right subset).
        
        for ls in left_sums:
            # Calculate target - ls
            # target = (1, x), ls = (n, d)
            # diff = (1, x) - (n, d) = (d - nx, xd)
            n_ls, d_ls = ls
            num_diff = d_ls - n_ls * x
            den_diff = d_ls * x
            
            if num_diff == 0:
                # diff is 0. We need right_sum = 0.
                # 0 is in right_sums (empty subset).
                # Is the combined subset non-empty?
                # If ls == 0 (empty left), then right subset must be non-empty to sum to target? No, if ls=0 and diff=0, target=0, impossible.
                # If ls != 0, left subset is non-empty. Right subset is empty. Total subset is non-empty. Valid violation.
                # If ls == 0, then target = 0, impossible.
                # So if num_diff == 0, it means ls == target.
                # This implies a subset of left sums to target. This is a valid violation.
                return False
            
            if num_diff < 0:
                # diff is negative. right_sums are all positive (or 0).
                # Can't match.
                continue
                
            g = math.gcd(num_diff, den_diff)
            if g > 1:
                num_diff //= g
                den_diff //= g
                
            if (num_diff, den_diff) in right_sums:
                return False
                
        return True

    def is_valid_full(subset):
        s = set(subset)
        for x in s:
            others = [y for y in s if y != x]
            if not is_safe(x, others):
                return False
        return True

    # Strategy 1: Greedy Descending
    def greedy_descending():
        selected = []
        for i in range(24, 0, -1):
            if time.time() > deadline:
                break
            if is_safe(i, selected):
                selected.append(i)
        return set(selected)

    # Strategy 2: Greedy Ascending
    def greedy_ascending():
        selected = []
        for i in range(1, 25):
            if time.time() > deadline:
                break
            if is_safe(i, selected):
                selected.append(i)
        return set(selected)

    # Strategy 3: Randomized Greedy
    def greedy_randomized():
        selected = []
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        for i in candidates:
            if time.time() > deadline:
                break
            if is_safe(i, selected):
                selected.append(i)
        return set(selected)

    best_set = set()
    best_size = 0
    
    # Run greedy strategies
    s1 = greedy_descending()
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    s2 = greedy_ascending()
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # Run a few randomized greeds
    for _ in range(5):
        if time.time() > deadline:
            break
        sr = greedy_randomized()
        if len(sr) > best_size:
            best_set = sr
            best_size = len(sr)

    # Local Search (Hill Climbing)
    # We try to improve best_set
    # Moves: Add, Remove, Swap
    # We prioritize increasing size.
    
    current_set = best_set
    current_size = best_size
    
    # Cache for is_safe checks? 
    # The set changes, so caching is tricky. 
    # But is_safe(x, others) depends only on x and others.
    # We can cache (x, tuple(sorted(others))) -> bool
    safe_cache = {}
    
    def is_safe_cached(x, others_list):
        key = (x, tuple(sorted(others_list)))
        if key in safe_cache:
            return safe_cache[key]
        res = is_safe(x, others_list)
        safe_cache[key] = res
        return res

    # We need to re-implement is_safe using the cache in the loop? 
    # No, let's just use the direct function but maybe add caching to the main loop logic if it's slow.
    # Given N=24, is_safe is ~2^12 ops. 24 * 2^12 ~ 100k ops per full check.
    # Local search steps might be 1000s. 100k * 1000 = 100M ops. Might be tight in Python.
    # Let's try to optimize is_safe further or limit search depth.
    
    # Let's use the cached version for the local search steps.
    # We'll wrap the check logic to use the cache.
    
    def check_validity_with_cache(subset):
        s = set(subset)
        for x in s:
            others = [y for y in s if y != x]
            if not is_safe_cached(x, others):
                return False
        return True

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements
        excluded = [i for i in range(1, 25) if i not in current_set]
        random.shuffle(excluded)
        for x in excluded:
            if time.time() > deadline:
                break
            new_set = current_set | {x}
            if check_validity_with_cache(new_set):
                current_set = new_set
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_set = current_set
                    best_size = current_size
                # Restart add loop with new set
                excluded = [i for i in range(1, 25) if i not in current_set]
                random.shuffle(excluded)
                break # Restart the for loop
        
        # If no add worked, try swaps
        if not improved:
            # Try 1-for-1 swaps
            # Remove y from current, add x from excluded
            # We want to maintain or increase size. Since size is fixed in a swap, we just want to find a valid swap that might allow further adds later?
            # Or maybe the current set is a local optimum for adds, but a swap could lead to a better configuration.
            # However, if we can't add anything, and we swap, we still have the same size.
            # The goal is to maximize size. If we are stuck at size K, swapping doesn't increase size immediately.
            # But it might unblock future adds.
            
            # Let's try a few random swaps to see if they lead to a state where an add is possible.
            excluded = [i for i in range(1, 25) if i not in current_set]
            if excluded and current_set:
                for _ in range(20): # Limit number of swap attempts per stall
                    if time.time() > deadline:
                        break
                    x = random.choice(excluded)
                    y = random.choice(list(current_set))
                    new_set = (current_set - {y}) | {x}
                    if check_validity_with_cache(new_set):
                        # Accept the swap? 
                        # It doesn't increase size. 
                        # But it might be a better "base".
                        # Let's accept it with a small probability or if it seems "better" (hard to define).
                        # For now, let's just try to add to the new set immediately.
                        # Actually, if we accept a swap, we might loop forever swapping without progress.
                        # Let's only accept swaps if they allow an immediate add?
                        # No, let's just try the swap and then check for adds.
                        current_set = new_set
                        # Try to add to this new set
                        excluded_new = [i for i in range(1, 25) if i not in current_set]
                        random.shuffle(excluded_new)
                        added = False
                        for z in excluded_new:
                            if time.time() > deadline:
                                break
                            new_set2 = current_set | {z}
                            if check_validity_with_cache(new_set2):
                                current_set = new_set2
                                current_size += 1
                                improved = True
                                if current_size > best_size:
                                    best_set = current_set
                                    best_size = current_size
                                added = True
                                break
                        if added:
                            break
                        # If no add, we might revert? Or keep the swap?
                        # Keeping the swap might be good for diversity.
                        # But if we don't improve size, we risk wasting time.
                        # Let's revert if no add happened, to stay on the "best known size" path?
                        # Actually, the "best known size" is global best_size. 
                        # If we are stuck at size K, and best_size is K, swapping doesn't help.
                        # If best_size > K (impossible as we only grow), it would help.
                        # So if we are stuck, swapping is only useful if it leads to an add.
                        # Since we tried to add and failed, we should probably revert to the previous state to avoid wandering.
                        current_set = (current_set - {x}) | {y} # Revert
                        # Actually, reverting requires knowing y and x.
                        # This logic is getting messy. Let's simplify.
                        
                        # Simplified Local Search:
                        # 1. Try to add. If success, done for this iteration.
                        # 2. If fail, try a limited number of swaps. For each swap, try to add.
                        #    If a swap+add succeeds, accept both.
                        #    If no swap+add succeeds, stop improving.
                        
                        # I will implement this cleaner logic below.
                        pass

    # Re-implementing the local search loop more cleanly
    current_set = best_set
    current_size = best_size
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add
        excluded = [i for i in range(1, 25) if i not in current_set]
        random.shuffle(excluded)
        for x in excluded:
            new_set = current_set | {x}
            if check_validity_with_cache(new_set):
                current_set = new_set
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_set = current_set
                    best_size = current_size
                break
        
        if improved:
            continue
            
        # 2. If no add, try swaps that lead to an add
        # We try a few random swaps. For each, we check if the new set is valid.
        # If valid, we try to add to it. If an add is possible, we accept the swap+add.
        
        excluded = [i for i in range(1, 25) if i not in current_set]
        if not excluded or not current_set:
            break
            
        found_improvement = False
        for _ in range(50): # Try 50 random swaps
            if time.time() > deadline:
                break
            x = random.choice(excluded)
            y = random.choice(list(current_set))
            
            # Swap y with x
            swapped_set = (current_set - {y}) | {x}
            
            # Check if swapped_set is valid
            if not check_validity_with_cache(swapped_set):
                continue
                
            # Try to add to swapped_set
            new_excluded = [i for i in range(1, 25) if i not in swapped_set]
            random.shuffle(new_excluded)
            for z in new_excluded:
                if z == y: # z is the element we just removed, adding it back is no-op effectively
                    continue
                new_set2 = swapped_set | {z}
                if check_validity_with_cache(new_set2):
                    # Accept swap and add
                    current_set = new_set2
                    current_size += 1 # Size increases by 1 (remove 1, add 1, add 1 -> net +1? No. Remove 1, Add 1 (swap) -> size same. Add 1 (z) -> size +1. Yes.)
                    improved = True
                    if current_size > best_size:
                        best_set = current_set
                        best_size = current_size
                    found_improvement = True
                    break
            if found_improvement:
                break
                
        if not found_improvement:
            # No improvement found in this round of adds/swaps.
            # We can try a different strategy or stop.
            # Given the small search space, if greedy + local search doesn't find a larger set, it's likely the optimum or very close.
            # We can try a few more random restarts if time permits.
            
            # Try a random restart from scratch?
            # Or just break.
            # Let's try 5 more random greedy starts if we have time.
            if time.time() < deadline - 10:
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    s_r = greedy_randomized()
                    if len(s_r) > best_size:
                        best_set = s_r
                        best_size = len(best_set)
                        # Restart local search from this new best?
                        # For simplicity, just update best and let the loop try to improve it?
                        # The loop uses current_set. Let's update current_set.
                        current_set = best_set
                        current_size = best_size
                        improved = True # Force another iteration
                        break
            else:
                break

    if not best_set:
        return [24]
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
