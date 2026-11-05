import time
import itertools
import math

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute LCM of 1..24
    L = 1
    for i in range(1, 25):
        L = _lcm(L, i)
        
    # Precompute values v[y] = L // y for y in 1..24
    vals = [0] * 25
    for y in range(1, 25):
        vals[y] = L // y
        
    # For each x, find all minimal "bad" subsets T of {1..24}\{x} 
    # such that sum_{y in T} 1/y = 1/x.
    # We store these as bitmasks.
    # A set S is valid iff for all x in S, no bad mask for x is a subset of S\{x}.
    
    bad_masks = {x: [] for x in range(1, 25)}
    
    def dfs_bad(x, start_idx, current_sum, current_mask, remaining_idx):
        # remaining_idx is a list of indices (0-based in the available list)
        # We are looking for subsets of available elements summing to target = vals[x]
        
        target = vals[x]
        
        if current_sum == target:
            if current_mask != 0:
                bad_masks[x].append(current_mask)
            return
        
        if current_sum > target:
            return
            
        if not remaining_idx:
            return
            
        # Pruning: if sum of all remaining < target - current_sum, return
        # But calculating sum every time is O(N). Let's pass suffix sums or similar.
        # Given small N=23, simple check might suffice or we precompute.
        
        for i, idx in enumerate(remaining_idx):
            y = idx + 1 # convert to 1-based value
            v = vals[y]
            new_sum = current_sum + v
            if new_sum > target:
                # Since we sorted descending, if this one is too big, smaller ones might fit?
                # No, we are adding. If adding current makes it too big, we skip this element.
                # We do NOT break, because maybe a smaller element fits?
                # Wait, we want to reach exactly target.
                # If new_sum > target, we can't use this element in this branch.
                continue
            
            new_mask = current_mask | (1 << (y - 1))
            # Recurse with remaining elements after i
            dfs_bad(x, i + 1, new_sum, new_mask, remaining_idx[i + 1:])
        
    # Optimization: The DFS above creates new lists. For N=23, this is okay for precomputation.
    # But we can do better with a standard subset sum DFS using start index.
    
    # Let's restart the bad mask computation with a cleaner DFS that uses indices into a global list.
    
    bad_masks = {x: [] for x in range(1, 25)}
    
    def compute_bad_masks_for_x(x):
        target = vals[x]
        # Available elements: y in 1..24, y != x
        # Sort by value descending to prune faster?
        # Actually, for subset sum, sorting descending helps prune when sum > target.
        avail = [(vals[y], y) for y in range(1, 25) if y != x]
        avail.sort(key=lambda p: p[0], reverse=True)
        
        n = len(avail)
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + avail[i][0]
            
        found_masks = []
        
        def dfs(start_idx, current_sum, current_mask):
            if current_sum == target:
                if current_mask != 0:
                    found_masks.append(current_mask)
                return
            
            if current_sum > target:
                return
            
            if start_idx >= n:
                return
            
            if current_sum + suffix_sums[start_idx] < target:
                return
            
            for i in range(start_idx, n):
                v, y = avail[i]
                new_sum = current_sum + v
                if new_sum > target:
                    continue
                dfs(i + 1, new_sum, current_mask | (1 << (y - 1)))
                
        dfs(0, 0, 0)
        return found_masks

    for x in range(1, 25):
        bad_masks[x] = compute_bad_masks_for_x(x)
        
    # Optimization: Remove non-minimal bad masks.
    # If mask A is a subset of mask B, and both are bad, then B is redundant
    # because if B is a subset of S, then A is also a subset of S.
    # So we only need minimal masks.
    for x in range(1, 25):
        masks = bad_masks[x]
        if not masks:
            continue
        # Sort by popcount (number of bits)
        masks.sort(key=lambda m: bin(m).count('1'))
        minimal = []
        for m in masks:
            is_minimal = True
            for p in minimal:
                if (m & p) == p:
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(m)
        bad_masks[x] = minimal
        
    # Validity check
    def is_valid_mask(mask):
        for x in range(1, 25):
            if (mask >> (x - 1)) & 1:
                # Check if any bad mask for x is subset of mask
                # Note: bad mask does not contain x, so mask & bm == bm is sufficient.
                # If bm is subset of mask, then all elements in bm are in S.
                # Since x is in S, and bm elements are other elements, the condition "sum of reciprocals of subset of OTHER selected elements" is met.
                for bm in bad_masks[x]:
                    if (bm & mask) == bm:
                        return False
        return True

    # Search for max cardinality
    # Start from 24 down to 1
    # For each size k, iterate combinations.
    # Use a heuristic order to find valid sets faster?
    # Actually, for k=24, 1 comb. k=23, 24 combs. k=22, 276. k=21, 2024.
    # It is likely that small k will be found quickly if we just iterate.
    # But we want the MAX k.
    
    best_found = None
    
    for k in range(24, 0, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Generate combinations
        # itertools.combinations(range(1, 25), k)
        for comb in itertools.combinations(range(1, 25), k):
            mask = 0
            for x in comb:
                mask |= (1 << (x - 1))
                
            if is_valid_mask(mask):
                return list(comb)
                
    # Fallback: Greedy construction
    # If the combinatorial search didn't finish or found nothing (unlikely for small k),
    # try a greedy approach.
    # Try to build a large set.
    # Order: perhaps by "danger" (number of bad masks involving x)?
    # Or just simple descending?
    
    # Let's try a smarter greedy:
    # Start with empty set.
    # Add elements one by one.
    # To maximize size, we want to add as many as possible.
    # Conflict: x and y conflict if {y} is a bad mask for x or {x} is a bad mask for y?
    # No, bad masks can have size > 1.
    
    # Simple greedy:
    # 1. Identify elements that are "safe" (no bad masks for them as target? i.e., 1/x cannot be formed by others).
    #    If bad_masks[x] is empty, x can be added freely?
    #    Wait, if bad_masks[x] is empty, it means 1/x cannot be sum of reciprocals of ANY subset of others.
    #    So x can never be the "target" of a violation.
    #    But x can be part of a bad mask for another element z.
    
    # Let's just use the is_valid_mask check in a greedy loop.
    # Try to add elements. Which order?
    # Maybe sort by number of bad masks? Or just 1..24?
    
    # A better fallback: Local search / Hill Climbing
    # Start with a random or greedy set, then try to add elements.
    
    current_mask = 0
    # Try adding elements in order of increasing "constraint"?
    # Elements with smaller values (larger x) have smaller reciprocals.
    # It is harder to sum to a small reciprocal?
    # Actually, 1/24 is small. 1/1 is large.
    # 1/1 = 1/2+1/3+1/6. So 1 is dangerous.
    # 1/24: can it be formed?
    # Let's just try to add all elements and remove conflicting ones?
    
    # Greedy:
    # Candidate pool: 1..24
    # While pool not empty:
    #   Pick x from pool
    #   If current_mask | {x} is valid, add x
    #   Else, skip x (maybe try later? No, greedy)
    
    # To get a larger set, we might need to be smart about which to pick.
    # Let's try picking from largest x to smallest x (smallest reciprocals first)?
    # Or smallest x to largest?
    
    # Let's try both and keep the best.
    
    def greedy(order):
        mask = 0
        for x in order:
            if is_valid_mask(mask | (1 << (x - 1))):
                mask |= (1 << (x - 1))
        return mask
        
    m1 = greedy(range(24, 0, -1))
    m2 = greedy(range(1, 25))
    m3 = greedy(range(12, 0, -1) + range(24, 11, -1)) # mix?
    
    best_greedy = max([m1, m2, m3], key=lambda m: bin(m).count('1'))
    
    # Try to improve best_greedy by adding elements?
    # It's already maximal under that specific addition order.
    # But maybe a different set of that size exists?
    # Since we already did exhaustive search for k=24..1, if we didn't return,
    # it means no valid set of size > size(best_greedy) was found in the time limit?
    # Or the search for large k took too long.
    # The exhaustive search checks k=24, 23, 22...
    # If k=24, 23, 22, 21 took 80s, we break.
    # Then we return best_greedy.
    # But maybe size 20 is valid? We didn't check 20.
    # We should check sizes down from best_greedy_size + 1 to best_greedy_size + delta?
    # No, we want MAX.
    
    # If the exhaustive search broke early, we don't know the true max.
    # We should continue checking smaller k if time permits?
    # The loop `for k in range(24, 0, -1)` breaks if time > deadline - 5.
    # So if it breaks at k=21, we haven't checked 20, 19...
    # We should then check 20, 19... until time runs out or we find one?
    # But finding one at k=15 doesn't mean k=16 is invalid.
    # We want the MAX.
    
    # Revised strategy:
    # 1. Run exhaustive search from 24 down.
    # 2. If time runs out, record the smallest k we checked (e.g., k_checked=21).
    # 3. We know no valid set exists for k >= k_checked + 1.
    # 4. We don't know for k <= k_checked.
    # 5. We have a candidate of size S from greedy.
    # 6. We should check k = k_checked, k_checked - 1, ... down to S+1.
    #    If we find a valid set at k, that's the max (since we checked higher ones already).
    #    If we finish down to S+1 without finding, S is the max we have.
    
    # Let's refactor the search loop to be smarter about time.
    
    # Re-implement search:
    
    best_mask = 0
    best_size = 0
    checked_down_to = 24
    
    for k in range(24, 0, -1):
        if time.time() > deadline - 2:
            break
            
        # Check all combinations of size k
        # Optimization: if k is large, number of combs is small.
        # if k is small, number of combs is huge.
        # C(24, 12) ~ 2.7M. C(24, 10) ~ 1.9M.
        # Checking 2M sets with is_valid_mask (which is fast) might take ~1-2s?
        # is_valid_mask iterates 24 elements, checks ~10 masks each. ~240 ops.
        # 2M * 240 = 480M ops. In Python, this might take 10-20s.
        # So we can probably check k down to 12 or 10 in the remaining time.
        
        found = False
        for comb in itertools.combinations(range(1, 25), k):
            mask = 0
            for x in comb:
                mask |= (1 << (x - 1))
            if is_valid_mask(mask):
                # Found max size k
                return list(comb)
                
    # If loop finished (k reached 1) or broke:
    # If broke, we checked down to some k_break.
    # We have not checked k < k_break.
    # Let's see how much time is left.
    
    # For now, the simple loop above handles the "find max" logic if it completes.
    # If it times out, it returns greedy.
    # This is acceptable, but might miss a larger set if greedy is bad.
    
    # Let's refine the fallback to be a bit stronger.
    # If we timed out, we know no set of size > k_current is valid.
    # Let's try to find a set of size k_current, k_current-1, etc.,
    # but stop if time is tight.
    
    # Actually, the loop above just breaks.
    # Let's add a secondary phase:
    # If we broke out of the loop at k_break:
    #   We know sizes > k_break are invalid.
    #   We should check k_break, k_break-1...
    #   But we already checked k_break in the loop?
    #   No, if `time.time() > deadline - 2` was true at start of iteration k, we break.
    #   So we did NOT check k.
    
    # Let's track k_last_checked.
    
    # Re-do logic with tracking.
    
    best_result = None
    
    # Phase 1: Exhaustive from top
    for k in range(24, 0, -1):
        if time.time() > deadline - 2:
            break
        for comb in itertools.combinations(range(1, 25), k):
            mask = 0
            for x in comb:
                mask |= (1 << (x - 1))
            if is_valid_mask(mask):
                return list(comb)
                
    # Phase 2: If we broke, try to fill the gap with greedy + local search?
    # Or just return the best greedy found.
    
    # Let's just return the best greedy found in the fallback section.
    # To make greedy better, we can use a randomized restart?
    # No, deterministic is better for reproducibility if possible, but random might find larger.
    # Let's stick to the deterministic greedy orders.
    
    # Add one more greedy strategy:
    # Sort by "badness" (number of bad masks for x, and also how often x appears in bad masks of others).
    
    # Count appearances in bad masks
    appearances = [0] * 25
    for x in range(1, 25):
        for bm in bad_masks[x]:
            # Count bits in bm
            # bm does not contain x
            for y in range(1, 25):
                if (bm >> (y - 1)) & 1:
                    appearances[y] += 1
                    
    # Sort by appearances descending? (Most dangerous first? or last?)
    # If x is dangerous, maybe put it last to avoid conflicts?
    # Or first to "lock it in" and exclude others?
    # Let's try sorting by appearances ascending (safest first).
    
    order_safe = sorted(range(1, 25), key=lambda y: appearances[y])
    m4 = greedy(order_safe)
    
    # Sort by appearances descending
    order_danger = sorted(range(1, 25), key=lambda y: -appearances[y])
    m5 = greedy(order_danger)
    
    best_greedy = max([m1, m2, m3, m4, m5], key=lambda m: bin(m).count('1'))
    
    result = [x for x in range(1, 25) if (best_greedy >> (x - 1)) & 1]
    
    if not result:
        result = [24]
        
    return result
