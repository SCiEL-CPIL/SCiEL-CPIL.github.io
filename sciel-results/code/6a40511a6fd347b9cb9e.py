import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def is_valid(s):
        n = len(s)
        # dp[c] is a set of sums for subsets of size c
        # Using sets of integers
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for x in s:
            # Iterate backwards to avoid using x multiple times in the same step
            for c in range(n - 1, -1, -1):
                if dp[c]:
                    # Optimization: if dp[c] is large, this is slow.
                    # But n is small (max ~20-30 expected), so sets are manageable.
                    new_sums = {v + x for v in dp[c]}
                    dp[c + 1].update(new_sums)
        
        # Check for collisions across different cardinalities
        seen = set()
        for c in range(n + 1):
            if not seen.isdisjoint(dp[c]):
                return False
            seen.update(dp[c])
        return True

    def try_extend(current_set, candidates, time_limit):
        # Try to add elements from candidates to current_set
        # candidates should be a list of ints not in current_set
        current = list(current_set)
        c_set = set(current)
        
        # Sort candidates to try smaller ones first? Or larger?
        # Smaller elements usually allow more elements to fit in 1..256.
        # Let's try a mix.
        
        # We will iterate through candidates and try to add them if valid.
        # To be more effective, we can do a local search:
        # 1. Try adding a new element.
        # 2. If stuck, try swapping an element.
        
        # Simple greedy first
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in c_set:
                continue
            new_set = current + [x]
            if is_valid(new_set):
                current = new_set
                c_set.add(x)
                
        # Local search refinement
        # Try to improve by swapping
        # We want to maximize size, so if we can't add, we try to swap 
        # a "bad" element for a "good" one that allows further growth?
        # Actually, the greedy usually gets stuck because it picks elements 
        # that block future additions.
        # A better local search: 
        # While time permits:
        #   Pick a random element in current, remove it.
        #   Try to add a random element from 1..256 not in current.
        #   If valid, keep the change.
        
        best_size = len(current)
        best_set = list(current)
        
        rng = random.Random(42)
        
        # Perform a limited number of random restarts/swaps
        # Since is_valid is O(n * 2^n) roughly (actually O(n * S) where S is num sums),
        # and n is small, we can do many checks.
        
        # Let's do a hill-climbing style search on the set itself.
        # Objective: Maximize size.
        # Move: Swap one element for another.
        
        # To avoid infinite loops, we track visited states or just run for a fixed number of steps.
        
        steps = 0
        max_steps = 2000 # Adjust based on speed
        
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            if not current:
                break
            
            # Pick a random element to remove
            idx = rng.randrange(len(current))
            removed = current.pop(idx)
            c_set.remove(removed)
            
            # Try to add a random element from 1..256
            # To bias towards finding larger sets, we might want to try
            # elements that are "compatible" with the current set.
            # But random sampling from 1..256 is simple.
            # We can also try to add the smallest available element first?
            # Let's try a few candidates.
            
            success = False
            # Try a few random candidates
            for _ in range(10):
                x = rng.randint(1, 256)
                if x in c_set:
                    continue
                new_set = current + [x]
                if is_valid(new_set):
                    current = new_set
                    c_set.add(x)
                    success = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                    break
            
            if not success:
                # Restore removed element
                current.append(removed)
                c_set.add(removed)
                # Sort to keep consistent state? Not strictly necessary for set logic,
                # but good for determinism if we were caching.
                
        return best_set

    # 1. Generate a strong baseline using Arithmetic Progressions
    # An AP {a, a+d, a+2d, ..., a+(k-1)d} has subset sums that are
    # multiples of d plus a * c.
    # Specifically, sum of c elements = c*a + d * (sum of c distinct indices from 0..k-1).
    # The set of possible values for (sum of c distinct indices) is an interval
    # [c(c-1)/2, c(2k-c-1)/2].
    # So sums for cardinality c lie in [c*a + d*c(c-1)/2, c*a + d*c(2k-c-1)/2].
    # For disjointness, we need these intervals for different c to be disjoint.
    # Interval I_c = [L_c, R_c].
    # L_c = c*a + d * c(c-1)/2
    # R_c = c*a + d * c(2k-c-1)/2
    # We need R_c < L_{c+1} for all c.
    # R_c < L_{c+1}
    # c*a + d * c(2k-c-1)/2 < (c+1)*a + d * (c+1)c/2
    # d * c(2k-c-1)/2 - d * c(c+1)/2 < a
    # d * c/2 * (2k - c - 1 - c - 1) < a
    # d * c/2 * (2k - 2c - 2) < a
    # d * c * (k - c - 1) < a
    
    # We want to maximize k (size).
    # Constraints:
    # 1. Elements distinct in 1..256.
    # 2. Max element a + (k-1)d <= 256.
    # 3. Min element a >= 1.
    # 4. Disjointness: d * c * (k - c - 1) < a for all c in 0..k-1.
    #    The max of c(k-c-1) for c in 0..k-1 is at c = (k-1)/2.
    #    Max value is roughly (k-1)^2 / 4.
    #    So we need d * (k-1)^2 / 4 < a.
    
    # Let's search for AP parameters (a, d, k) that satisfy these.
    # We want to maximize k.
    
    best_ap_set = []
    
    for k in range(60, 0, -1):
        found = False
        # Try small d first, as they allow larger k for fixed range?
        # Actually, larger d makes the disjointness condition harder (requires larger a).
        # But larger d allows elements to be spaced further, potentially fitting more?
        # No, range is fixed 1..256.
        # Range = (k-1)d < 256. So d <= 256/(k-1).
        # Disjointness: a > d * max(c(k-c-1)).
        # Also a >= 1.
        # And a + (k-1)d <= 256.
        
        max_c_term = max(c * (k - c - 1) for c in range(k))
        
        for d in range(1, 256 // (k - 1) + 1 if k > 1 else 1):
            min_a_required = d * max_c_term + 1
            max_a_allowed = 256 - (k - 1) * d
            
            if min_a_required <= max_a_allowed:
                # Valid AP exists
                # Pick a = min_a_required (smallest a allows largest d? No, d is fixed here)
                # Actually, any a in [min_a_required, max_a_allowed] works.
                # Let's pick a = min_a_required to keep elements small?
                # Or a = 1? No, a must be >= min_a_required.
                a = min_a_required
                # Check if elements are distinct and in range
                elements = [a + i * d for i in range(k)]
                if all(1 <= e <= 256 for e in elements) and len(set(elements)) == k:
                    # Verify with exact check just in case
                    if is_valid(elements):
                        best_ap_set = elements
                        found = True
                        break
        
        if found:
            break
    
    current_best = best_ap_set
    
    # 2. Refine using local search
    # The AP might not be optimal. We can try to add more elements or swap.
    # We will run the local search starting from the AP.
    
    if time.time() < deadline:
        # Prepare candidates for extension
        # The local search function handles swapping.
        # We need to pass a list of candidates? 
        # My try_extend function uses random candidates from 1..256.
        # It takes current_set and a list of candidates (which it ignores mostly, using random).
        # Let's adapt try_extend to not rely on a specific candidate list but use random sampling.
        
        # Re-defining a more robust local search for the AP baseline
        def refine_ap(ap_set, time_limit):
            current = list(ap_set)
            c_set = set(current)
            best_set = list(current)
            best_size = len(current)
            rng = random.Random(123)
            
            # Hill climbing on size
            # Move: Remove one, add one.
            # If size increases, keep. If size same, keep with some probability (simulated annealing style)?
            # Or just keep if valid.
            
            steps = 0
            max_steps = 5000
            
            while time.time() < time_limit and steps < max_steps:
                steps += 1
                if not current:
                    break
                
                # Pick random element to remove
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                c_set.remove(removed)
                
                success = False
                # Try to add a random element
                # Bias towards elements that might fit? 
                # Just random for now.
                for _ in range(20):
                    x = rng.randint(1, 256)
                    if x in c_set:
                        continue
                    new_set = current + [x]
                    if is_valid(new_set):
                        current = new_set
                        c_set.add(x)
                        success = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = list(current)
                        break
                
                if not success:
                    current.append(removed)
                    c_set.add(removed)
            
            return best_set

        current_best = refine_ap(current_best, deadline)

    # 3. If AP failed or was small, try random greedy as fallback
    if len(current_best) < 10:
        # Fallback to random greedy
        rng = random.Random(99)
        current = []
        c_set = set()
        for _ in range(1000): # Try many times
            if time.time() > deadline:
                break
            x = rng.randint(1, 256)
            if x in c_set:
                continue
            new_set = current + [x]
            if is_valid(new_set):
                current = new_set
                c_set.add(x)
        
        if len(current) > len(current_best):
            current_best = current
            
        # Refine this too if time permits
        if time.time() < deadline and len(current_best) > 0:
             current_best = refine_ap(current_best, deadline)

    # Final check and return
    if is_valid(current_best):
        return [int(x) for x in current_best]
    else:
        # Should not happen, but safety
        return [1]

