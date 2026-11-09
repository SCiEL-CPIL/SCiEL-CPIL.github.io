"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    # We will use a deterministic backtracking approach.
    # The key insight is that checking divisibility can be done efficiently
    # by iterating through each sum 'a' and checking if any multiple 2a, 3a, ... 
    # exists in the set of sums. This is much faster than O(N^2) pair checks
    # when the maximum sum is bounded (<= 256 * 30 = 7680).
    
    MAX_VAL = 256
    MAX_SIZE_TARGET = 15  # We try to find a set of this size, or as close as possible
    
    # Pre-compute nothing, just use the logic.
    
    def is_valid_sums(sums_list):
        """
        Check if a list of subset sums satisfies the non-divisibility condition.
        Assumes sums_list contains distinct values.
        """
        if not sums_list:
            return True
        
        sums_set = set(sums_list)
        max_sum = max(sums_list)
        
        # For each sum a, check if any multiple k*a (k>=2) is in the set
        # We can optimize by only checking a if a <= max_sum // 2
        for a in sums_list:
            if a * 2 > max_sum:
                continue
            multiple = 2 * a
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += a
        return True

    def backtrack(current_set, current_sums, target_size, candidates):
        """
        Recursively try to add candidates to current_set to reach target_size.
        current_sums is a list of all non-empty subset sums of current_set.
        candidates is a list of available integers.
        """
        if len(current_set) == target_size:
            return current_set[:]
        
        # Optimization: If we can't reach target_size with remaining candidates, return None
        # But we just iterate, so if loop finishes, return None.
        
        for i, cand in enumerate(candidates):
            # Try adding cand
            new_set = current_set + [cand]
            
            # Compute new subset sums
            # New sums are:
            # 1. All old sums
            # 2. cand
            # 3. old_sum + cand for all old_sum in current_sums
            new_sums = list(current_sums)
            new_sums.append(cand)
            for s in current_sums:
                new_sums.append(s + cand)
            
            # Check for duplicates
            if len(set(new_sums)) != len(new_sums):
                continue
            
            # Check divisibility
            if not is_valid_sums(new_sums):
                continue
            
            # If valid, recurse
            # We need to pass a new candidates list excluding cand
            # To avoid modifying the list, we can pass indices or create a new list
            # Creating a new list is O(N), which is fine for N=256
            new_candidates = candidates[:i] + candidates[i+1:]
            
            result = backtrack(new_set, new_sums, target_size, new_candidates)
            if result is not None:
                return result
        
        return None

    # We want to maximize size.
    # We'll try target sizes from high to low? Or low to high?
    # Low to high is safer to ensure we get *some* result.
    # But we want to maximize.
    # Let's try to find the maximum possible size up to 15.
    
    # Candidates: 1 to 256.
    # To improve search speed, we might want to order candidates.
    # Larger numbers often lead to fewer divisibility conflicts in the "multiple" check?
    # Actually, small numbers are more likely to divide large sums.
    # So avoiding small numbers might be beneficial.
    # Let's try candidates in descending order.
    
    candidates = list(range(256, 0, -1))
    
    best_set = []
    
    # Try increasing sizes
    # Start from a reasonable size, e.g., 10, and go up.
    # Or start from 15 and go down if 15 is too hard.
    # Given 60s, let's try 12, 13, 14.
    
    for target in range(15, 5, -1):
        # Reset candidates for each target? No, we can reuse the logic.
        # But the search space is huge.
        # Let's try a single run for a high target.
        # If it fails, try lower.
        
        # To speed up, we can use a timeout or node limit.
        # For now, let's just run it.
        # We'll add a simple node counter to prevent infinite hangs if logic is flawed.
        
        node_count = 0
        max_nodes = 50000 # Limit to ensure 60s
        
        def backtrack_limited(current_set, current_sums, target_size, candidates, node_limit):
            nonlocal node_count
            node_count += 1
            if node_count > node_limit:
                return None
            
            if len(current_set) == target_size:
                return current_set[:]
            
            # Pruning: If remaining candidates + current size < target, fail
            if len(current_set) + len(candidates) < target_size:
                return None
                
            for i, cand in enumerate(candidates):
                new_set = current_set + [cand]
                
                # Compute new sums
                # Optimization: Use a set for duplicates check first?
                # new_sums = current_sums + [cand] + [s+cand for s in current_sums]
                # This creates a list.
                
                # Check duplicates efficiently
                # We know current_sums are distinct.
                # We need to check if cand is in current_sums (it isn't, as cand > max(current_sums)? No, cand can be small).
                # Actually, cand is a new element.
                # Duplicates can occur if:
                # 1. cand is in current_sums
                # 2. s1 + cand == s2 for some s1, s2 in current_sums
                # 3. s1 + cand == cand => s1 == 0 (impossible for non-empty)
                # 4. s1 + cand == s2 + cand => s1 == s2 (impossible)
                
                # So we just need to check if cand is in current_sums,
                # and if any (s + cand) is in current_sums or equals another (s' + cand).
                # Since current_sums are distinct, s+cand == s'+cand implies s==s'.
                # So we just need to check if (s + cand) is in current_sums for any s.
                
                # Let's just build the set and check length. It's fast enough for small N.
                new_sums_set = set(current_sums)
                new_sums_set.add(cand)
                for s in current_sums:
                    new_sums_set.add(s + cand)
                
                if len(new_sums_set) != len(current_sums) * 2 + 1:
                    continue
                
                new_sums_list = list(new_sums_set)
                
                if not is_valid_sums(new_sums_list):
                    continue
                
                new_candidates = candidates[:i] + candidates[i+1:]
                
                result = backtrack_limited(new_set, new_sums_list, target_size, new_candidates, node_limit)
                if result is not None:
                    return result
            
            return None

        result = backtrack_limited([], [], target, candidates, max_nodes)
        if result is not None:
            best_set = result
            break

    if not best_set:
        # Fallback to a known valid small set
        # [3, 7, 19] was checked in comments? 
        # Let's verify [3, 7, 19]
        # Sums: 3, 7, 10, 19, 22, 26, 29
        # Divisibility:
        # 3 divides 22? No. 3 divides 26? No. 3 divides 29? No.
        # 7 divides 22? No. 7 divides 26? No. 7 divides 29? No.
        # 10 divides ... No.
        # So [3, 7, 19] is valid.
        best_set = [3, 7, 19]
        
    return list(best_set)

# EVOLVE-BLOCK-END
