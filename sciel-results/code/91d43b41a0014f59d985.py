"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    MAX_N = 512
    MAX_SIZE = 48
    
    def update_dp(dp, x, max_k):
        """Update DP table by adding element x. dp[k] is bitmask of sums of size k."""
        n = len(dp)
        new_dp = [0] * min(n + 1, max_k + 1)
        new_dp[0] = 1
        for k in range(1, min(n + 1, max_k + 1)):
            val = dp[k] if k < n else 0
            if k > 0 and (k - 1) < n:
                val |= (dp[k-1] << x)
            new_dp[k] = val
        return new_dp
    
    def greedy_incremental(order, max_size=48):
        """
        Greedy construction that maintains DP tables for S and S\{a} for each a in S.
        This allows O(N^2) check for adding a new element x.
        """
        selected = []
        n = 0
        dp_all = [1]  # dp_all[k] for current set S
        dp_without = {} # dp_without[a][k] for set S \ {a}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break
            
            # Check 1: Is x the mean of any subset of S of size >= 2?
            valid = True
            for k in range(2, n + 1):
                target = k * x
                # Max sum of k elements from S is bounded by k * 512
                if target > 512 * k: 
                    continue
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in S, is a the mean of any subset of (S U {x}) \ {a} of size >= 2?
            # The subsets of (S U {x}) \ {a} are:
            # 1. Subsets of S \ {a} (already valid, so no violation)
            # 2. Subsets of (S \ {a}) U {x} that contain x.
            #    Let T be a subset of S \ {a} of size k-1.
            #    New subset is T U {x} of size k.
            #    Mean condition: a = (sum(T) + x) / k  => sum(T) = k*a - x.
            #    So we check if (k*a - x) is in dp_without[a][k-1].
            
            for a in selected:
                # k is the size of the subset of (S U {x}) \ {a}
                # Since it must contain x, and size >= 2, k >= 2.
                # Max k is n + 1.
                for k in range(2, n + 2):
                    target_sum = k * a - x
                    if target_sum < 0:
                        continue
                    # Max sum of k-1 elements from S\{a} is (k-1)*512
                    if target_sum > 512 * (k - 1):
                        continue
                    # Check if target_sum is achievable with k-1 elements from S\{a}
                    # dp_without[a] has size n (since S has n elements, S\{a} has n-1)
                    # So indices 0..n-1 are valid. k-1 must be <= n-1 => k <= n.
                    # Wait, if k = n+1, we need n elements from S\{a}, which has size n-1. Impossible.
                    # So k-1 <= n-1 => k <= n.
                    if k - 1 < len(dp_without[a]):
                        if (dp_without[a][k-1] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Accept x
            selected.append(x)
            n += 1
            
            # Update dp_all
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x, max_size)
            
            # Update dp_without
            # For existing a in selected[:-1] (all except x), S\{a} becomes (S_old\{a}) U {x}
            for a in selected[:-1]:
                dp_without[a] = update_dp(dp_without[a], x, max_size)
            
            # For the new x, S\{x} is S_old. So dp_without[x] is old_dp_all.
            dp_without[x] = old_dp_all
        
        return selected

    def validate_set(s):
        """Full bitset verification for safety."""
        n = len(s)
        if n < 3:
            return True
        
        dp_all = [1]
        for x in s:
            dp_all = update_dp(dp_all, x, n)
        
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp_others = [1]
            for x in others:
                dp_others = update_dp(dp_others, x, m)
            
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True
    
    def local_search_improve(s, time_limit=5.0):
        """
        Try to improve the set by swapping elements.
        For each element a in s, try to remove it and add a new element x not in s.
        """
        if not s:
            return s
        
        s = list(s)
        n = len(s)
        start = time.time()
        
        # Precompute dp for the full set
        dp_all = [1]
        for x in s:
            dp_all = update_dp(dp_all, x, n)
        
        # For each element a, precompute dp of s\{a}
        dp_without = {}
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            dp_o = [1]
            for x in others:
                dp_o = update_dp(dp_o, x, n-1)
            dp_without[a] = dp_o
        
        # Try to swap elements
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            n = len(s)
            
            # Recompute dp_all
            dp_all = [1]
            for x in s:
                dp_all = update_dp(dp_all, x, n)
            
            # Recompute dp_without
            dp_without = {}
            for i, a in enumerate(s):
                others = s[:i] + s[i+1:]
                dp_o = [1]
                for x in others:
                    dp_o = update_dp(dp_o, x, n-1)
                dp_without[a] = dp_o
            
            # Try each element for removal
            for i, a in enumerate(s):
                if (time.time() - start) > time_limit:
                    break
                
                # Try to find a replacement x
                s_without_a = s[:i] + s[i+1:]
                dp_s_without_a = dp_without[a]
                
                # Find candidates not in s
                in_set = set(s)
                
                # Try random candidates
                candidates = [x for x in range(1, 513) if x not in in_set]
                random.shuffle(candidates)
                
                for x in candidates[:200]:  # Limit to 200 candidates per removal
                    if (time.time() - start) > time_limit:
                        break
                    
                    # Check if x can be added to s_without_a
                    valid = True
                    
                    # Check 1: Is x the mean of any subset of s_without_a of size >= 2?
                    for k in range(2, len(s_without_a) + 1):
                        target = k * x
                        if target > 512 * k:
                            continue
                        if (dp_s_without_a[k] >> target) & 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Check 2: For each b in s_without_a, is b the mean of any subset of (s_without_a U {x}) \ {b} of size >= 2?
                    # Subsets containing x: T U {x} where T is subset of s_without_a\{b} of size k-1
                    # Mean condition: b = (sum(T) + x) / k => sum(T) = k*b - x
                    # Check if (k*b - x) is in dp of s_without_a\{b} at size k-1
                    for b in s_without_a:
                        # Compute dp of s_without_a\{b}
                        # This is expensive, so we approximate by checking against dp_s_without_a
                        # Actually, we need exact check. Let's compute it.
                        others_b = [y for y in s_without_a if y != b]
                        dp_others_b = [1]
                        for y in others_b:
                            dp_others_b = update_dp(dp_others_b, y, len(others_b))
                        
                        for k in range(2, len(others_b) + 2):
                            target_sum = k * b - x
                            if target_sum < 0:
                                continue
                            if target_sum > 512 * (k - 1):
                                continue
                            if k - 1 < len(dp_others_b):
                                if (dp_others_b[k-1] >> target_sum) & 1:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid:
                        # Replace a with x
                        s[i] = x
                        improved = True
                        break
                
                if improved:
                    break
        
        return s

    def construct_from_structure():
        """
        Construct a set using a structural approach:
        Use numbers that are far apart to reduce the chance of being means.
        """
        # Start with a sparse set
        base = []
        for i in range(1, 513, 11):  # Step by 11
            base.append(i)
        
        # Filter to valid set
        result = []
        for x in base:
            test = result + [x]
            if validate_set(test):
                result.append(x)
        
        return result

    best_set = []
    best_len = 0
    
    # Strategy 1: Structural construction
    structural_set = construct_from_structure()
    if len(structural_set) > best_len:
        best_set = structural_set
        best_len = len(structural_set)
    
    # Strategy 2: Greedy with different orderings
    for seed in range(50):
        if time.time() - start_time > deadline:
            break
        
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Try different orderings
        orderings = [
            candidates,
            sorted(candidates),
            sorted(candidates, reverse=True),
        ]
        
        # Add some structured orderings
        evens = [x for x in range(1, 513, 2)]
        odds = [x for x in range(2, 513, 2)]
        orderings.append(evens + odds)
        orderings.append(odds + evens)
        
        for order in orderings:
            if time.time() - start_time > deadline:
                break
            
            res = greedy_incremental(order)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)
                
                # Try local search improvement
                if best_len < MAX_SIZE:
                    improved = local_search_improve(res, time_limit=2.0)
                    if len(improved) > best_len:
                        best_set = improved
                        best_len = len(best_set)
            
            if best_len >= MAX_SIZE:
                break
    
    # Final safety check
    if best_set and not validate_set(best_set):
        # If something went wrong, try to repair by removing elements
        valid_set = []
        for x in best_set:
            test = valid_set + [x]
            if validate_set(test):
                valid_set.append(x)
        best_set = valid_set

    return best_set

# EVOLVE-BLOCK-END
