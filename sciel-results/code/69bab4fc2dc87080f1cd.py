import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 192
    max_size = 48
    
    def is_valid_vectorized(S):
        """
        Check if S is valid using vectorized operations.
        S is a numpy array of shape (k,).
        Returns True if valid, False otherwise.
        """
        k = len(S)
        if k < 4:
            return True
        
        # We need to check for any a in S if there exist b, c, d in S\{a} distinct
        # such that a | (b+c+d).
        
        # For each a, the sum b+c+d must be a multiple of a.
        # Let S be sorted.
        # To speed up, we can precompute the set of S.
        S_set = set(S)
        
        # Iterate over each a in S
        for i, a in enumerate(S):
            # Candidates for b, c, d are all elements in S except a.
            # Let's get the indices of others.
            # Instead of creating a list, we can just iterate.
            # But vectorized check for a specific a:
            # We need b, c, d in S\{a}, b!=c!=d!=a, b+c+d = m*a.
            
            # Let's try to find if ANY triplet sums to a multiple of a.
            # Triples from S\{a}.
            
            # Optimization: 
            # If min_sum > a * max_possible_m, no violation.
            # If max_sum < a * min_possible_m, no violation.
            
            others = [x for x in S if x != a]
            if len(others) < 3:
                continue
                
            others_arr = np.array(others)
            others_set = set(others)
            
            min_sum = others_arr[0] + others_arr[1] + others_arr[2]
            max_sum = others_arr[-1] + others_arr[-2] + others_arr[-3]
            
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            if start_m > end_m:
                continue
                
            # For each multiple m*a in range, check if it can be formed by 3 distinct others.
            # We can precompute all pair sums of others?
            # Size of others is ~47. Pairs ~ 1000.
            # For each target, check if target - b - c is in others.
            
            # Let's create a set of others for O(1) lookup.
            # We also need to ensure b, c, d are distinct.
            
            # Iterate pairs (b, c) from others.
            # To avoid permutations, assume b < c.
            # Then d = target - b - c.
            # Check if d in others_set and d > c.
            
            # We can vectorize the pair sums.
            # Create upper triangular indices for pairs in others.
            idx = np.triu_indices(len(others), k=1)
            b_arr = others_arr[idx[0]]
            c_arr = others_arr[idx[1]]
            pair_sums = b_arr + c_arr
            
            # For each target m*a, we want to know if there exists a pair sum p
            # such that target - p is in others_set and target - p > c (where c corresponds to p).
            # This is slightly tricky to vectorize fully across all targets and pairs at once
            # because of the "d > c" and "d in set" constraints which depend on specific pairs.
            
            # However, we can check for each target individually if it's fast enough.
            # Number of targets is small (range of sums / a).
            
            for m in range(start_m, end_m + 1):
                target = m * a
                
                # We need b, c, d in others, distinct, b+c+d=target.
                # Using the precomputed pairs:
                # d = target - pair_sums
                # Valid if d > c_arr AND d in others_set.
                
                d_vals = target - pair_sums
                
                # Condition 1: d > c
                valid_mask = d_vals > c_arr
                
                # Condition 2: d in others_set
                # We can't easily vectorize "in set" with numpy directly without loops or fancy tricks.
                # But we can check if d_vals[valid_mask] are in others_set.
                # And we must ensure d is distinct from b and c.
                # Since d > c and b < c, d is automatically distinct from b and c if d != b and d != c.
                # d > c implies d != c.
                # d > c > b implies d != b.
                # So distinctness is guaranteed by d > c > b.
                
                if not np.any(valid_mask):
                    continue
                    
                candidate_ds = d_vals[valid_mask]
                
                # Check if any candidate_ds is in others_set
                # We can convert candidate_ds to a set and intersect with others_set?
                # Or loop?
                # Given the small size, a loop over candidate_ds might be okay, but let's try to be smart.
                # Actually, just checking intersection is fast.
                if set(candidate_ds).intersection(others_set):
                    return False
                    
        return True

    def find_violation(S):
        """
        Find one violation (a, b, c, d) or return None.
        """
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return None
            
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                # Iterate pairs
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            return (a, b, c, d)
        return None

    def can_add(S, x):
        """
        Check if x can be added to S.
        S is a list or set.
        """
        if x in S:
            return False
            
        # New set S' = S U {x}
        # Violations can involve:
        # 1. x as the divisor: x | (a+b+c) for a,b,c in S.
        # 2. a in S as divisor: a | (x+b+c) for b,c in S.
        
        S_list = list(S)
        others_set = set(S_list)
        
        # Case 1: x is the divisor
        # Sum of any 3 from S must not be divisible by x.
        # Precompute all triple sums of S?
        # Size of S is ~47. Triples ~ 16k.
        # If x is large, few multiples.
        
        # Let's just check if any triple sum is divisible by x.
        # We can do this efficiently.
        # S is small.
        
        # Actually, we can just create the new array and call is_valid_vectorized?
        # That might be too slow if called for every candidate.
        # Let's optimize can_add specifically.
        
        # Case 1: x divides sum of 3 from S.
        # We need to check if any sum of 3 distinct elements from S is a multiple of x.
        
        # Let's generate all triple sums of S.
        # This is O(N^3). N=47. 47^3 ~ 100k.
        # Doing this for every candidate x (up to 192) is 192 * 100k = 19M ops.
        # In Python, this is too slow.
        
        # Alternative:
        # For a fixed x, we want to know if there exist a,b,c in S such that a+b+c = k*x.
        
        # Let's use the vectorized check on the new set?
        # is_valid_vectorized(S U {x})
        # This iterates over all elements in S U {x}.
        # For each a, it checks triples.
        # If we call this for every candidate, it's expensive.
        
        # Heuristic:
        # Only check the new element x as divisor, and existing elements as divisors.
        
        # Check 1: x as divisor.
        # We need a+b+c = k*x.
        # We can precompute pair sums of S.
        # Then for each c in S, check if k*x - c is in pair_sums.
        
        # Check 2: a in S as divisor.
        # We need a | (x+b+c).
        # So x+b+c = k*a.
        # b+c = k*a - x.
        # We need pair sum from S (excluding nothing? b,c distinct from a? Yes, b,c in S\{a}? No, b,c in S.
        # Wait, the definition says "3 DISTINCT OTHER selected integers".
        # If a is the divisor, the summands are 3 distinct elements from S\{a}.
        # If we add x, the new set is S' = S U {x}.
        # For a in S, the summands are 3 distinct elements from S'\{a}.
        # These can be:
        # - 3 from S\{a} (already checked in S)
        # - 2 from S\{a} and 1 x
        # - 1 from S\{a} and 2 x (impossible, x is unique)
        # - 3 x (impossible)
        
        # So we only need to check:
        # 1. x | (a+b+c) for a,b,c in S.
        # 2. a | (x+b+c) for a in S, b,c in S\{a}.
        
        # Let's implement these checks.
        
        S_arr = np.array(S_list)
        
        # Precompute pair sums of S for Check 2?
        # Or just use sets.
        
        # Check 1: x | (a+b+c)
        # Triple sums of S.
        # Let's iterate a, b in S, and check if c = k*x - a - b is in S.
        # This is O(N^2 * M) where M is number of multiples.
        
        # Check 2: a | (x+b+c)
        # For each a in S:
        #   We need b,c in S\{a} such that x+b+c = k*a.
        #   b+c = k*a - x.
        #   We can precompute pair sums of S\{a}? That's expensive per a.
        #   Instead, precompute all pair sums of S.
        #   Let pair_sums be a set of (sum, indices) or just sums?
        #   If we just use sums, we might pick b=a or c=a.
        #   So we need to be careful.
        
        # Let's use a set of pairs (b, c) with b < c.
        # And a set of pair sums? No, we need to verify distinctness from a.
        
        # Simpler:
        # For Check 1:
        #   For each a in S:
        #     For each b in S, b > a:
        #       target_c = k*x - a - b
        #       if target_c > b and target_c in S: return False
        #   This is O(N^2 * M).
        
        # For Check 2:
        #   For each a in S:
        #     For each k:
        #       target_pair_sum = k*a - x
        #       We need b, c in S\{a} such that b+c = target_pair_sum and b != c.
        #       We can iterate b in S\{a}:
        #         c = target_pair_sum - b
        #         if c > b and c in S and c != a: return False
        #   This is O(N^2 * M).
        
        # Total O(N^2 * M) for both. N=47, M is small.
        # This should be fast enough in Python if optimized?
        # 47^2 ~ 2200. M ~ 10. 22000 ops per candidate.
        # 192 candidates * 22000 = 4M ops.
        # This is borderline for 60s in pure Python.
        
        # Let's try to vectorize Check 1 and Check 2.
        
        # Check 1: x | (a+b+c)
        # a, b, c in S.
        # Let's get all triple sums? No, too many.
        
        # Let's just try the loop approach first, it's simpler and might be fast enough with early breaks.
        
        S_set = set(S_list)
        
        # Check 1: x as divisor
        # We need a+b+c = k*x.
        # Since a,b,c <= 192, sum <= 576.
        # k*x <= 576 => k <= 576/x.
        
        max_k1 = 576 // x
        for k in range(1, max_k1 + 1):
            target = k * x
            # Find a, b, c in S such that a+b+c = target.
            # Iterate a in S
            for a in S_list:
                # b+c = target - a
                rem1 = target - a
                # Iterate b in S, b > a (to avoid duplicates/permutations)
                # Actually, we just need existence.
                for b in S_list:
                    if b <= a: continue
                    rem2 = rem1 - b
                    if rem2 <= b: continue
                    if rem2 in S_set:
                        # Found a, b, rem2. Are they distinct?
                        # a < b < rem2. Yes.
                        # Are they in S? Yes.
                        return False
        
        # Check 2: a in S as divisor, sum involves x.
        # Sum = x + b + c = k*a.
        # b, c in S\{a}.
        
        for a in S_list:
            max_k2 = (x + 192 + 191) // a # Rough upper bound
            for k in range(1, max_k2 + 1):
                target = k * a
                # b + c = target - x
                rem1 = target - x
                if rem1 <= 0: continue
                # Find b, c in S\{a} such that b+c = rem1.
                # Iterate b in S\{a}
                for b in S_list:
                    if b == a: continue
                    # c = rem1 - b
                    c = rem1 - b
                    if c <= b: continue
                    if c == a: continue
                    if c in S_set:
                        # Found b, c. Distinct from a? Yes.
                        # b != c? Yes (c > b).
                        return False
                        
        return True

    best_S = set()
    
    # Initial set: Large numbers are safer.
    # Start with top 48 numbers.
    S = set(range(192 - 48 + 1, 193))
    
    # Repair phase
    while time.time() < deadline - 10:
        if len(S) >= max_size and is_valid_vectorized(np.array(list(S))):
            if len(S) > len(best_S):
                best_S = S
            break
            
        if not is_valid_vectorized(np.array(list(S))):
            viol = find_violation(S)
            if viol:
                a, b, c, d = viol
                # Remove one of them.
                # Heuristic: Remove the one that is the divisor 'a'?
                # Or the largest one to keep the set "large"?
                # Removing 'a' removes the constraint source.
                S.remove(a)
            else:
                break
        else:
            # Valid, try to add
            if len(S) >= max_size:
                if len(S) > len(best_S):
                    best_S = S
                break
                
            candidates = [x for x in range(1, n + 1) if x not in S]
            # Sort descending to prefer large numbers
            candidates.sort(reverse=True)
            
            added = False
            for x in candidates:
                if time.time() > deadline - 10:
                    break
                if can_add(S, x):
                    S.add(x)
                    added = True
                    break
                    
            if not added:
                # Try swaps?
                # If we can't add, maybe we can swap a small element for a larger one?
                # Or just break.
                if len(S) > len(best_S):
                    best_S = S
                break
                
    # Final check
    if len(best_S) == 0:
        # Fallback
        best_S = set(range(183, 193))
        
    # Ensure best_S is valid
    if is_valid_vectorized(np.array(list(best_S))):
        return list(best_S)
    else:
        # If best_S is invalid, return a small valid set
        return list(range(183, 193))
