import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Generate a 3-AP free set in [1, 512]
    # Greedy 3-AP free construction
    sap_set = []
    for i in range(1, 513):
        ok = True
        for j in range(len(sap_set)):
            for k in range(j+1, len(sap_set)):
                a, b = sap_set[j], sap_set[k]
                # Check if i forms a 3-AP with a, b
                # i, a, b: 2a = i+b -> i = 2a-b
                # a, i, b: 2i = a+b -> i = (a+b)/2
                # a, b, i: 2b = a+i -> i = 2b-a
                if 2*a == i + b:
                    ok = False
                    break
                if 2*i == a + b:
                    ok = False
                    break
                if 2*b == a + i:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sap_set.append(i)
            
    if len(sap_set) < 48:
        # Fallback if greedy 3-AP is too small (unlikely for 512)
        # Try random 3-AP free generation or just return what we have
        # But 3-AP free set in 512 is usually > 50
        pass

    # Shuffle to avoid bias towards small numbers
    random.seed(42)
    random.shuffle(sap_set)
    
    selected = []
    
    def is_valid_mean_free(candidate_set, target):
        """
        Check if 'target' is the mean of any subset of 'candidate_set' 
        (excluding target if present) with size >= 2.
        We assume candidate_set does not contain target.
        """
        # Shift values: v' = v - target
        # We want subset sum of v' to be 0 with count >= 2.
        # Since we avoid 3-APs, count=2 is impossible.
        # We check count >= 2 just in case, or >= 3 if we trust 3-AP freedom.
        # Let's check >= 2 to be safe.
        
        # DP for subset sums
        # Max sum range: 48 * 512 = 24576
        # Offset to handle negative sums
        offset = 25000
        max_sum = 50000
        
        # dp[k] is a bitset of sums reachable with exactly k elements
        # Using Python integers as bitsets
        # Bit i represents sum (i - offset)
        
        # Initialize
        # dp[0] has sum 0 (bit at offset)
        dp = [0] * (len(candidate_set) + 1)
        dp[0] = 1 << offset
        
        # Process each element
        for v in candidate_set:
            shifted = v - target
            # Update dp in reverse to avoid using same element twice
            for k in range(len(candidate_set) - 1, -1, -1):
                if dp[k] != 0:
                    # Shift bitset
                    if shifted >= 0:
                        new_bits = dp[k] << shifted
                    else:
                        new_bits = dp[k] >> (-shifted)
                    dp[k+1] |= new_bits
            
            # Early termination: check if sum 0 is reachable with count >= 2
            # Bit at offset represents sum 0
            for k in range(2, len(candidate_set) + 1):
                if (dp[k] >> offset) & 1:
                    return False # Invalid, mean found
        
        return True # Valid

    def is_full_valid(new_set):
        """
        Check if new_set is a valid mean-free set.
        """
        for a in new_set:
            others = [x for x in new_set if x != a]
            if not is_valid_mean_free(others, a):
                return False
        return True

    # Greedy selection
    for x in sap_set:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
            
        candidate = selected + [x]
        
        # Optimization: Only check validity for 'x' and existing elements that might be affected?
        # Actually, if 'selected' was valid, adding 'x' only creates new means involving 'x'.
        # 1. Is x a mean of subset of selected?
        # 2. Is any a in selected a mean of subset of (selected U {x}) \ {a}?
        #    This implies the subset must contain x.
        
        # Check 1: x is mean of subset of selected
        if not is_valid_mean_free(selected, x):
            continue
            
        # Check 2: Any a in selected is mean of subset involving x
        valid = True
        for a in selected:
            # Subset must contain x.
            # Sum(T) + x = k * a  => Sum(T) = k*a - x
            # T is subset of selected \ {a}
            # We need to check if there exists T subset of (selected \ {a}) and k >= 2
            # such that sum(T) = k*a - x.
            # Note: k is the size of the total subset (T U {x}), so |T| + 1 = k.
            # So |T| >= 1.
            
            # We can use a similar DP or just check subset sums.
            # Since selected is small, we can just check all subsets? No, 2^48 is too big.
            # Use DP for subset sums of (selected \ {a}).
            # Target sums: k*a - x for k in range(2, len(selected)+1)
            
            others = [y for y in selected if y != a]
            # We need to know if any subset of 'others' sums to k*a - x
            # Let's compute all possible subset sums of 'others'
            # And check if any is equal to k*a - x for some k.
            
            # DP for subset sums of 'others'
            # Range of sums: +/- 25000
            # We can use a set of reachable sums? Or bitset.
            # Bitset is faster.
            
            offset2 = 25000
            max_sum2 = 50000
            sums_bits = 1 << offset2 # sum 0
            
            for v in others:
                if v >= 0:
                    sums_bits |= sums_bits << v
                else:
                    sums_bits |= sums_bits >> (-v)
            
            # Check if any target sum is present
            # Target: k*a - x
            # k ranges from 2 to len(selected) (since x is added, max size is len(selected)+1)
            # Actually, the subset size is k. The subset is T U {x}.
            # T is subset of others. |T| can be 1 to len(others).
            # So k = |T| + 1.
            # Max k = len(others) + 1 = len(selected).
            
            found_mean = False
            for k in range(2, len(selected) + 1):
                target_sum = k * a - x
                # Check if target_sum is in sums_bits
                # target_sum must be within range
                if -offset2 <= target_sum <= offset2:
                    if (sums_bits >> (target_sum + offset2)) & 1:
                        found_mean = True
                        break
            
            if found_mean:
                valid = False
                break
        
        if valid:
            selected = candidate

    return selected
