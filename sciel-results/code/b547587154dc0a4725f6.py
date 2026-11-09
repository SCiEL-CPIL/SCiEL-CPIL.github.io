import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85
    best = []
    
    # Try multiple randomized greedy orders
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Create a randomized order of candidates 1..512
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if len(selected) >= 48:
                break
                
            # Check if adding x keeps the set valid
            # An element a is invalid if it is the mean of a subset of >=2 OTHER elements
            # i.e., sum(subset) == len(subset) * a
            # This is equivalent to: sum(subset - [a]) == 0 is not right.
            # Let S be the subset of OTHER elements. sum(S) == |S| * a.
            # Let S' = S - {a} is not applicable since a is not in S.
            # So we check: does there exist a subset S of (selected + [x]) \ {a} such that sum(S) == |S| * a?
            
            # Optimization: We only need to check if the NEW element x creates a violation,
            # or if x violates the condition for existing elements?
            # Actually, adding x can only create new subsets.
            # 1. Can x be the mean of a subset of OTHER elements?
            #    Subset S from selected. sum(S) == |S| * x.
            # 2. Can an existing element a be the mean of a subset involving x?
            #    Subset S from selected + [x] \ {a}.
            #    If x not in S, it was already checked.
            #    So we only need to check subsets S that INCLUDE x.
            #    Let S = {x} union T, where T is a subset of selected.
            #    sum({x} union T) == |{x} union T| * a
            #    x + sum(T) == (1 + |T|) * a
            #    sum(T) - |T| * a == a - x
            #    Let T' = {t - a for t in T}. sum(T') == a - x.
            #    So we need to check if there is a subset T of selected such that sum(t - a for t in T) == a - x.
            #    Note: T can be empty? If T is empty, sum=0, so a-x=0 => a=x, but x is new and distinct.
            #    So T must be non-empty? The problem says "subset of at least two OTHER elements".
            #    So |S| >= 2. Since x is in S, |T| >= 1.
            
            valid = True
            
            # Check 1: Is x the mean of a subset of selected?
            # sum(T) == |T| * x, T subset of selected, |T| >= 2.
            # This is equivalent to sum(t - x for t in T) == 0.
            # We can use a DP or set of possible sums of (t-x).
            # Since n=512 and size<=48, the range of sums is manageable?
            # Max sum of (t-x) is roughly 48*512.
            # But we can be smarter.
            
            # Let's just use a simple DP for subset sums of differences.
            # For Check 1:
            # We want to know if there exists T subset of selected, |T|>=2, such that sum(T) == |T|*x.
            # This is sum(t - x for t in T) == 0.
            # Let's compute all possible sums of (t-x) for subsets of selected.
            # But we also need to track size? No, just sum.
            # Wait, if sum(t-x for t in T) == 0, it doesn't guarantee |T|>=2.
            # If |T|=1, t-x=0 => t=x, impossible.
            # If |T|=0, sum=0, but |T|<2.
            # So if we find sum=0 with |T|>=2, it's invalid.
            
            # For Check 2:
            # For each a in selected:
            # We want to know if there exists T subset of selected, |T|>=1, such that sum(t - a for t in T) == a - x.
            # Note: T cannot contain a? No, T is subset of selected. a is in selected.
            # The subset of OTHER elements for a is (selected + [x]) \ {a}.
            # This set is (selected \ {a}) U {x}.
            # So T is a subset of (selected \ {a}).
            # So T cannot contain a.
            
            # Algorithm:
            # 1. For Check 1 (x is mean of others):
            #    Compute possible sums of (t-x) for subsets of selected.
            #    If 0 is a possible sum with size >= 2, invalid.
            #    Actually, we can just compute the set of all possible sums.
            #    Since we only care if 0 is achievable, and we know |T|>=2 is required.
            #    We can do DP: dp[s] = max size of subset achieving sum s? Or just boolean?
            #    If boolean, we might get 0 from empty set or size 1.
            #    We need to distinguish.
            #    Let's use a set of (sum, size) pairs? Or two sets: sums with size 0, 1, >=2?
            #    Simpler: Just iterate all subsets? No, 2^48 is too big.
            #    Use DP.
            #    Range of sums: t in [1,512], x in [1,512]. t-x in [-511, 511].
            #    Sum range: [-48*511, 48*511] approx [-24528, 24528].
            #    This is small enough for a bitset or array.
            
            # Let's implement a helper to check if a target sum is achievable by a subset of a list with size constraints.
            
            def is_mean_violation(element, others_list):
                # Check if element is mean of a subset of others_list with size >= 2
                # sum(subset) == len(subset) * element
                # sum(t - element for t in subset) == 0
                # subset is from others_list, size >= 2
                
                # DP for sums of (t - element)
                # We need to know if sum 0 is achievable with size >= 2.
                # Let's track min and max size for each sum?
                # Or just track if sum 0 is achievable with size >= 2.
                
                # Initialize dp: dict sum -> set of sizes?
                # Or since we only care about 0, maybe we can prune?
                
                # Let's use a dictionary: sum_val -> bitmask of possible sizes?
                # Max size 48. 2^48 is too big for bitmask in Python int? No, Python ints are arbitrary precision.
                # But 48 bits is fine.
                
                # dp[s] is an integer where bit k is 1 if sum s is achievable with size k.
                # Initial: dp[0] = 1 (size 0)
                
                dp = {0: 1}
                
                for t in others_list:
                    val = t - element
                    new_dp = dict(dp)
                    for s, mask in dp.items():
                        new_s = s + val
                        # Shift mask by 1 to increase size by 1
                        new_mask = mask << 1
                        if new_s in new_dp:
                            new_dp[new_s] |= new_mask
                        else:
                            new_dp[new_s] = new_mask
                    dp = new_dp
                
                # Check if sum 0 is achievable with size >= 2
                if 0 in dp:
                    mask = dp[0]
                    # Check if any bit >= 2 is set
                    if mask & ~1: # ~1 is ...1110
                        return True
                return False

            # Check 1: x is mean of subset of selected
            if is_mean_violation(x, selected):
                valid = False
                break
            
            # Check 2: For each a in selected, is a mean of subset of (selected \ {a}) U {x}?
            # This reduces to: exists T subset of (selected \ {a}), |T|>=1, such that sum(t - a for t in T) == a - x.
            for a in selected:
                others_for_a = [t for t in selected if t != a]
                target = a - x
                # Check if target is achievable by subset of others_for_a with size >= 1.
                # sum(t - a for t in T) == target
                # Note: T size >= 1.
                
                dp = {0: 1} # sum 0, size 0
                for t in others_for_a:
                    val = t - a
                    new_dp = dict(dp)
                    for s, mask in dp.items():
                        new_s = s + val
                        new_mask = mask << 1
                        if new_s in new_dp:
                            new_dp[new_s] |= new_mask
                        else:
                            new_dp[new_s] = new_mask
                    dp = new_dp
                
                if target in dp:
                    mask = dp[target]
                    # Check if any bit >= 1 is set (size >= 1)
                    if mask: # Any size >= 0 is possible. But we need size >= 1.
                        # Bit 0 corresponds to size 0.
                        # If mask has any bit set other than bit 0, then size >= 1 is possible.
                        # However, if target == 0, size 0 is possible (empty set).
                        # But we need size >= 1.
                        # So we check if mask has any bit set for k >= 1.
                        if mask & ~1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
            
    return best
