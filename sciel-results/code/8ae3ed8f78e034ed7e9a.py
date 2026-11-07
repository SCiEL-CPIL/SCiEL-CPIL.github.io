"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
from functools import reduce

def construct():
    """
    Construct a set of 1..30 distinct integers in 1..256 such that no two distinct
    nonempty subset sums divide one another.
    
    Strategy: Greedy construction with backtracking/heuristic search.
    We try to build a large set by adding elements one by one, checking the divisibility
    constraint. To improve efficiency, we prioritize elements that are 'safe' (less likely
    to cause divisibility conflicts) and use a randomized restart approach.
    
    Key insight: If we pick numbers that are all close to 256/2 = 128 or higher, their
    subset sums will be large, and it's harder for one sum to divide another. Specifically,
    if all subset sums are in a range where no sum is a multiple of another, we're good.
    
    A stronger approach: Use a greedy algorithm that tries to add elements in decreasing
    order of 'safety'. We can also use a simple local search: start with a random set,
    then try to replace elements to increase size or fix conflicts.
    
    Given the time constraint (60s), we'll use a randomized greedy approach with multiple
    restarts.
    """
    
    def is_valid_subset_sums(nums):
        """Check if all nonempty subset sums are distinct and no two divide each other."""
        if not nums:
            return True
        
        # Generate all subset sums
        n = len(nums)
        subset_sums = set()
        # Use bit masking for up to ~20 elements, but we want up to 30.
        # 2^30 is about 1 billion, which is too slow in Python.
        # We need a more efficient way or a different strategy.
        # For n=30, we can't enumerate all 2^30 subsets in Python within 60s easily.
        # We need to check the condition incrementally or use a different method.
        
        # Alternative: Check pairwise divisibility of sums as we build.
        # But we need to ensure NO two sums divide each other.
        # Let's use a set of sums and check divisibility against existing sums.
        
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += nums[i]
            # Check if s divides any existing sum or is divided by any existing sum
            # This is O(|sums|) per subset, and |sums| can be large.
            # This is too slow for n=30.
            pass
        
        # Since enumeration is too slow for n=30, we use a heuristic:
        # If all numbers are greater than 128, then the smallest sum is > 128.
        # The largest sum is < 30*256 = 7680.
        # If we ensure that for any two sums a, b, a does not divide b.
        # A sufficient condition: all sums are in the range (M, 2M] for some M.
        # But subset sums can vary widely.
        
        # Let's try a different approach: use a set of numbers that are all in [129, 256].
        # Then any subset sum is at least 129.
        # If we have two sums a and b with a < b, we need a not to divide b.
        # If b < 2a, then a cannot divide b (since the next multiple is 2a > b).
        # So if we ensure that for any two distinct subset sums a < b, we have b < 2a,
        # then no sum divides another.
        # The maximum sum is S_max, the minimum is S_min.
        # We need S_max < 2 * S_min? No, that's not sufficient for all pairs.
        # We need that for any a, b in the set of sums, if a < b then b < 2a.
        # This is equivalent to: the ratio of max sum to min sum is < 2.
        # S_max / S_min < 2.
        # S_min is the smallest element (if we pick positive ints).
        # S_max is the sum of all elements.
        # So we need sum(nums) < 2 * min(nums).
        # If we pick numbers all close to 256, say in [200, 256], and k numbers,
        # sum is ~228k, min is ~200. 228k < 400 => k < 1.75. Not possible.
        
        # So that condition is too strong.
        
        # Let's go back to incremental checking.
        # We can maintain the set of all subset sums. When adding a new element x,
        # the new sums are old_sums U (old_sums + x) U {x}.
        # We need to check that no new sum divides an old sum or vice versa,
        # and no two new sums divide each other.
        # This is still expensive if the number of sums is large.
        # But for k up to 30, the number of distinct sums can be up to 2^30, which is huge.
        # However, in practice, with random numbers, the sums are spread out, and
        # the number of distinct sums is much smaller? No, it can still be large.
        
        # Given the constraints, let's try a simpler heuristic:
        # Pick numbers that are all in [129, 256] and check pairwise divisibility
        # of the sums using a sampled approach or a bounded check.
        
        # Actually, let's try to construct a set where all elements are primes near 256.
        # Subset sums of primes are unlikely to divide each other.
        
        # Let's implement a greedy algorithm that adds elements one by one,
        # and for each new element, checks if the new subset sums violate the condition
        # against the existing subset sums. To keep it fast, we limit the check to
        # a reasonable number of sums or use a probabilistic check.
        
        # For now, let's return a known valid small set and try to extend it.
        # The seed returns [31, 32, 34]. Let's verify this.
        # Sums: 31, 32, 34, 63, 65, 66, 97.
        # 31 divides 93? No. 32 divides 64? No. 34 divides 68? No.
        # 63 divides 126? No. 65 divides 130? No. 66 divides 132? No.
        # 97 is prime, next multiple 194 > 97+34=131? No, 97+34=131, 97+32=129, 97+31=128.
        # Max sum is 97. So 97 doesn't divide any other sum.
        # Does 31 divide 63? No. 31 divide 65? No. 31 divide 66? No.
        # Does 32 divide 63? No. 32 divide 65? No. 32 divide 66? No.
        # Does 34 divide 63? No. 34 divide 65? No. 34 divide 66? No.
        # So [31, 32, 34] is valid. Size 3.
        
        # Let's try to find a larger set using a greedy approach with a fast check.
        # We'll use a set of current sums. When adding x, new sums are {s+x for s in sums} U {x}.
        # We check if any new sum y has a relationship with any existing sum z:
        #   y % z == 0 or z % y == 0.
        # If the number of sums is too large, we abort and try a different x.
        
        max_size = 0
        best_set = []
        
        for _ in range(20):  # 20 restarts
            random.seed(_ * 12345 + 67890)
            candidates = list(range(129, 257))
            random.shuffle(candidates)
            
            current_set = []
            current_sums = set()
            
            for x in candidates:
                # Try to add x
                new_sums = set(current_sums)
                new_sums.add(x)
                for s in current_sums:
                    new_sums.add(s + x)
                
                # Check divisibility
                valid = True
                # Check each new sum against all existing sums
                # To speed up, we only check if the new sum is a multiple of an old sum
                # or an old sum is a multiple of the new sum.
                # We'll check all pairs between new_sums and current_sums, and within new_sums.
                # This is O(|new_sums| * |current_sums|) which can be large.
                # Let's limit the number of sums we track. If |current_sums| > 10000, we skip.
                if len(current_sums) > 5000:
                    continue
                
                # Check new sums against old sums
                for ns in new_sums:
                    if ns in current_sums:
                        # Duplicate sum, which is allowed? No, "Equal sum values are counted once"
                        # but "Two different subsets with the same sum still make the set invalid"
                        # So if ns is in current_sums, it means two different subsets have the same sum.
                        # This is invalid.
                        valid = False
                        break
                    if valid:
                        for cs in current_sums:
                            if ns % cs == 0 or cs % ns == 0:
                                valid = False
                                break
                    if not valid:
                        break
                
                # Check within new_sums (excluding those already in current_sums)
                if valid:
                    new_only = new_sums - current_sums
                    # Check pairs within new_only
                    new_list = list(new_only)
                    for i in range(len(new_list)):
                        if not valid:
                            break
                        for j in range(i + 1, len(new_list)):
                            a = new_list[i]
                            b = new_list[j]
                            if a % b == 0 or b % a == 0:
                                valid = False
                                break
                
                if valid:
                    current_set.append(x)
                    current_sums = new_sums
                    if len(current_set) > max_size:
                        max_size = len(current_set)
                        best_set = current_set.copy()
                    if len(current_set) >= 30:
                        break
            
            if len(current_set) >= 30:
                break
        
        if len(best_set) > 3:
            return best_set
        else:
            return [31, 32, 34]
# EVOLVE-BLOCK-END
