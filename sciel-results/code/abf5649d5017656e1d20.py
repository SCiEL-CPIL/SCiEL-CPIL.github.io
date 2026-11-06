import time
import sys

sys.setrecursionlimit(10000)

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    containing 1,2,5,11 within [1,284].
    
    Uses Backtracking with Forward Checking and Branch-and-Bound targeting specific sizes.
    """
    prescribed = [1, 2, 5, 11]
    
    # Pre-calculate used sums for prescribed elements
    def get_initial_sums(arr):
        sums = set()
        for i, x in enumerate(arr):
            for y in arr[i:]:
                sums.add(x + y)
        return sums

    initial_sums = get_initial_sums(prescribed)
    start_time = time.time()
    time_limit = 110.0
    
    # We try to find a set of size 20, then 19, etc.
    # Max possible size is bounded by sqrt(2*284) ~ 23, but practically ~20.
    # We start from the highest likely size.
    
    for target_size in range(20, 3, -1):
        if time.time() - start_time > time_limit:
            break
            
        found_set = None
        
        def backtrack(arr, used_sums, start_val):
            nonlocal found_set, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False

            if len(arr) == target_size:
                found_set = list(arr)
                return True

            # Pruning: Can we even fit the remaining elements?
            # We need (target_size - len(arr)) more elements.
            # Let m = target_size - len(arr)
            m = target_size - len(arr)
            
            # Simple bound: The last element must be at most 284.
            # The minimum span for m new elements in a Sidon set is roughly m*(m-1)/2 + 1?
            # Actually, if we append m elements to a set ending at 'last', 
            # the new elements must be > last.
            # The tightest packing for m new marks is a Golomb ruler of m marks.
            # The length of the shortest Golomb ruler with m marks is L_min(m).
            # If last + L_min(m) > 284, we can prune.
            # Approximation for L_min(m): m*(m-1)/2 is a lower bound on differences, 
            # but for sums it's similar.
            # Let's use a conservative lower bound: last + m*(m+1)//2 > 284
            # This might be too aggressive or too weak.
            # A safer bound: The elements must be distinct integers.
            # So we need at least m integers in [start_val, 284].
            if (284 - start_val + 1) < m:
                return False
            
            # Stronger bound: 
            # If we add m elements, the largest sum will be at least arr[-1] + (arr[-1] + m*(m+1)//2)?
            # No, let's stick to the count of available integers first.
            
            # Iterate through candidates
            for x in range(start_val, 285):
                # Check validity
                valid = True
                new_sums_to_add = set()
                for y in arr:
                    s = x + y
                    if s in used_sums:
                        valid = False
                        break
                    new_sums_to_add.add(s)
                
                if not valid:
                    continue
                
                # Check 2x
                if 2 * x in used_sums:
                    continue
                new_sums_to_add.add(2 * x)
                
                # Check if new_sums_to_add has internal conflicts?
                # x+y for y in arr are distinct if y are distinct.
                # 2x is distinct from x+y if x != y (which is true since x > max(arr)).
                # So internal consistency of new sums is guaranteed.
                
                # Recurse
                next_arr = arr + [x]
                next_used = used_sums | new_sums_to_add
                
                if backtrack(next_arr, next_used, x + 1):
                    return True

            return False

        # Start backtracking for this target size
        # We must include prescribed elements.
        # The search starts from the smallest value > max(prescribed) = 11.
        # But we can also permute? No, Sidon sets are usually considered as sets, 
        # but the "pairwise sums" constraint is symmetric.
        # However, the problem asks for "distinct integers... containing 1,2,5,11".
        # The order in the list doesn't matter for the property, but for backtracking,
        # we assume we are building the set in increasing order to avoid permutations.
        # Since 1,2,5,11 are fixed, we just add elements > 11.
        
        if backtrack(prescribed, initial_sums, 12):
            return found_set
            
    # If no target size found, return the best found so far?
    # The loop above breaks if found. If it finishes without finding, 
    # it means we couldn't find size 20, 19, etc.
    # But backtrack returns False if it fails.
    # We should keep the best set found in the *last* successful target?
    # Actually, if target 20 fails, we try 19. If 19 succeeds, we return it.
    # If all fail, we should return the prescribed set? Or a larger one found during failed attempts?
    # The current code only returns if `found_set` is set.
    # If we exhaust all targets and find nothing (shouldn't happen for size 4+), 
    # we might return an empty list or the prescribed set.
    
    # To be safe, if the loop completes without returning, 
    # it implies we didn't find a set of size > 3? 
    # But we start with size 4. 
    # The backtracking for target_size=4 would succeed immediately?
    # No, `len(arr) == target_size` check is at the top.
    # If target_size is 4, and arr is size 4, it returns True immediately.
    # So it will at least return the prescribed set if nothing else is found.
    
    return prescribed
