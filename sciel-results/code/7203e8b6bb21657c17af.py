import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None

        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)

        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def find_min_diameter_set(max_candidate=100):
        """
        Uses backtracking search to find a Sidon set with minimal maximum element.
        Iterates D from lower bound upwards.
        """
        best_set = None
        
        # Heuristic lower bound: 
        # We need 30 distinct differences. 
        # If max element is D, max difference is D. 
        # So D >= 30. 
        # Also, Sidon property restricts packing. 
        # Start from 30.
        
        for D in range(30, max_candidate + 1):
            if time.time() > deadline - 10:
                break
            
            found = None
            
            # State:
            # current_elements: sorted list of chosen elements
            # all_diffs: set of all pairwise differences generated so far
            # covered: set of differences in 1..30 that are in all_diffs
            
            # We fix 0 in the set.
            # Candidates: 1..D
            
            # Optimization: Pre-compute potential conflicts?
            # Or just check on the fly.
            
            def backtrack(current_elements, all_diffs, covered, start_idx):
                nonlocal found
                if found is not None:
                    return
                
                if time.time() > deadline - 10:
                    return
                
                # Goal check
                if len(covered) == 30:
                    found = current_elements[:]
                    return
                
                # Pruning 1: Can we cover the remaining differences?
                # Current size m = len(current_elements)
                # Remaining candidates count k = D - start_idx + 1 (indices start_idx to D)
                # Actually, candidates are from start_idx to D.
                m = len(current_elements)
                k = D - start_idx + 1
                missing = 30 - len(covered)
                
                # Max new differences if we add all k candidates:
                # Total elements would be m + k.
                # Total diffs = (m+k)(m+k-1)/2
                # Current diffs = m(m-1)/2
                # Max new = Total - Current
                if k > 0:
                    max_total_elems = m + k
                    max_total_diffs = max_total_elems * (max_total_elems - 1) // 2
                    current_diffs_count = m * (m - 1) // 2
                    max_new_diffs = max_total_diffs - current_diffs_count
                    if max_new_diffs < missing:
                        return
                else:
                    if missing > 0:
                        return

                # Try adding each candidate from start_idx to D
                for c in range(start_idx, D + 1):
                    # Check if adding c creates duplicate differences
                    ok = True
                    new_diffs = []
                    for x in current_elements:
                        d = c - x
                        if d in all_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Update state
                    new_covered = covered.copy()
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_covered.add(d)
                    
                    new_all_diffs = all_diffs.union(new_diffs)
                    new_elements = current_elements + [c]
                    
                    # Recurse
                    backtrack(new_elements, new_all_diffs, new_covered, c + 1)
                    
                    if found is not None:
                        return
            
            # Initial call
            # Start with [0]
            # Candidates start from 1
            backtrack([0], set(), set(), 1)
            
            if found is not None:
                return found
                
        return None

    def greedy_fallback(time_limit):
        """
        Simple greedy heuristic if backtracking fails or is too slow.
        """
        a = [0]
        diffs = set()
        x = 1
        while time.time() < time_limit:
            # Find smallest x not in a such that adding x keeps Sidon property
            # and helps cover missing diffs
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            
            if valid:
                # Check if this x covers any new required diff
                new_cover = False
                for y in a:
                    if x > y:
                        d = x - y
                        if 1 <= d <= 30 and d not in diffs:
                            new_cover = True
                            break
                            
                # Accept if it covers new diff or if we have fewer than 9 elements (to ensure enough room)
                if new_cover or len(a) < 9:
                    for y in a:
                        if x > y:
                            diffs.add(x - y)
                    a.append(x)
                    
                    if all(d in diffs for d in range(1, 31)):
                        return a
            
            x += 1
            if x > 1000: # Safety break
                break
                
        return a

    # Main Execution
    # 1. Try exact search for minimal diameter
    best_set = find_min_diameter_set(max_candidate=80)
    
    if best_set is None:
        # 2. Fall back to greedy if exact search fails (shouldn't happen for D<80 usually)
        print("Exact search failed or timed out, using greedy fallback", file=sys.stderr)
        best_set = greedy_fallback(deadline - 5)
        
    # Final Validation
    if best_set is None:
        # Emergency: just return a known valid set if even greedy fails?
        # A known Sidon set covering 1..30: 
        # {0, 1, 4, 10, 12, 28, 32, 45, 63} has diameter 63.
        # Let's verify quickly.
        emergency = [0, 1, 4, 10, 12, 28, 32, 45, 63]
        v, _, _ = is_valid(emergency)
        if v:
            return emergency
        else:
            # This should never happen, but just in case
            return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    return best_set

# EVOLVE-BLOCK-END
