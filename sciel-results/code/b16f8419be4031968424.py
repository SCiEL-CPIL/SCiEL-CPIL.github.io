"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def check_valid(arr):
        """Check if all subset sums are distinct."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return len(sums) == 4096
    
    def get_forbidden(sums, min_val):
        """Get the set of forbidden values (differences between subset sums) >= min_val."""
        forbidden = set()
        s_list = list(sums)
        n = len(s_list)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(s_list[i] - s_list[j])
                if diff >= min_val:
                    forbidden.add(diff)
        return forbidden
    
    def greedy_with_forbidden(deadline, start_arr=None):
        """Greedy construction using forbidden set (differences)."""
        if start_arr is None:
            arr = [1]
        else:
            arr = list(start_arr)
        
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            sums = sums | new_sums
        
        for step in range(len(arr), 12):
            if time.time() > deadline:
                return None
            
            min_val = arr[-1] + 1 if arr else 1
            forbidden = get_forbidden(sums, min_val)
            
            found = False
            x = min_val
            while x <= 2048:
                if time.time() > deadline:
                    return None
                
                if x not in forbidden:
                    # Also check that x is not already in arr
                    if x not in arr:
                        new_sums = {s + x for s in sums}
                        # Verify no collision
                        if not (new_sums & sums):
                            sums = sums | new_sums
                            arr.append(x)
                            found = True
                            break
                x += 1
            
            if not found:
                return None
        
        return arr
    
    # Try greedy with forbidden set
    result = greedy_with_forbidden(deadline)
    
    if result is not None:
        if len(result) == 12 and len(set(result)) == 12 and all(x > 0 for x in result) and max(result) <= 1048576:
            if check_valid(result):
                return result
    
    # Fallback: powers of 2
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
