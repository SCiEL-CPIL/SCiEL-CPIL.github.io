"""Improved program for 4-AP-free sets using efficient constructions and local search."""
import time
import sys
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses base-3 Salem-Spencer construction as baseline, then local search to improve.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 512
    k = 4  # progression length
    
    def is_4ap_free(S):
        """Check if set S contains no nonconstant 4-term AP."""
        S_set = set(S)
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                    return False
        return True
    
    def base3_01_construction(n):
        """Numbers with only 0s and 1s in base 3. This is 3-AP-free, hence 4-AP-free."""
        result = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(x)
        return sorted(result)
    
    def greedy_construction(n, k=4):
        """Greedy: add x if it doesn't create a 4-AP with existing elements."""
        selected = set()
        for x in range(1, n + 1):
            valid = True
            for d in range(1, (x - 1) // (k - 1) + 1):
                # Check all positions x could be in a 4-AP
                # Position 4 (last): a, a+d, a+2d, a+3d=x -> a=x-3d
                a = x - 3 * d
                if a >= 1 and a in selected and (a + d) in selected and (a + 2*d) in selected:
                    valid = False
                    break
                # Position 3: a, a+d, a+2d=x, a+3d -> a=x-2d
                a = x - 2 * d
                if a >= 1 and a in selected and (a + d) in selected and (a + 3*d) in selected:
                    valid = False
                    break
                # Position 2: a, a+d=x, a+2d, a+3d -> a=x-d
                a = x - d
                if a >= 1 and a in selected and (a + 2*d) in selected and (a + 3*d) in selected:
                    valid = False
                    break
                # Position 1: a=x, a+d, a+2d, a+3d
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return sorted(selected)
    
    def local_search(S, n, k=4, max_iterations=50000, time_limit=60):
        """
        Local search: try to add elements or swap to improve size.
        Uses efficient conflict checking.
        """
        S = set(S)
        start = time.time()
        improved = True
        iteration = 0
        
        while improved and time.time() - start < time_limit and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try adding each element not in S
            for x in range(1, n + 1):
                if time.time() - start > time_limit:
                    return sorted(S)
                if x in S:
                    continue
                
                # Check if adding x creates a 4-AP
                conflicts = 0
                for d in range(1, (n - x) // 3 + 1):
                    # x as position 1: x, x+d, x+2d, x+3d
                    if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                        conflicts += 1
                        break
                for d in range(1, x // 3 + 1):
                    # x as position 4: x-3d, x-2d, x-d, x
                    if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                        conflicts += 1
                        break
                for d in range(1, min(x - 1, n - x) // 2 + 1):
                    # x as position 2: x-d, x, x+d, x+2d
                    if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                        conflicts += 1
                        break
                    # x as position 3: x-2d, x-d, x, x+d
                    if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                        conflicts += 1
                        break
                
                if conflicts == 0:
                    S.add(x)
                    improved = True
            
            # Try removing one element to allow adding multiple others
            # This is more complex; for now just try simple additions
            if not improved and iteration < max_iterations // 2:
                # Try a swap: remove one element, see if we can add two
                for x_remove in list(S):
                    if time.time() - start > time_limit:
                        return sorted(S)
                    S_temp = S - {x_remove}
                    added = 0
                    for x_add in range(1, n + 1):
                        if x_add in S_temp or x_add == x_remove:
                            continue
                        # Quick check
                        valid = True
                        for d in range(1, (n - x_add) // 3 + 1):
                            if (x_add + d) in S_temp and (x_add + 2*d) in S_temp and (x_add + 3*d) in S_temp:
                                valid = False
                                break
                        if valid:
                            for d in range(1, x_add // 3 + 1):
                                if (x_add - 3*d) in S_temp and (x_add - 2*d) in S_temp and (x_add - d) in S_temp:
                                    valid = False
                                    break
                        if valid:
                            S_temp.add(x_add)
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2:
                        S = S_temp
                        improved = True
                        break
        
        return sorted(S)
    
    # Start with base-3 construction
    b3 = base3_01_construction(n)
    best_set = b3
    best_size = len(b3)
    
    # Also try greedy
    greedy_set = greedy_construction(n)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Try local search on the best set
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        time_remaining = deadline - 20 - elapsed
        searched = local_search(best_set, n, time_limit=min(time_remaining, 60))
        if len(searched) > best_size:
            best_set = searched
            best_size = len(searched)
    
    # Try randomized approaches if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        random.seed(42)
        for trial in range(20):
            if time.time() - start_time > deadline - 10:
                break
            S = set()
            order = list(range(1, n + 1))
            random.shuffle(order)
            for x in order:
                if time.time() - start_time > deadline - 10:
                    break
                valid = True
                for d in range(1, (n - x) // 3 + 1):
                    if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                        valid = False
                        break
                if valid:
                    for d in range(1, x // 3 + 1):
                        if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                            valid = False
                            break
                if valid:
                    for d in range(1, min(x - 1, n - x) // 2 + 1):
                        if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                            valid = False
                            break
                        if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                            valid = False
                            break
                if valid:
                    S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_set = sorted(S)
    
    # Final verification
    if not is_4ap_free(best_set):
        # Fallback
        best_set = greedy_construction(n)
    
    return best_set
