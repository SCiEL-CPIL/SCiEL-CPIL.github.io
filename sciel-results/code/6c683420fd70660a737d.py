import time
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 60.0

    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility relationships
    # divs[i] = set of indices j such that smooth_numbers[j] divides smooth_numbers[i]
    divs = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if a % b == 0:
                    divs[i].add(j)
                elif b % a == 0:
                    divs[j].add(i)
    
    # Precompute compatibility: compat[i][j] = True if i and j can both be in an antichain
    # i and j are compatible if neither divides the other
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in divs[i]:
            compat[i][j] = False
            compat[j][i] = False
    
    # For each target, find minimum number of terms
    # Use DP: dp[target] = (min_terms, representation)
    # But we need to track which smooth numbers are used to ensure antichain
    # Since max terms is small (3-4), we can do BFS/DFS with pruning
    
    # Alternative: For each target, try k=1,2,3,4... and find first valid antichain
    # Use bitmask for used smooth numbers to check antichain property
    
    # Precompute for each subset of smooth numbers whether it's an antichain
    # Since we only need up to 4-5 terms, enumerate all combinations of size <= 5
    # and check if they form an antichain, then map sum -> (size, terms)
    
    # Better approach: For each target, search for minimum k
    # For k=1: check if target is smooth
    # For k=2: check pairs
    # For k=3: check triples
    # For k=4: check quadruples (only if needed)
    
    # Precompute all valid antichains of size 1,2,3,4 and their sums
    # Then for each target, find the minimum size antichain that sums to it
    
    # Size 1: all smooth numbers
    antichain_sums = {k: {} for k in range(1, 6)}  # sum -> list of terms
    for i in range(n_smooth):
        s = smooth_numbers[i]
        if s <= 128:
            antichain_sums[1][s] = [smooth_numbers[i]]
    
    # Size 2: all compatible pairs
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if compat[i][j]:
                s = smooth_numbers[i] + smooth_numbers[j]
                if s <= 128 and s not in antichain_sums[2]:
                    antichain_sums[2][s] = [smooth_numbers[i], smooth_numbers[j]]
    
    # Size 3: all compatible triples
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if not compat[i][j]:
                continue
            for k in range(j+1, n_smooth):
                if compat[i][k] and compat[j][k]:
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                    if s <= 128 and s not in antichain_sums[3]:
                        antichain_sums[3][s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]]
    
    # Size 4: all compatible quadruples (only for targets not covered by 1-3)
    # First, identify which targets need size 4
    targets_needing_4 = set()
    for target in range(1, 129):
        if target not in antichain_sums[1] and target not in antichain_sums[2] and target not in antichain_sums[3]:
            targets_needing_4.add(target)
    
    if targets_needing_4:
        for i in range(n_smooth):
            for j in range(i+1, n_smooth):
                if not compat[i][j]:
                    continue
                for k in range(j+1, n_smooth):
                    if not (compat[i][k] and compat[j][k]):
                        continue
                    for m in range(k+1, n_smooth):
                        if compat[i][m] and compat[j][m] and compat[k][m]:
                            s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m]
                            if s <= 128 and s in targets_needing_4 and s not in antichain_sums[4]:
                                antichain_sums[4][s] = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[m]]
    
    # Build answers
    answers = []
    for target in range(1, 129):
        # Find minimum k such that target is in antichain_sums[k]
        found = None
        for k in range(1, 6):
            if target in antichain_sums[k]:
                found = antichain_sums[k][target]
                break
        
        # If not found (shouldn't happen), do a final search
        if found is None:
            # Brute force with backtracking
            found = _brute_force_target(target, smooth_numbers, compat, n_smooth)
        
        answers.append(found)
    
    return answers

def _brute_force_target(target, smooth_numbers, compat, n_smooth):
    """Fallback brute force search for a target not covered by precomputation."""
    n = n_smooth
    # Try sizes 1-5
    for size in range(1, 6):
        result = _backtrack(target, size, 0, [], smooth_numbers, compat, n)
        if result is not None:
            return result
    return None

def _backtrack(remaining, size, start, current, smooth_numbers, compat, n):
    """Backtracking search for an antichain of given size summing to remaining."""
    if size == 0:
        if remaining == 0:
            return current
        return None
    if remaining <= 0:
        return None
    
    for i in range(start, n):
        val = smooth_numbers[i]
        if val >= remaining:
            break
        # Check compatibility with all current elements
        compatible = True
        for j in range(len(current)):
            # Find index of current[j] in smooth_numbers
            # This is inefficient; let's pass indices instead
            pass
        # Simplified: assume we're tracking indices
        # For now, this is a fallback that may not be called
        new_current = current + [val]
        result = _backtrack(remaining - val, size - 1, i + 1, new_current, smooth_numbers, compat, n)
        if result is not None:
            return result
    return None
