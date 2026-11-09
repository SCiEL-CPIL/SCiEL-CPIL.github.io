import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    def get_subset_sums(elements):
        """Get all nonempty subset sums as a set."""
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums
    
    def is_valid(elements):
        """Check if the set of elements is valid."""
        if len(elements) == 0:
            return True
        sums = get_subset_sums(elements)
        if len(sums) != (1 << len(elements)) - 1:
            return False  # duplicate subset sums
        # Check no two distinct sums divide each other
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % sorted_sums[i] == 0:
                    return False
        return True
    
    def is_valid_fast(elements):
        """Faster validity check using incremental approach."""
        n = len(elements)
        if n == 0:
            return True
        # Get all subset sums
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        # Check divisibility
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            si = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                sj = sorted_sums[j]
                if sj % si == 0:
                    return False
        return True
    
    # Greedy with backtracking
    best = [31, 32, 34]
    
    def backtrack(current, candidates, depth):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) > len(best):
            best = list(current)
        
        if not candidates:
            return
        
        # Try adding each candidate
        for i, cand in enumerate(candidates):
            new_current = current + [cand]
            if is_valid_fast(new_current):
                remaining = candidates[i+1:]
                backtrack(new_current, remaining, depth + 1)
    
    # Try greedy: start with large numbers first (they're less likely to be divisors)
    # Try multiple orderings
    for order in range(3):
        if time.time() > deadline:
            break
        
        if order == 0:
            # Descending order
            candidates = list(range(256, 1, -1))
        elif order == 1:
            # Only numbers > 128, descending
            candidates = list(range(256, 128, -1))
        else:
            # Numbers > 100, descending
            candidates = list(range(256, 100, -1))
        
        current = []
        # Greedy pass first
        for cand in candidates:
            if time.time() > deadline:
                break
            new_current = current + [cand]
            if is_valid_fast(new_current):
                current = new_current
        
        if len(current) > len(best):
            best = list(current)
        
        # Try to improve with limited backtracking
        if len(current) < 10 and time.time() < deadline - 10:
            # Try swapping out one element for another
            for idx in range(len(current)):
                if time.time() > deadline:
                    break
                saved = current[idx]
                current.pop(idx)
                for cand in candidates:
                    if cand == saved or cand in current:
                        continue
                    new_current = current + [cand]
                    if is_valid_fast(new_current):
                        current = new_current
                        break
                else:
                    current.insert(idx, saved)
    
    # Ensure we return a valid set
    if not is_valid_fast(best):
        # Fallback to known valid set
        best = [31, 32, 34]
    
    # Convert to Python ints
    return [int(x) for x in best]
