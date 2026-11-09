import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    def get_all_nonempty_subset_sums(elements):
        """Get all nonempty subset sums, returning a set (duplicates removed)."""
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums
    
    def is_valid_set(elements):
        """Check if all nonempty subset sums are pairwise non-dividing and unique."""
        sums = get_all_nonempty_subset_sums(elements)
        # Check no two distinct sums divide each other
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % sorted_sums[i] == 0:
                    return False
        return True
    
    def try_add_element(current_set, candidate):
        """Try adding candidate to current_set and check validity."""
        new_set = current_set + [candidate]
        return is_valid_set(new_set)
    
    # Candidate pool: prefer larger numbers to reduce divisibility conflicts
    # Use numbers that are "spread out" - avoid small numbers that create many divisors
    candidates = list(range(256, 0, -1))  # descending order
    
    best_set = []
    
    # Greedy with backtracking
    def greedy_build(remaining_candidates, current_set, depth):
        nonlocal best_set
        if time.time() > deadline:
            return
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        
        if not remaining_candidates:
            return
        
        for i, cand in enumerate(remaining_candidates):
            if time.time() > deadline:
                return
            if try_add_element(current_set, cand):
                new_remaining = remaining_candidates[:i] + remaining_candidates[i+1:]
                greedy_build(new_remaining, current_set + [cand], depth + 1)
            # Don't break - try all candidates at this level for better search
            # But this is exponential... need to be smarter
    
    # Smarter approach: greedy with limited backtracking
    # First pass: pure greedy
    current = []
    remaining = candidates[:]
    for cand in remaining[:]:
        if time.time() > deadline:
            break
        if try_add_element(current, cand):
            current.append(cand)
            remaining.remove(cand)
    
    best_set = list(current)
    
    # Second pass: try to improve by swapping or adding
    # Try different starting points
    for start_idx in range(0, min(20, len(candidates))):
        if time.time() > deadline:
            break
        current = []
        remaining = candidates[:]
        # Start with candidates[start_idx]
        if try_add_element(current, candidates[start_idx]):
            current.append(candidates[start_idx])
            remaining.remove(candidates[start_idx])
        
        for cand in remaining[:]:
            if time.time() > deadline:
                break
            if try_add_element(current, cand):
                current.append(cand)
                remaining.remove(cand)
        
        if len(current) > len(best_set):
            best_set = list(current)
    
    # Third pass: try to extend best_set by trying all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand not in best_set:
            if try_add_element(best_set, cand):
                best_set.append(cand)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
