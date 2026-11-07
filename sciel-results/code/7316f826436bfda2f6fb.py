import time
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    
    # Focus on range where divisibility is hardest
    # If a > 256/3 ≈ 85.3, then 2a > 170 > any single element
    # So no single element can divide another, and no single can divide a pair sum
    # unless pair sum >= 2*single, but pair sums are > 170 and singles <= 256
    # Actually: single s divides pair p+s' if p+s' = k*s for k>=2
    # If s > 128, then 2s > 256 >= any element, so 2s can't equal any pair sum
    # Wait: pair sum can be up to 255+256=511, so 2*228=456 could be a pair sum
    
    # Let's be more careful. Use numbers > 128 to be safe for singles vs pairs
    # Then check pair-pair and higher interactions
    
    candidates = list(range(129, 257))  # 128 candidates
    
    def is_valid_set(S):
        """Check if all non-empty subset sums are pairwise divisibility-incomparable."""
        if not S:
            return True
        n = len(S)
        sums = set()
        # Generate all subset sums
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        
        # Check no two sums divide each other
        sorted_sums = sorted(sums)
        for i, a in enumerate(sorted_sums):
            for b in sorted_sums[i+1:]:
                if b % a == 0:
                    return False
                if b > 2 * a:  # early exit: if b > 2a and b%a != 0, 
                    # actually we still need to check, but if b is much larger
                    # it's less likely to be a multiple
                    pass
        return True
    
    # Greedy with backtracking
    # Sort candidates in descending order (larger numbers first, harder to divide)
    candidates_sorted = sorted(candidates, reverse=True)
    
    best_set = []
    
    def backtrack(start, current, remaining_time):
        nonlocal best_set
        if time.time() > remaining_time:
            return
        
        if len(current) > len(best_set):
            best_set = current[:]
        
        # Try to add more elements
        for i in range(start, len(candidates_sorted)):
            if time.time() > remaining_time:
                return
            
            candidate = candidates_sorted[i]
            new_current = current + [candidate]
            
            # Quick check: verify new set is valid
            if is_valid_set(new_current):
                backtrack(i + 1, new_current, remaining_time)
    
    # Start greedy from largest
    backtrack(0, [], deadline)
    
    # Try different starting points and orderings
    # Also try ascending
    candidates_asc = sorted(candidates)
    
    def backtrack_asc(start, current, remaining_time):
        nonlocal best_set
        if time.time() > remaining_time:
            return
        
        if len(current) > len(best_set):
            best_set = current[:]
        
        for i in range(start, len(candidates_asc)):
            if time.time() > remaining_time:
                return
            
            candidate = candidates_asc[i]
            new_current = current + [candidate]
            
            if is_valid_set(new_current):
                backtrack_asc(i + 1, new_current, remaining_time)
    
    if time.time() < deadline - 5:
        backtrack_asc(0, [], deadline)
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result
