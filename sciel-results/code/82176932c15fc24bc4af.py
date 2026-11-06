from fractions import Fraction
import itertools

def construct():
    # Start with the full set
    S = list(range(1, 25))
    
    def get_conflict(S):
        """
        Returns (x, subset_sum_elements) if 1/x is a subsum of S\{x}, else None.
        """
        S_set = set(S)
        for x in S:
            R = [y for y in S if y != x]
            # Check if 1/x is a subset sum of R
            # Use DP with set of Fractions
            sums = {Fraction(0, 1)}
            for y in R:
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, y))
                sums = new_sums
            if Fraction(1, x) in sums:
                # We found a conflict, but we don't have the specific subset Y.
                # We just need to know a conflict exists to remove something.
                # To remove the right thing, we can just return x.
                return x
        return None

    # Simple local search: remove elements causing conflicts
    # We want to maximize size, so we should try to keep as many as possible.
    # Removing small elements is usually better because they are more likely to be part of conflicts.
    
    # However, the above get_conflict is slow if it runs to completion for many x.
    # Let's optimize the check.
    
    def is_valid(S):
        S_set = set(S)
        for x in S:
            R = [y for y in S if y != x]
            sums = {Fraction(0, 1)}
            for y in R:
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, y))
                sums = new_sums
            if Fraction(1, x) in sums:
                return False
        return True

    # Greedy removal
    # Sort S in ascending order? We want to keep large numbers.
    # So we try to remove small numbers first.
    
    # Start with full set
    current = list(range(1, 25))
    
    # We will iteratively check and remove
    # To be safe and fast, we can just check the full set, and if invalid, 
    # remove the smallest element and repeat.
    # This might be suboptimal but fast.
    
    # Better: 
    # Try to find a valid subset by removing elements.
    
    # Let's try a heuristic: 
    # Keep all numbers >= 13.
    # Try to add 12, 11, etc.
    
    S = list(range(13, 25))
    
    # Try to add smaller numbers
    for x in range(12, 0, -1):
        candidate = S + [x]
        if is_valid(candidate):
            S = candidate
            
    return S
