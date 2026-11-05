import sys
from fractions import Fraction

def is_valid_subset(S):
    # S is a list of ints
    # Check if any x in S has 1/x = sum(1/y for y in T) for T subset of S\{x}
    # This is slow. We need incremental check or precomputed constraints.
    pass

def construct():
    # Use backtracking to find max cardinality subset
    # We can use a recursive function that builds the set
    # To speed up, we can precompute all "bad" triples/tuples
    # But simpler: just check validity incrementally
    
    best = []
    
    def backtrack(start, current_set, current_sums):
        nonlocal best
        # Pruning: if len(current_set) + (24 - start + 1) <= len(best):
        #     return
        
        if len(current_set) > len(best):
            best = current_set.copy()
            
        for i in range(start, 25):
            x = i
            rx = Fraction(1, x)
            
            # Check if rx is in current_sums (subset of current_set sums to rx)
            if rx in current_sums:
                continue
                
            # Check if adding x makes any existing a invalid
            valid_add = True
            for a in current_set:
                ra = Fraction(1, a)
                target = ra - rx
                if target > 0:
                    if target in current_sums:
                        valid_add = False
                        break
            
            if valid_add:
                # Add x
                new_set = current_set + [x]
                # Update sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + rx)
                
                backtrack(i + 1, new_set, new_sums)
                
                # Remove x (backtrack)
                # No need to explicitly remove from lists if we pass copies,
                # but this is expensive. 
                # Better to use in-place modification with undo, or just pass copies for N=24.
                
    backtrack(1, [], {Fraction(0, 1)})
    
    return best

