from fractions import Fraction

def construct():
    # We want to find a subset of [1, 22] with max cardinality
    # such that all reciprocal subset sums are distinct.
    
    # This is equivalent to: for any x added to S, 1/x is not a difference of two subset sums of S.
    
    candidates = list(range(1, 23))
    
    # We will try to find a set of size k, starting from 22 down to 1.
    # To be efficient, we use backtracking.
    
    def is_valid_addition(S, D, x):
        """
        Check if x can be added to S.
        S: list of current elements
        D: set of differences of subset sums of S
        x: candidate element
        """
        inv_x = Fraction(1, x)
        if inv_x in D:
            return False
        return True

    def update_diffs(D, x):
        """
        Update the set of differences when adding x.
        D_new = D union (D + 1/x) union (D - 1/x)
        """
        inv_x = Fraction(1, x)
        # Optimization: if D is large, this is expensive.
        # But for k <= 12, 2^k is small.
        # D size is roughly 2^k * 2^k = 4^k?
        # Actually, the number of distinct differences is bounded by the number of pairs of subsets.
        # For k=10, 1024 subsets. 1024^2 = 1M pairs.
        # Storing 1M Fractions might be slow/memory heavy.
        
        # Alternative: Store sums, and check differences on the fly?
        # No, checking if 1/x is a difference requires scanning all pairs.
        # O(N^2) per addition. N=2^k.
        # For k=10, N=1024. N^2 = 1M.
        # Backtracking depth 10. 10 * 1M = 10M ops.
        # With branching factor, this might be too slow for k=22.
        
        # We need a better way.
        # However, we are searching for MAX k.
        # Likely max k is small (e.g., 8-10).
        # So N is small.
        
        new_D = set(D)
        for d in D:
            new_D.add(d + inv_x)
            new_D.add(d - inv_x)
        return new_D

    def backtrack(current_S, current_D, remaining_candidates, target_size):
        if len(current_S) == target_size:
            return current_S.copy()
        
        # Pruning: if not enough candidates left
        if len(current_S) + len(remaining_candidates) < target_size:
            return None
            
        # Try each candidate
        # Order: Let's try candidates in descending order of value (22, 21, ...)
        # Why? Smaller reciprocals.
        # Or ascending?
        # Let's try ascending (1, 2, 3...) first in the list, but iterate in reverse?
        
        for i, x in enumerate(remaining_candidates):
            if is_valid_addition(current_S, current_D, x):
                new_S = current_S + [x]
                new_D = update_diffs(current_D, x)
                res = backtrack(new_S, new_D, remaining_candidates[i+1:], target_size)
                if res is not None:
                    return res
        return None

    # Try to find max size
    # We can just try k from 22 down to 1.
    # But backtracking for k=22 will fail quickly?
    # If we order candidates smartly, it might fail fast.
    
    # Let's try a few k values.
    # Based on previous analysis, size 7 is possible.
    # Let's try k=12, 11, 10, 9, 8, 7...
    
    # To be safe and fast, we can limit the search.
    # If k is large, the probability of collision is high.
    
    # Let's try k from 12 down to 1.
    # If we find one, return it.
    
    # We should also try different orderings if one fails?
    # Let's stick to one ordering: Descending value (22, 21, ..., 1).
    # This picks small reciprocals first.
    
    for k in range(12, 0, -1):
        # Sort candidates descending
        cands = sorted(candidates, reverse=True)
        res = backtrack([], {Fraction(0)}, cands, k)
        if res is not None:
            return res
            
    # Fallback
    return [1]
