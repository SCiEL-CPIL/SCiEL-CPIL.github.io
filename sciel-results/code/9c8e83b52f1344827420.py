"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    
    def is_valid_set(S):
        """Check if the set S satisfies the property."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        # Pairwise: a < x => a does not divide 2x
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                x = S_sorted[j]
                if (2 * x) % a == 0:
                    return False
        # Ternary: a < b < c => a does not divide b+c
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for l in range(j+1, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True

    # Greedy Initialization: Start with large numbers
    # Large numbers are less likely to be divisors 'a'
    S = set()
    for num in range(n, 0, -1):
        # Check if adding num is valid
        # We can do a fast check against current S
        valid = True
        S_list = sorted(S)
        # Check num as 'a' (smallest)
        # If num is added, it must be smaller than existing b,c? 
        # No, we are adding to set. We need to check all roles.
        
        # Fast check:
        # 1. num as a: For all b,c in S with num < b <= c, num !| b+c
        # 2. num as b: For all a in S with a < num, and c in S with num <= c (including c=num? No, c distinct if b<c, but b=c=num case is pairwise)
        #    Actually, if num is b, and c=num, then a | 2num. This is covered by pairwise check for a,num.
        #    If num is b, c > num: a | num+c.
        # 3. num as c: For all a,b in S with a < b <= num.
        #    If b=num, a | b+num = 2num. Covered by pairwise.
        #    If b < num, a | b+num.
        
        # Let's just use the full check for simplicity in the inner loop, 
        # but optimize by pre-sorting or just checking against the list.
        
        candidate_S = S | {num}
        if is_valid_set(candidate_S):
            S.add(num)
            
    best_S = S
    best_len = len(S)
    
    # Local Search Improvement
    # Try to add elements that were skipped, by removing some others?
    # Or just try random swaps to increase size.
    
    # A simple hill climbing: 
    # 1. Try to add any missing number.
    # 2. If none can be added, try swapping one in with one out to allow adding a third?
    #    Or just swap to a configuration that allows more additions later.
    
    # Given time constraints, a randomized restart with different greedy orders might be better.
    # Or, since we want max cardinality, we can try to "repair" the set.
    
    # Let's try a few random permutations of greedy order
    best_S = set()
    best_len = 0
    
    for _ in range(50): # 50 restarts
        S = set()
        # Randomize the order of candidates? 
        # Or just vary the threshold.
        # Let's try greedy from small to large, but with random tie-breaking or order.
        # Actually, large-to-small is usually better for "divisor" constraints.
        # Let's try a mix.
        
        # Strategy: Pick numbers randomly, but prefer those that are "safe".
        # Simpler: Just run the large-to-small greedy, then do a local search.
        
        S = set()
        for num in range(n, 0, -1):
            if is_valid_set(S | {num}):
                S.add(num)
        
        if len(S) > best_len:
            best_len = len(S)
            best_S = S.copy()
            
        # Local Search on S
        improved = True
        while improved:
            improved = False
            # Try to add any missing number
            for num in range(1, n+1):
                if num not in S:
                    if is_valid_set(S | {num}):
                        S.add(num)
                        improved = True
                        break # Restart loop as S changed
            
            # If no single addition works, try swaps?
            # Swapping is harder to guarantee improvement.
            # Let's try removing one element and adding two.
            if not improved and len(S) > 0:
                # This is expensive. Skip for now, or limit it.
                pass

    return sorted(best_S)
# EVOLVE-BLOCK-END
