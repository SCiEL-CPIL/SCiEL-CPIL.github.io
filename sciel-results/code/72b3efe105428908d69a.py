import time

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 ≡ 1 (mod 1009)
    # For prime p, x^2 ≡ 1 means x ≡ 1 or x ≡ p-1
    self_inv = {1, MOD - 1}
    
    # For each position i, find the smallest j > i such that
    # j is a self-inverse OR inv[j] is in [i, j-1]
    # This gives us the "next obstruction" after i
    # 
    # An interval [s, e] is valid iff for all x in [s,e]:
    #   - x is not self-inverse
    #   - inv[x] is not in [s, e]
    # 
    # Key insight: if we fix start s, the maximum valid end is determined by
    # the first obstruction. As s increases, the valid end can only increase or stay same
    # (monotonicity), so we can use a two-pointer approach.
    
    # For a given start s, the interval [s, e] is valid iff:
    # 1. No element in [s,e] is self-inverse
    # 2. For no x in [s,e] is inv[x] in [s,e]
    
    # Let's precompute: for each s, what's the maximum valid end?
    # We'll use a sweep approach.
    
    # For each position i (1 to N), define:
    # - is_self_inv[i] = 1 if i is self-inverse
    # - For the inverse constraint: if i is in interval [s,e], then inv[i] must NOT be in [s,e]
    
    # Two-pointer: start = 1, end = 1
    # Maintain a set of elements in [start, end]
    # When we can't extend end, increment start
    
    start = 1
    end = 1
    best_start = 1
    best_length = 0
    
    # We need to track which elements are in [start, end]
    # and check if adding end+1 would create a conflict
    
    # An element x can be in the interval iff:
    # - x is not self-inverse
    # - inv[x] is not in the interval
    
    # Let's use a different approach: for each start s, find the max valid end
    # by checking constraints.
    
    # For start s, the interval [s, e] is valid iff:
    # - s is not self-inverse
    # - For all x in [s, e]: inv[x] < s or inv[x] > e
    
    # This means: for all x in [s, e], inv[x] is NOT in [s, e]
    
    # Let's precompute for each pair (x, inv[x]) the constraint they impose.
    # If x and inv[x] are both in [s, e], that's a conflict.
    # So for each unordered pair {x, inv[x]} where x != inv[x], 
    # we cannot have both x and inv[x] in the interval.
    
    # This is like an interval graph problem.
    
    # Simpler approach: for each start s, greedily extend end.
    # The interval [s, e] is valid iff:
    # 1. No self-inverse in [s, e]
    # 2. For no x in [s, e] is inv[x] in [s, e]
    
    # Precompute: for each i, next_self_inv[i] = smallest self-inverse >= i, or N+1 if none
    next_si = [N + 1] * (N + 2)
    next_si[N + 1] = N + 1
    for i in range(N, 0, -1):
        if i in self_inv:
            next_si[i] = i
        else:
            next_si[i] = next_si[i + 1]
    
    # For the inverse constraint: for a given start s, as we extend end,
    # we need to ensure that for no x in [s, end] is inv[x] in [s, end].
    # 
    # Equivalently: for no x in [s, end] with inv[x] >= s and inv[x] <= end.
    # 
    # Let's define for each x: if inv[x] > x, then the pair (x, inv[x]) creates a constraint.
    # If both x and inv[x] are >= s, then we can't have both in the interval.
    # So if x is in [s, e] and inv[x] > x, we need e < inv[x].
    # Similarly if inv[x] < x, we need s > inv[x], but s is fixed.
    
    # For fixed s, the constraint from pair {x, inv[x]} is:
    # - If x >= s and inv[x] >= s: we can't have both, so if x <= e, we need inv[x] > e (or vice versa)
    # - More precisely: if x is in [s,e], then inv[x] must be < s or > e
    
    # For fixed s, the maximum valid end is:
    # e_max = min(
    #   next_si[s] - 1,  # before next self-inverse
    #   min over x in [s, e_max] of (inv[x] - 1) if inv[x] > x and inv[x] >= s
    # )
    # This is circular, so let's just do the sweep.
    
    # Two-pointer sweep:
    start = 1
    end = 0  # empty interval initially
    # We'll maintain the current interval [start, end] and try to extend end.
    
    # For efficiency, precompute for each i: 
    # if i is in interval [s, e], the constraint is that inv[i] must not be in [s, e].
    
    # Let's just do a direct O(n^2) but with early termination, which is fast enough for n=1008.
    
    best_start = 1
    best_length = 0
    
    for s in range(1, N + 1):
        # Early termination: max possible length from s is N - s + 1
        if N - s + 1 <= best_length:
            break
        
        # Check if s itself is valid (not self-inverse)
        if s in self_inv:
            continue
        
        # Try to extend from s
        e = s
        # Check all x in [s, e] for conflicts as we extend
        # For the interval [s, e] to be valid:
        # - No self-inverse in [s, e]
        # - For no x in [s, e]: inv[x] in [s, e]
        
        # Start with e = s, check validity
        # Then try e = s+1, s+2, ...
        
        valid = True
        # First check s is not self-inverse (already done)
        
        for e_candidate in range(s, N + 1):
            # Check if adding e_candidate keeps [s, e_candidate] valid
            # Need: e_candidate not self-inverse
            # And: for all x in [s, e_candidate], inv[x] not in [s, e_candidate]
            
            if e_candidate in self_inv:
                break
            
            # Check if inv[e_candidate] is in [s, e_candidate]
            ie = inv[e_candidate]
            if s <= ie <= e_candidate:
                break
            
            # Also need to check that for all x in [s, e_candidate-1], inv[x] != e_candidate
            # i.e., e_candidate is not the inverse of any x in [s, e_candidate-1]
            # inv[x] = e_candidate means x = inv[e_candidate]
            # So we need inv[e_candidate] < s or inv[e_candidate] > e_candidate
            # Which is the same check: ie = inv[e_candidate], need not (s <= ie <= e_candidate)
            # Already checked above.
            
            # Wait, we also need: for x in [s, e_candidate], inv[x] not in [s, e_candidate]
            # When we add e_candidate, we need:
            # 1. inv[e_candidate] not in [s, e_candidate]  (checked)
            # 2. For x in [s, e_candidate-1]: inv[x] != e_candidate
            #    inv[x] = e_candidate iff x = inv[e_candidate]
            #    So we need inv[e_candidate] < s (since inv[e_candidate] > e_candidate-1 is possible)
            #    Actually: inv[e_candidate] could be in [s, e_candidate-1]
            #    If inv[e_candidate] in [s, e_candidate-1], then x = inv[e_candidate] is in [s, e_candidate-1]
            #    and inv[x] = e_candidate which is in [s, e_candidate], conflict!
            #    So we need inv[e_candidate] NOT in [s, e_candidate]
            #    Which is: inv[e_candidate] < s or inv[e_candidate] > e_candidate
            #    We already check s <= ie <= e_candidate and break if true.
            #    So if we pass, ie < s or ie > e_candidate.
            #    If ie > e_candidate, that's fine for now, but when we extend further, 
            #    ie might enter the interval.
            
            e = e_candidate
            current_length = e - s + 1
            if current_length > best_length:
                best_length = current_length
                best_start = s
        
    return [best_start, best_length]
