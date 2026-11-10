"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008

    # Precompute modular inverses for 1 to 1008
    inv = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Self-inverse residues: 1 and 1008 (since 1*1=1 and 1008*1008=1 mod 1009)
    # These cannot appear in any valid interval.
    # So valid intervals are within [2, 1007].
    
    # For each x in [2, 1007], determine the constraint it imposes:
    # - If inv[x] > x: including x means the interval's right end must be < inv[x]
    #   (because inv[x] would be in the interval if right >= inv[x], creating a pair)
    # - If inv[x] < x: including x means the interval's left start must be > inv[x]
    #   (because inv[x] would be in the interval if left <= inv[x], creating a pair)
    # - inv[x] == x: self-inverse, cannot be included (but we already exclude 1 and 1008)
    
    # Precompute for each x:
    #   max_right[x] = inv[x] - 1 if inv[x] > x, else MAX_VAL (no constraint on right)
    #   min_left[x] = inv[x] + 1 if inv[x] < x, else 1 (no constraint on left)
    
    max_right = [0] * (MAX_VAL + 1)
    min_left = [0] * (MAX_VAL + 1)
    
    for x in range(2, MAX_VAL):  # 2 to 1007
        ix = inv[x]
        if ix > x:
            max_right[x] = ix - 1  # right end must be <= ix-1
        else:
            max_right[x] = MAX_VAL  # no constraint on right from this x
            
        if ix < x:
            min_left[x] = ix + 1  # left start must be >= ix+1
        else:
            min_left[x] = 1  # no constraint on left from this x
    
    # Now use a sliding window / two-pointer approach:
    # For each left boundary s, find the maximum right boundary e such that
    # for all x in [s, e]: e <= max_right[x] and s >= min_left[x]
    # 
    # Equivalently: e <= min(max_right[x] for x in [s,e]) and s >= max(min_left[x] for x in [s,e])
    #
    # Since we're looking for the longest valid interval, we can iterate s from 2 to 1007,
    # and for each s, find the maximum e such that the interval [s,e] is valid.
    #
    # Key observation: as s increases, the constraint from min_left relaxes (fewer elements
    # in the window with min_left > s). But max_right constraints tighten as we include more elements.
    #
    # Actually, let me think differently. For a fixed s, the valid e must satisfy:
    #   e <= min(max_right[x] for x in [s, e])
    # This is a self-referential constraint. But we can greedily extend:
    #   Start with e = s. Check if max_right[s] >= s (always true since max_right[s] >= s when inv[s] > s, 
    #   and max_right[s] = MAX_VAL otherwise).
    #   Then try e = s+1: need max_right[s+1] >= s+1 (always) AND for all x in [s, s+1], e <= max_right[x].
    #   i.e., s+1 <= max_right[x] for all x in [s, s+1].
    #
    # So the greedy approach: for each s, extend e as far as possible while maintaining
    # e <= max_right[x] for all x in [s, e].
    #
    # This is still O(n²) in worst case. Let me think of a better approach.
    #
    # Alternative: For each possible right boundary e, find the minimum valid left boundary s.
    # [s, e] is valid iff for all x in [s, e]: s >= min_left[x] and e <= max_right[x].
    # So s >= max(min_left[x] for x in [s, e]) and e <= min(max_right[x] for x in [s, e]).
    #
    # For a fixed e, we need the largest s such that:
    #   s >= max(min_left[x] for x in [s, e])  ... (*)
    #   e <= min(max_right[x] for x in [s, e])  ... (**)
    #
    # Constraint (**) means: for all x in [s, e], max_right[x] >= e.
    # So s must be greater than any x < s that has max_right[x] < e.
    # Equivalently, s > max({x : x < s, max_right[x] < e}) = max({x : 2 <= x < s, max_right[x] < e})
    #
    # Let me define: for each e, let forbidden_left(e) = max({x : 2 <= x <= e, max_right[x] < e}, default=1)
    # Then s must be > forbidden_left(e), so s >= forbidden_left(e) + 1.
    #
    # But we also need s >= max(min_left[x] for x in [s, e]).
    #
    # This is getting complex. Let me just use the O(n²) greedy but with early termination,
    # since n=1006 is small enough.
    
    best_start = 2
    best_length = 0
    
    # For each start s from 2 to 1007, greedily extend e to the right.
    # The interval [s, e] is valid iff for all x in [s, e]:
    #   - If inv[x] > x: e < inv[x]  (i.e., e <= max_right[x])
    #   - If inv[x] < x: s > inv[x]  (i.e., s >= min_left[x])
    #
    # When extending from e to e+1:
    #   New element e+1 imposes: if inv[e+1] > e+1, then we need e+1 <= max_right[e+1] (always true)
    #   But also, for all existing x in [s, e], we still need e+1 <= max_right[x].
    #   So the constraint is: e+1 <= min(max_right[x] for x in [s, e+1])
    #   i.e., e+1 <= max_right[e+1] AND e+1 <= min(max_right[x] for x in [s, e])
    #
    # Let me track min_max_right = min(max_right[x] for x in [s, current_e]) as we extend.
    
    for s in range(2, MAX_VAL):  # s from 2 to 1007
        if time.time() > deadline:
            break
        
        # Check if s itself is valid: s >= min_left[s]
        if s < min_left[s]:
            continue
        
        e = s
        # Track the minimum max_right in the current window [s, e]
        min_mr = max_right[s]
        
        # Try to extend the interval to the right
        while e < MAX_VAL - 1:  # e can go up to 1007
            next_e = e + 1
            # Check if next_e can be added:
            # 1. next_e <= max_right[next_e] (always true by construction)
            # 2. next_e <= min_mr (the min max_right of existing elements)
            # 3. s >= min_left[next_e] (the new element's left constraint)
            
            if next_e <= min_mr and s >= min_left[next_e]:
                e = next_e
                # Update min_mr
                if max_right[e] < min_mr:
                    min_mr = max_right[e]
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    # Fallback if no valid interval found (should not happen as [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
# EVOLVE-BLOCK-END
