"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MOD = 1009
    
    # Precompute modular inverses
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each position x in [1, 1008], determine the "constraint window" it imposes.
    # If x is in the interval [s, e], then inv[x] must NOT be in [s, e].
    # This means:
    #   - If inv[x] > x: then e < inv[x], i.e., e <= inv[x] - 1
    #   - If inv[x] < x: then s > inv[x], i.e., s >= inv[x] + 1
    #   - If inv[x] == x: x is self-inverse (only 1 and 1008), so x cannot be in any valid interval.
    
    # For a given start s, we want to find the maximum e such that for all x in [s, e]:
    #   - x is not self-inverse (x != 1, x != 1008)
    #   - inv[x] < s or inv[x] > e
    
    # Let's precompute for each x the "max_e_allowed_by_x" and "min_s_allowed_by_x".
    # Actually, let's think of it differently.
    # For a fixed s, the constraint from each x in [s, e] is:
    #   - If inv[x] >= s: then we need inv[x] > e, so e < inv[x], i.e., e <= inv[x] - 1
    #   - If inv[x] < s: no constraint on e from this x (since inv[x] is already outside [s, e])
    
    # So for a fixed s, the maximum valid e is the minimum over all x in [s, e] of:
    #   - inv[x] - 1 if inv[x] >= s
    #   - infinity if inv[x] < s
    # But e must also be at least s, and x ranges from s to e.
    
    # This is a bit circular. Let's use a different approach.
    # For each possible start s (from 2 to 1007), compute the maximum valid end e.
    
    # Key insight: for a fixed s, as we extend e from s to 1007, the interval becomes invalid
    # at the first point where some x in [s, e] has inv[x] in [s, e].
    # 
    # We can precompute for each x: if inv[x] >= x, then having x in the interval requires
    # e < inv[x]. If inv[x] < x, having x requires s > inv[x].
    
    # Let's precompute an array max_end[x] = the maximum e such that x can be in [s, e] 
    # given that s is fixed... but s varies.
    
    # Alternative: for each x, define:
    #   - upper_bound[x] = inv[x] - 1 if inv[x] > x else 1008  (if inv[x] is to the right, e must be < inv[x])
    #   - lower_bound[x] = inv[x] + 1 if inv[x] < x else 1     (if inv[x] is to the left, s must be > inv[x])
    #   - If inv[x] == x, x is forbidden.
    
    upper_bound = [1008] * MOD
    lower_bound = [1] * MOD
    for x in range(1, MOD):
        if inv[x] == x:
            upper_bound[x] = 0  # forbidden
            lower_bound[x] = MOD
        elif inv[x] > x:
            upper_bound[x] = inv[x] - 1
        elif inv[x] < x:
            lower_bound[x] = inv[x] + 1
    
    # Now, for an interval [s, e] to be valid:
    #   - For all x in [s, e]: x is not forbidden (upper_bound[x] > 0 and lower_bound[x] < MOD)
    #   - For all x in [s, e]: e <= upper_bound[x] and s >= lower_bound[x]
    #   - Equivalently: e <= min(upper_bound[x] for x in [s,e]) and s >= max(lower_bound[x] for x in [s,e])
    
    # For a fixed s, the maximum valid e is the largest e such that:
    #   - For all x in [s, e]: upper_bound[x] >= e and lower_bound[x] <= s
    
    # Let's iterate over s and find the max valid e efficiently.
    # For each s, start with e = s and extend while possible.
    
    best_start = 2
    best_length = 0
    
    for s in range(2, 1008):
        if time.time() > deadline:
            break
        
        # Check if s itself is valid (not self-inverse)
        if upper_bound[s] < s or lower_bound[s] > s:
            continue
        
        e = s
        # Extend e as far as possible
        while e < 1007:
            if time.time() > deadline:
                break
            next_e = e + 1
            # Check if next_e can be added
            if upper_bound[next_e] >= next_e and lower_bound[next_e] <= s:
                # Also need to check that all previous x in [s, e] still satisfy upper_bound[x] >= next_e
                # But that's expensive. Let's check differently.
                # The constraint is: for all x in [s, next_e], upper_bound[x] >= next_e and lower_bound[x] <= s.
                # We already know lower_bound[x] <= s for x in [s, e] (since the interval was valid up to e).
                # We need to check: for all x in [s, e], upper_bound[x] >= next_e.
                # And for x = next_e: upper_bound[next_e] >= next_e and lower_bound[next_e] <= s.
                
                # To avoid O(n) check each time, let's precompute a segment tree or just do the check.
                # Given n=1008, even O(n^2) is ~1M operations, which is fast.
                
                # Let's just check if the interval [s, next_e] is valid by checking all x in [s, next_e].
                # But that's O(n) per step, making the whole thing O(n^3) in worst case.
                # For n=1008, that's ~10^9, which might be too slow.
                
                # Better approach: maintain the minimum upper_bound in [s, e] as we extend.
                # If min(upper_bound[x] for x in [s, e]) >= next_e, and upper_bound[next_e] >= next_e,
                # and lower_bound[next_e] <= s, then [s, next_e] is valid.
                
                # Let's track min_upper as we go.
                pass
            else:
                break
        
        # Simpler: just do a direct check for each (s, e) pair, but break early.
        # For n=1008, the total number of (s, e) pairs is ~500K, and each check is O(1) if we precompute.
        
        # Let's precompute prefix minimums of upper_bound for each starting point.
        # Actually, let's just do the brute force with early termination.
        
        e = s
        min_upper = upper_bound[s]
        while e < 1007:
            if time.time() > deadline:
                break
            next_e = e + 1
            # Check if we can extend to next_e
            # Need: min(upper_bound[x] for x in [s, next_e]) >= next_e
            #      and lower_bound[next_e] <= s
            #      and upper_bound[next_e] >= next_e
            #      and for all x in [s, e], lower_bound[x] <= s (already ensured)
            
            if lower_bound[next_e] <= s and upper_bound[next_e] >= next_e:
                new_min_upper = min(min_upper, upper_bound[next_e])
                if new_min_upper >= next_e:
                    e = next_e
                    min_upper = new_min_upper
                else:
                    break
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
