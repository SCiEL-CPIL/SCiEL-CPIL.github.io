"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    self_inv = [False] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if x * x % p == 1:
            self_inv[x] = True
    
    # Self-inverse: x * x % p == 1 => x = 1 or x = p-1 = 1008
    # So 1 and 1008 can never be in a valid interval
    
    # Build conflict map: for each position i, find the nearest conflict partner
    # A conflict pair is (x, inv[x]) where x != inv[x]
    # For a valid interval [s, e], we cannot have both x and inv[x] in it
    
    # Strategy: For each element x in [2, 1007] (excluding self-inverses 1 and 1008),
    # its inverse partner inv[x] creates a constraint. If x is in the interval, 
    # inv[x] cannot be in the interval.
    
    # Precompute for each position, the "barrier" that prevents extending left or right
    # left_barrier[i] = smallest s such that [s, i] is valid (considering only conflicts with elements <= i)
    # Actually, let's use a different approach:
    
    # For each start s, the maximal end e is determined by: for all x in [s, e], inv[x] not in [s, e]
    # This means: min over x in [s,e] of (inv[x] if inv[x] >= s else infinity) > e
    # And: max over x in [s,e] of (inv[x] if inv[x] <= e else -infinity) < s
    
    # More precisely: the interval [s,e] is valid iff for all x in [s,e], inv[x] < s or inv[x] > e
    
    # Let's precompute: for each x, if inv[x] > x, then the pair (x, inv[x]) means that
    # an interval containing x cannot extend to inv[x] or beyond (if it also contains inv[x])
    # And an interval containing inv[x] cannot start before x (if it also contains x)
    
    # Better approach: use a sweep. For each possible start s, find the maximum valid end.
    # The constraint is: for all x in [s, e], inv[x] not in [s, e].
    # Equivalently: min_{x in [s,e]} (inv[x] if inv[x] >= s) > e  (no inverse partner to the right within interval)
    # AND: max_{x in [s,e]} (inv[x] if inv[x] <= e) < s  (no inverse partner to the left within interval)
    
    # Let's define: for each x, partner[x] = inv[x]
    # An interval [s,e] is valid iff for all x in [s,e]: partner[x] < s or partner[x] > e
    
    # This is equivalent to: 
    #   For all x in [s,e] with partner[x] >= s: partner[x] > e
    #   For all x in [s,e] with partner[x] <= e: partner[x] < s
    
    # The first condition: min over x in [s,e] where partner[x] >= s of partner[x] > e
    # The second condition: max over x in [s,e] where partner[x] <= e of partner[x] < s
    
    # Let's precompute arrays to make this efficient.
    
    # For a fixed s, as we extend e from s upward, we need to track:
    # - The minimum partner[x] among x in [s, e] where partner[x] >= s
    # - The maximum partner[x] among x in [s, e] where partner[x] <= e (this changes with e)
    
    # Actually, let's think about it differently. The interval [s,e] is invalid iff there exists
    # x in [s,e] such that partner[x] is also in [s,e].
    
    # For each pair (x, partner[x]) with x < partner[x], the interval [s,e] is invalid if
    # s <= x and partner[x] <= e.
    
    # So for a fixed s, the maximum valid e is: the minimum over all pairs (x, partner[x]) with x >= s
    # of (partner[x] - 1), but only if x is in [s, e]... this is circular.
    
    # Let me just do the direct approach but with better pruning.
    
    best_start = 2
    best_len = 1
    
    # For each start s, find max valid end e
    for s in range(2, 1009):  # skip 1 (self-inverse)
        if time.time() > deadline:
            break
        
        # Pruning: if the max possible length from s is not better than best_len, skip
        if 1008 - s + 1 <= best_len:
            continue
        
        # Find the maximum valid end for start s
        # An interval [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e
        
        # Let's track the minimum "right barrier" and maximum "left barrier"
        # right_barrier = min over x in [s, current_e] of inv[x] (if inv[x] >= s)
        # left_barrier = max over x in [s, current_e] of inv[x] (if inv[x] <= current_e)
        
        # Start with e = s
        # Check if s is self-inverse
        if self_inv[s]:
            continue
        
        # For start s, the first element is s. inv[s] must be < s or > e.
        # Since we're extending e, initially e = s, so inv[s] must be < s or > s.
        # If inv[s] == s, it's self-inverse (already handled).
        
        # Let's track: min_right = minimum inv[x] for x in [s, e] where inv[x] >= s
        #              max_left = maximum inv[x] for x in [s, e] where inv[x] <= e
        
        # For e to be valid: min_right > e AND max_left < s
        
        # Initialize
        min_right = inv[s] if inv[s] >= s else float('inf')
        # max_left: inv[s] <= s? If inv[s] < s, then max_left could be inv[s]
        max_left = inv[s] if inv[s] <= s else -1
        
        # Check if [s, s] is valid: min_right > s AND max_left < s
        # min_right > s: if inv[s] >= s, then inv[s] > s (since inv[s] != s for non-self-inverse)
        #   So min_right = inv[s] > s. OK.
        #   If inv[s] < s, then min_right = inf > s. OK.
        # max_left < s: if inv[s] <= s, then max_left = inv[s] < s (since inv[s] != s). OK.
        #   If inv[s] > s, then max_left = -1 < s. OK.
        # So [s, s] is always valid for non-self-inverse s.
        
        e = s
        while e < 1008:
            if time.time() > deadline:
                break
            
            new_e = e + 1
            
            # Check if new_e is self-inverse
            if self_inv[new_e]:
                break
            
            # Update barriers with new_e
            new_inv = inv[new_e]
            
            # Update min_right: if new_inv >= s, consider it
            if new_inv >= s:
                if new_inv < min_right:
                    min_right = new_inv
            
            # Update max_left: if new_inv <= new_e, consider it
            # Note: we need to check if new_inv <= new_e
            if new_inv <= new_e:
                if new_inv > max_left:
                    max_left = new_inv
            
            # Check validity of [s, new_e]: min_right > new_e AND max_left < s
            if min_right > new_e and max_left < s:
                e = new_e
            else:
                break
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
