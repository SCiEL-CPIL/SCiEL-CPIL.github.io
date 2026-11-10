"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    p = 1009
    
    # 1. Precompute forbidden pairs
    # For an interval to be valid, it cannot contain any pair (x, y) such that x*y % p == 1.
    # This includes x=y (i.e., x^2 % p == 1).
    
    # We can model this as: an interval is valid if it contains NO pair from the set of forbidden pairs.
    # However, it's easier to think about "breakpoints".
    # If x is in the interval, then x_inv (the modular inverse of x) cannot be in the interval.
    
    # Let's precompute the inverse for each x in [1, 1008].
    # inv[x] = y such that x*y % 1009 == 1.
    inv = [0] * (p + 1)
    for x in range(1, p):
        for y in range(1, p):
            if (x * y) % p == 1:
                inv[x] = y
                break
        # Optimization: use pow(x, p-2, p) for prime p
        # Since 1009 is prime, inv[x] = pow(x, p-2, p)
    
    # Actually, pow is much faster and guaranteed to work for prime p.
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # 2. Find the longest interval [s, e] such that for all x in [s, e], inv[x] is NOT in [s, e].
    # This is equivalent to saying that the interval [s, e] does not contain any "conflict pair".
    # A conflict pair is {x, inv[x]}. Note that if x == inv[x], it's a self-conflict.
    # Self-conflicts occur when x^2 % p == 1 => x=1 or x=p-1 (1008).
    # So 1 and 1008 cannot be in ANY valid interval.
    
    # Let's create a list of all forbidden pairs (min(x, inv[x]), max(x, inv[x])).
    # If an interval contains both elements of any such pair, it's invalid.
    # If an interval contains a self-inverse element (1 or 1008), it's invalid.
    
    # We can use a sliding window or a simple O(N^2) search since N=1008 is small.
    # N=1008, so N^2 is ~10^6, which is very fast.
    
    best_len = 0
    best_start = 1
    best_end = 1
    
    # Precompute a set of "bad" pairs for quick lookup? 
    # Or just check directly.
    # For a given interval [s, e], it is valid iff:
    # 1. 1 is not in [s, e]
    # 2. 1008 is not in [s, e]
    # 3. For all x in [s, e], inv[x] is not in [s, e].
    
    # Since 1 and 1008 are self-inverses, condition 3 covers them if we check x=1 and x=1008.
    # If s <= 1 <= e, then x=1 is in interval. inv[1]=1. 1 is in interval. Conflict.
    # So we just need to check condition 3 for all x in [s, e].
    
    # To speed up, we can precompute for each x, the "range of influence" or just use the O(N^2) loop.
    # Given N=1008, O(N^2) is ~1 million iterations. Python can do ~10-50 million simple ops per second.
    # So this will take < 0.1 seconds.
    
    for s in range(1, p):
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
            
        # We can optimize the inner loop.
        # As we extend e from s to p-1, we can keep track of if the interval is valid.
        # But it's easier to just check each (s, e) pair.
        
        # Actually, for a fixed s, as e increases, once it becomes invalid, it stays invalid?
        # No. If [s, e] is invalid because x and inv[x] are in it, then [s, e+1] is also invalid.
        # So for a fixed s, the set of valid e's is a prefix [s, max_e].
        # We can find the max valid e for each s.
        
        valid = True
        max_e = s - 1
        
        for e in range(s, p):
            # Check if adding e breaks the interval
            # We need to check if e conflicts with any x in [s, e-1]
            # Or if e is a self-conflict (e == inv[e])
            
            # Since we are extending, we only need to check if e conflicts with existing elements.
            # But we don't have a set of existing elements easily accessible in O(1) for "does any x in [s, e-1] have inv[x] == e?"
            # This is equivalent to: does inv[e] exist in [s, e-1]?
            # Note: if inv[e] == e, then e is a self-conflict.
            # If inv[e] < s, then no conflict with previous elements (since they are >= s).
            # If s <= inv[e] < e, then conflict.
            # If inv[e] > e, then no conflict with previous elements (since they are <= e-1).
            # Wait, if inv[e] > e, it's not in [s, e-1]. So no conflict yet.
            # But when we add inv[e] later, it will conflict with e.
            
            # So, for the interval [s, e] to be valid, we must ensure that for ALL x in [s, e], inv[x] is NOT in [s, e].
            # This is a global property.
            
            # Let's stick to the simple O(N^2) check but optimize it.
            # For a fixed s, we can iterate e and maintain a boolean "is_valid".
            # Actually, the "prefix" property holds: if [s, e] is invalid, [s, e+1] is invalid.
            # So we can break early for a fixed s once we find an invalid e.
            
            # How to check if [s, e] is valid efficiently?
            # We can precompute for each e, the "minimum s" such that [s, e] is valid? No.
            
            # Let's just do the O(N^2) with early termination.
            # For fixed s:
            #   for e from s to p-1:
            #      check if [s, e] is valid.
            #      if valid, update best.
            #      if invalid, break (since larger e will also be invalid).
            
            # How to check validity of [s, e] in O(1) or O(log N)?
            # It is valid iff min_{x in [s, e]} (something)?
            
            # Let's define conflict(x, y) = True if x*y % p == 1.
            # Interval [s, e] is valid iff for all x, y in [s, e], not conflict(x, y).
            
            # We can precompute a 2D prefix sum of conflicts? N=1008, N^2 = 10^6.
            # conflict_matrix[x][y] = 1 if x*y % p == 1 else 0.
            # Then sum of conflicts in [s, e] x [s, e] can be found with 2D prefix sums in O(1).
            # If sum == 0, it's valid.
            
            # This is a great approach!
            # Precomputation: O(N^2)
            # Query: O(1)
            # Total: O(N^2)
            
            pass # We'll implement this below
            
        # End for e
        
    # Let's restart with the 2D prefix sum approach, it's cleaner and faster.
    
    N = p - 1 # 1008
    # conflict[x][y] is 1 if x*y % p == 1
    # We only care about x, y in [1, 1008]
    
    # To save memory and time, we can just use the property:
    # For a fixed x, there is only ONE y such that x*y % p == 1. That is y = inv[x].
    # So conflict(x, y) is 1 iff y == inv[x].
    
    # So the number of conflicts in [s, e] is the number of x in [s, e] such that inv[x] is also in [s, e].
    
    # We can precompute an array "conflict_count" for each possible interval? No, too big.
    # But we can use the early termination property for fixed s.
    
    # For fixed s, as e increases, the set of x in [s, e] grows.
    # An interval [s, e] becomes invalid if there exists x in [s, e] such that inv[x] is in [s, e].
    # Let's define for each x, the "partner" inv[x].
    # If x is in [s, e], we must ensure inv[x] is NOT in [s, e].
    # This means inv[x] < s or inv[x] > e.
    
    # So for a fixed s, the maximum valid e is limited by the "closest" partner that falls inside the interval.
    # Specifically, for each x >= s, if inv[x] >= s, then e must be < inv[x] (if inv[x] > x) or e must be < x (if inv[x] < x, but x is already in interval, so this is a self-conflict or conflict with smaller element).
    
    # Actually, simpler:
    # For a fixed s, let's find the minimum "blocking" value.
    # An element x in [s, e] blocks the interval if its partner inv[x] is also in [s, e].
    # This happens if s <= inv[x] <= e.
    # So, for a fixed s, the interval [s, e] is valid iff for ALL x in [s, e], inv[x] < s OR inv[x] > e.
    
    # This implies that e must be less than min{ inv[x] | x in [s, e] and inv[x] >= s }.
    # Let M(s, e) = min{ inv[x] | x in [s, e] and inv[x] >= s }.
    # Then [s, e] is valid iff e < M(s, e).
    
    # This is a bit circular. Let's just iterate e for each s and maintain the "minimum partner in range".
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        if time.time() > deadline - 2.0:
            break
            
        # We need to find the max e such that [s, e] is valid.
        # Let's maintain a variable `min_partner_in_range` which is the minimum inv[x] for x in [s, e] such that inv[x] >= s.
        # Actually, if inv[x] < s, it doesn't block e (as long as e < inv[x] is not required, wait).
        # If inv[x] < s, then inv[x] is NOT in [s, e]. So x does not block e.
        # If inv[x] >= s, then for x to not block e, we must have inv[x] > e.
        # So e must be < inv[x].
        
        # So, for a fixed s, let's find the smallest inv[x] among all x >= s such that inv[x] >= s.
        # Let min_inv = min{ inv[x] | x >= s and inv[x] >= s }.
        # Then for any e < min_inv, is the interval [s, e] valid?
        # Not necessarily. We must ensure that for ALL x in [s, e], inv[x] > e OR inv[x] < s.
        # If we pick e < min_inv, then for any x in [s, e], if inv[x] >= s, then inv[x] >= min_inv > e.
        # So inv[x] > e. Thus no conflict.
        
        # So the maximum valid e for a given s is min_inv - 1.
        # Wait, we also need to check if x itself is a self-inverse (inv[x] == x).
        # If inv[x] == x, then min_inv will be x (if x >= s).
        # Then e must be < x. But x is in [s, e] if e >= x.
        # So if e < x, x is not in the interval.
        # But we are iterating x >= s.
        
        # Let's refine:
        # For a fixed s, we want the largest e such that for all x in [s, e], inv[x] is not in [s, e].
        # This is equivalent to: e < min{ inv[x] | x in [s, e] and inv[x] >= s }.
        
        # Let's just iterate e and keep track of the minimum inv[x] seen so far that is >= s.
        
        min_inv_so_far = p # infinity
        max_valid_e = s - 1
        
        for e in range(s, p):
            # Check if adding e is valid
            # We need inv[e] to not be in [s, e].
            # inv[e] is in [s, e] iff s <= inv[e] <= e.
            
            # Also, we need to ensure that previous x in [s, e-1] don't conflict with e.
            # This is symmetric. If x in [s, e-1] conflicts with e, then inv[x] = e.
            # This means e is the partner of x.
            # Since we are increasing e, we can just check if inv[e] is in [s, e-1].
            # And if inv[e] == e (self-conflict).
            
            if inv[e] >= s and inv[e] <= e:
                # Conflict found. [s, e] is invalid.
                # And any larger e will also be invalid?
                # Yes, because the conflict (x, e) remains in the interval.
                break
            else:
                # No conflict with e and previous elements.
                # So [s, e] is valid.
                max_valid_e = e
                if e - s + 1 > best_len:
                    best_len = e - s + 1
                    best_start = s
                    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
