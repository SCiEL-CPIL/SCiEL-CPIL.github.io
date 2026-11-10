"""Find the longest interval in [1,1008] with no modular inverse pair modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    p = 1009
    
    # Precompute modular inverses: inv[x] = y such that x*y % p == 1
    # For x in 1..1008, inv[x] exists since p is prime
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For an interval [start, start+length-1] to be valid:
    # For every x, y in the interval (including x=y), x*y % p != 1
    # This means: for every x in the interval, inv[x] must NOT be in the interval
    # (because if inv[x] is in the interval, then x * inv[x] % p == 1)
    
    # So a position x "blocks" any interval that contains both x and inv[x].
    # Note: if x == inv[x] (self-inverse), then x blocks any interval containing x.
    # Self-inverses: x^2 % p == 1, so x = 1 or x = p-1 = 1008.
    
    # Strategy: For each possible start, find the maximum length such that
    # no x in [start, start+len-1] has inv[x] also in [start, start+len-1].
    
    # An interval [s, e] is invalid if there exists x in [s,e] with inv[x] in [s,e].
    # For a given start s, the interval [s, e] is valid iff for all x in [s,e],
    # inv[x] < s or inv[x] > e.
    
    # For a fixed start s, as we extend e, the first time we hit an invalidity
    # is when some x in [s, e] has inv[x] in [s, e].
    # The constraint from x is: e < inv[x] if inv[x] >= s, i.e., e <= inv[x] - 1
    # (when inv[x] >= s). If inv[x] < s, no constraint on e from x.
    
    # So for start s, the maximum valid e is:
    # min over all x in [s, 1008] with inv[x] >= s of (inv[x] - 1)
    # but also e >= s (nonempty), and e <= 1008.
    # Actually we need e such that for ALL x in [s, e], inv[x] is outside [s, e].
    # This is a fixed-point condition. Let's think differently.
    
    # For a given start s, define for each x >= s: if inv[x] >= s, then x can only
    # be included up to e < inv[x], i.e., e <= inv[x] - 1.
    # The maximum valid e for start s is the largest e such that for all x in [s, e],
    # inv[x] < s or inv[x] > e.
    
    # Let's just do it directly: for each start s, extend e from s upward,
    # checking validity incrementally. But that's O(n^2) worst case.
    
    # Better: for each start s, the maximum e is determined by the minimum
    # "right boundary" imposed by elements in [s, e]. This is a classic problem.
    
    # Let's precompute: for each position x, its "conflict partner" is inv[x].
    # An interval [s, e] is valid iff no pair (x, inv[x]) with x != inv[x] has both
    # in [s,e], AND no self-inverse (1, 1008) is in [s,e].
    
    # For a fixed start s, the maximum valid end e:
    # We need to find the largest e such that for all x in [s, e], inv[x] not in [s, e].
    # Equivalently, e < min{inv[x] : x in [s,e], inv[x] >= s} (if such x exists).
    
    # Let's compute this by scanning. For each s, we can compute the answer.
    # Since n=1008, O(n^2) is fine (about 10^6 operations).
    
    best_start = 1
    best_len = 1
    
    for s in range(1, p):  # s from 1 to 1008
        # Find maximum e >= s such that [s, e] is valid
        # Start with e = s, check if valid, extend if possible
        # A cleaner way: the constraint is that for all x in [s, e], inv[x] < s or inv[x] > e.
        # Let's find the max valid e by considering all x >= s.
        # For x in [s, 1008], if inv[x] >= s, then e must be < inv[x] (i.e., e <= inv[x]-1)
        # OR x > e (x not in interval).
        # 
        # Let's just iterate e from s to 1008 and check validity incrementally.
        # But we can be smarter: the first x in [s, e] that causes a problem limits e.
        
        # For start s, the limiting factor is: among all x >= s, if x is in [s,e],
        # we need inv[x] < s or inv[x] > e.
        # The tightest constraint: e_max = min(inv[x] - 1 for x in [s, e_max] if inv[x] >= s)
        # This is circular. Let's just do a direct check.
        
        # Direct approach: for start s, the maximum valid end is:
        # e_max = 1008 initially
        # For each x in [s, 1008]: if inv[x] >= s and inv[x] <= e_max, then e_max = min(e_max, inv[x]-1)
        # But we also need x <= e_max for x to be in the interval... this is getting circular.
        
        # Simplest correct approach: binary search or linear scan for each s.
        # For each s, try e from s to 1008, and check if [s,e] is valid.
        # [s,e] is valid iff for all x in [s,e], inv[x] < s or inv[x] > e.
        
        # Even simpler: precompute a "conflict" array.
        # For interval [s, e], it's invalid if there exists x in [s,e] with s <= inv[x] <= e.
        
        # Let's use the following: for each s, the max valid e is the largest e such that
        # max(inv[x] for x in [s,e] if inv[x] >= s) > e  OR  no such x exists.
        # Equivalently, for all x in [s,e] with inv[x] >= s, we need inv[x] > e.
        
        # Let's just do O(n) per start:
        e_max = p - 1  # 1008
        # The interval [s, e] is valid iff for all x in [s, e], inv[x] < s or inv[x] > e.
        # Let's find the smallest e that makes it invalid, then answer is e-1.
        
        # For start s, scan x from s to 1008:
        # If x is in the candidate interval and inv[x] is also in [s, e], invalid.
        
        # Let me just do it directly and efficiently:
        # For start s, the constraint from element x (where x >= s) is:
        #   if inv[x] >= s: then e must satisfy e < inv[x] OR x > e
        #   i.e., if x <= e, then we need inv[x] > e or inv[x] < s
        #   since we're considering inv[x] >= s, we need inv[x] > e
        # So for x in [s, e]: inv[x] > e (when inv[x] >= s)
        
        # The maximum valid e for start s:
        # Let's collect all "upper bounds": for each x >= s with inv[x] >= s,
        # if x is included (x <= e), then e < inv[x].
        # So e < min(inv[x] for x in [s, e] with inv[x] >= s)
        
        # Iterative approach:
        e = p - 1
        changed = True
        while changed:
            changed = False
            for x in range(s, e + 1):
                if inv[x] >= s and inv[x] <= e:
                    # x and inv[x] are both in [s, e], invalid
                    e = inv[x] - 1
                    changed = True
                    if e < s:
                        break
            # Re-check after reducing e
            for x in range(s, e + 1):
                if inv[x] >= s and inv[x] <= e:
                    e = min(e, inv[x] - 1)
                    changed = True
                    if e < s:
                        break
        
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
        
        if time.time() > deadline:
            break
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
